from tqdm import tqdm
from model import GameNGen, ActionEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from config import ModelConfig, TrainingConfig
import pandas as pd
from torchvision import transforms
import os
from PIL import Image
import json
import logging
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from huggingface_hub import hf_hub_download
from peft import LoraConfig
import mlflow
import argparse


class NextFrameDataset(Dataset):
    def __init__(self, num_actions: int, metadata_path: str, frames_dir: str, image_size: tuple, history_len: int, subset_percentage: float):
        self.metadata = pd.read_csv(metadata_path)
        self.frames_dir = frames_dir
        # List files and filter out non-image files if necessary
        self.frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        # Calculate the number of frames to use based on the percentage
        num_to_use = int(len(self.frame_files) * subset_percentage)
        self.frame_files = self.frame_files[:num_to_use]
        self.metadata = self.metadata.iloc[:num_to_use]
        print(f"Using a {subset_percentage*100}% subset of the data: {len(self.frame_files)} frames.")
        self.num_actions = num_actions
        self.total_frames = len(self.frame_files)
        self.history_len = history_len

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize VAE to [-1, 1]
        ])

    def __len__(self) -> int:
        # We can't use the first `history_len` frames as they don't have enough history
        return min(len(self.metadata), self.total_frames) - self.history_len - 1

    def __getitem__(self, idx: int) -> dict:
        # We are getting the item at `idx` in our shortened dataset.
        # The actual index in the video/metadata is `idx + self.history_len`.
        actual_idx = idx + self.history_len

        history_frames = []
        for i in range(self.history_len):
            frame_idx = actual_idx - self.history_len + i
            # Use the sorted file list to get the correct frame
            img_path = os.path.join(self.frames_dir, self.frame_files[frame_idx])
            try:
                pil_image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                raise IndexError(f"Could not read history frame {frame_idx} from {img_path}.")
            history_frames.append(self.transform(pil_image))

        history_tensor = torch.stack(history_frames)

        # Get the target frame (next_frame)
        next_frame_img_path = os.path.join(self.frames_dir, self.frame_files[actual_idx])
        try:
            next_pil_image = Image.open(next_frame_img_path).convert("RGB")
        except FileNotFoundError:
            raise IndexError(f"Could not read frame {actual_idx} from {next_frame_img_path}.")
        next_image = self.transform(next_pil_image)

        # Get the action that led to the `next_frame`
        action_row = self.metadata.iloc[actual_idx]
        action_data = json.loads(str(action_row['action']))
        action_int = int(action_data[0] if isinstance(action_data, list) else action_data)
        curr_action = torch.zeros(self.num_actions)
        curr_action[action_int] = 1.0   

        return {
            "frame_history": history_tensor,
            "action": curr_action,
            "next_frame": next_image
        }

def train():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser(description="GameNGen Finetuning")
    parser.add_argument("--metadata_input", type=str, required=True, help="Path to the metadata CSV file")
    parser.add_argument("--frames_input", type=str, required=True, help="Path to the frames directory")
    parser.add_argument("--experiment_name", type=str, default="GameNGen Finetuning", help="Name of the MLflow experiment.")
    args = parser.parse_args()

    # --- MLflow Integration ---
    # Check for Azure ML environment.
    # The v1 SDK may set AZUREML_MLFLOW_URI, while v2 sets MLFLOW_TRACKING_URI.
    is_azureml_env = "AZUREML_MLFLOW_URI" in os.environ or \
                     ("MLFLOW_TRACKING_URI" in os.environ and "azureml" in os.environ["MLFLOW_TRACKING_URI"])

    if is_azureml_env:
        # In Azure ML, MLflow is configured automatically by environment variables.
        # We don't need to set the tracking URI or experiment name.
        logging.info("✅ MLflow using Azure ML environment configuration.")
    else:
        # For local runs, explicitly set up a local tracking URI and experiment.
        # This will save runs to a local 'mlruns' directory.
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(args.experiment_name)
        logging.info(f"⚠️  Using local MLflow tracking (./mlruns) for experiment '{args.experiment_name}'.")
    
    # --- Setup ---
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1
    )
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    # Define file paths using the config
    metadata_path = args.metadata_input
    frames_dir = args.frames_input
    
    engine = GameNGen(model_config.model_id, model_config.num_timesteps, history_len=model_config.history_len)

    # --- Memory Saving Optimizations ---
    engine.unet.enable_gradient_checkpointing()
    # try:
    #     engine.unet.enable_xformers_memory_efficient_attention()
    #     logging.info("xformers memory-efficient attention enabled.")
    # except ImportError:
    #     logging.warning("xformers is not installed. For better memory efficiency, run: pip install xformers")

    dataset = NextFrameDataset(model_config.num_actions, metadata_path, frames_dir, model_config.image_size, history_len=model_config.history_len, subset_percentage=train_config.subset_percentage)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0
    )  

    cross_attention_dim = engine.unet.config.cross_attention_dim 
    action_encoder = ActionEncoder(model_config.num_actions, cross_attention_dim)
    

    if model_config.use_lora:
        engine.unet.requires_grad_(False)
        lora_config = LoraConfig(
            r=train_config.lora_rank,
            lora_alpha=train_config.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="lora_only",
        )
        engine.unet.add_adapter(lora_config)
        lora_layers = filter(lambda p: p.requires_grad, engine.unet.parameters())
        params_to_train = list(lora_layers) + list(action_encoder.parameters()) 
    else:
        params_to_train = list(engine.unet.parameters()) + list(action_encoder.parameters()) 
    
    optim = torch.optim.AdamW(params=params_to_train, lr=train_config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optim, num_warmup_steps=500, num_training_steps=len(dataloader) * train_config.num_epochs
    )
    engine, action_encoder, optim, dataloader, lr_scheduler = accelerator.prepare(
        engine, action_encoder, optim, dataloader, lr_scheduler
    )

    mlflow.autolog(log_models=False)
    
    # --- Add an output directory for checkpoints ---
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Starting training loop...")

    mlflow.log_params({
            "learning_rate": train_config.learning_rate,
            "batch_size": train_config.batch_size,
            "num_epochs": train_config.num_epochs,
            "use_lora": model_config.use_lora,
            "lora_rank": train_config.lora_rank if model_config.use_lora else None,
            "subset_percentage": train_config.subset_percentage
        })

    global_step = 0
    for epoch in range(train_config.num_epochs):
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for batch in dataloader:
            optim.zero_grad()
            next_frames, actions, frame_history = batch["next_frame"], batch["action"], batch["frame_history"]

            # Encode into latent space
            with torch.no_grad():
                vae = accelerator.unwrap_model(engine).vae
                latent_dist = vae.encode(next_frames).latent_dist
                clean_latents = latent_dist.sample() * vae.config.scaling_factor

                # Encode history frames
                bs, hist_len, C, H, W = frame_history.shape
                frame_history = frame_history.view(bs * hist_len, C, H, W)
                history_latents = vae.encode(frame_history).latent_dist.sample()
                _, latent_C, latent_H, latent_W = history_latents.shape
                history_latents = history_latents.reshape(bs, hist_len * latent_C, latent_H, latent_W)

            # Add noise to history latents to prevent drift (noise augmentation)
            noise_level = 0.1 # Start with a small, fixed amount of noise
            history_noise = torch.randn_like(history_latents) * noise_level
            corrupted_history_latents = history_latents + history_noise

            # Conditioning is now only the action
            action_conditioning = action_encoder(actions)
            conditioning_batch = action_conditioning.unsqueeze(1)

            # create random noise
            noise = torch.randn_like(clean_latents)

            # pick random timestep. High timstep means more noise
            timesteps = torch.randint(0, engine.scheduler.config.num_train_timesteps, (clean_latents.shape[0], ), device=clean_latents.device).long() 

            noisy_latents = engine.scheduler.add_noise(clean_latents, noise, timesteps)

            # Concatenate history latents with noisy latents
            model_input = torch.cat([noisy_latents, corrupted_history_latents], dim=1)

            with accelerator.accumulate(engine):
                noise_pred = engine(model_input, timesteps, conditioning_batch)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(engine.unet.parameters(), 1.0)
                optim.step()
                lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            
            # Log metrics to MLflow
            if global_step % 10 == 0:  # Log every 10 steps to avoid too much overhead
                mlflow.log_metric("loss", logs["loss"], step=global_step)
                mlflow.log_metric("learning_rate", logs["lr"], step=global_step)

            progress_bar.set_postfix(**logs)
            global_step += 1
            
        progress_bar.close()

        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch} complete. Saving checkpoint...")
            
            # Define a unique directory for this epoch's checkpoint
            checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
            
            # Use accelerator.save_state to save everything
            accelerator.save_state(checkpoint_dir)
            
            logging.info(f"Checkpoint saved to {checkpoint_dir}")

    # Save models at the end of training
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(engine).unet
        unwrapped_action_encoder = accelerator.unwrap_model(action_encoder)
        
        try:
            # Log the action encoder
            mlflow.pytorch.log_model(unwrapped_action_encoder, "action_encoder")
            logging.info("✅ Action encoder logged to MLflow")

            # Log the UNet (or its LoRA weights)
            if model_config.use_lora:
                from peft import get_peft_model_state_dict
                import json
                
                lora_save_path = "unet_lora_weights"
                os.makedirs(lora_save_path, exist_ok=True)
                
                # Save LoRA weights using PEFT method
                lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                torch.save(lora_state_dict, os.path.join(lora_save_path, "pytorch_lora_weights.bin"))
                
                # Save adapter config
                adapter_config = unwrapped_unet.peft_config
                with open(os.path.join(lora_save_path, "adapter_config.json"), "w") as f:
                    json.dump(adapter_config, f, indent=2, default=str)
                
                mlflow.log_artifacts(lora_save_path, artifact_path="unet_lora")
                logging.info("✅ LoRA weights logged to MLflow")
            else:
                mlflow.pytorch.log_model(unwrapped_unet, "unet")
                logging.info("✅ UNet logged to MLflow")

            logging.info(f"✅ Training completed. MLflow Run ID: {mlflow.active_run().info.run_id}")
            
        except Exception as e:
            logging.error(f"❌ Error logging models to MLflow: {e}")
            # Save models locally as fallback
            torch.save(unwrapped_action_encoder.state_dict(), os.path.join(output_dir, "action_encoder.pth"))
            if model_config.use_lora:
                try:
                    from peft import get_peft_model_state_dict
                    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                    torch.save(lora_state_dict, os.path.join(output_dir, "lora_weights.bin"))
                    logging.info("📁 LoRA weights saved locally")
                except Exception as lora_e:
                    logging.error(f"❌ Error saving LoRA weights: {lora_e}")
                    torch.save(unwrapped_unet.state_dict(), os.path.join(output_dir, "unet_full.pth"))
                    logging.info("📁 Full UNet saved locally as fallback")
            else:
                torch.save(unwrapped_unet.state_dict(), os.path.join(output_dir, "unet.pth"))
            logging.info("📁 Models saved locally as fallback")

if __name__ == "__main__":
    train()