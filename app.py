import os
import io
import base64
import json
from PIL import Image
import cv2
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import GameNGen, ActionEncoder
from src.config import ModelConfig, PredictionConfig
from huggingface_hub import hf_hub_download

# --- FastAPI App ---
app = FastAPI()

# --- CORS ---
origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = ModelConfig()
pred_config = PredictionConfig()

# Load models
engine = GameNGen(model_config.model_id, model_config.num_timesteps, history_len=model_config.history_len).to(device)
cross_attention_dim = engine.unet.config.cross_attention_dim
action_encoder = ActionEncoder(model_config.num_actions, cross_attention_dim).to(device)

# --- Model Download ---
def download_model_from_hub(filename, model_repo_id, output_dir, repo_type="model"):
    """Downloads a model file from HF Hub if it doesn't exist locally."""
    local_path = os.path.join(output_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {model_repo_id}...")
        try:
            hf_hub_download(
                repo_id=model_repo_id,
                filename=filename,
                local_dir=output_dir,
                repo_type=repo_type,
                local_dir_use_symlinks=False # Good for Windows compatibility
            )
            print(f"Successfully downloaded {filename}.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            # Use Local file if it exists
            gamelogs_path = os.path.join("gamelogs/frames", os.path.basename(filename))
            if os.path.exists(gamelogs_path):
                print(f"Using local file {gamelogs_path}.")
                return gamelogs_path
            else:
                print(f"Error: {filename} not found in {output_dir}, not in gamelogs, and not in HF Hub.")
                return None
    return local_path

# Create output directory if it doesn't exist
os.makedirs(pred_config.output_dir, exist_ok=True)


# --- Load Weights ---
# NOTE: Update epoch number if you have a different checkpoint
epoch = pred_config.prediction_epoch
model_repo_id = pred_config.model_repo_id
output_dir = pred_config.output_dir

if model_config.use_lora:
    unet_filename = "pytorch_lora_weights.bin"
    unet_path = download_model_from_hub(unet_filename, model_repo_id, output_dir)
    if unet_path and os.path.exists(unet_path):
        print(f"Loading LoRA weights from {unet_path}")
        # Manually load the state dict with torch
        state_dict = torch.load(unet_path, map_location=device)
        # Pass the loaded state_dict directly
        engine.unet.load_attn_procs(state_dict)
        print("LoRA weights loaded successfully.")
    else:
        print(f"Warning: LoRA weights not found or failed to download. Using base UNet.")
else:
    # This shouldn't run if use_lora is True
    unet_filename = "unet.pth"
    unet_path = download_model_from_hub(unet_filename, model_repo_id, output_dir)
    if unet_path and os.path.exists(unet_path):
        print(f"Loading UNet weights from {unet_path}")
        engine.unet.load_state_dict(torch.load(unet_path))
        print("UNet weights loaded successfully.")
    else:
        print(f"Warning: UNet weights not found or failed to download. Using base UNet.")


action_encoder_filename = "action_encoder.pth"
action_encoder_path = download_model_from_hub(action_encoder_filename, model_repo_id, output_dir)
if action_encoder_path and os.path.exists(action_encoder_path):
    print(f"Loading Action Encoder weights from {action_encoder_path}")
    action_encoder.load_state_dict(torch.load(action_encoder_path))
    print("Action Encoder weights loaded successfully.")
else:
    print(f"Warning: Action encoder weights not found or failed to download. Using randomly initialized weights.")


engine.eval()
action_encoder.eval()

# --- Session State & History ---
# Using a simple in-memory dictionary for a single-user demo.
# In a real multi-user app, this would be a more robust session management system.
session_state = {
    "frame_history": None,  # A deque of latent tensors
    "action_history": None, # A deque of action tensors
}


# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize(model_config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def tensor_to_base64(tensor):
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    # Convert to PIL Image
    img = transforms.ToPILImage()(tensor.squeeze(0))
    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- Action Mapping ---
action_map = pred_config.action_map

# --- API Endpoints ---
class PredictRequest(BaseModel):
    action: str

@app.get("/api/start")
def start_game():
    """Initializes a new game session and returns the first frame."""
    global session_state
    try:
        # Download the first frame from the dataset repository
        print("Downloading starting frame...")
        first_frame_path = download_model_from_hub(
            filename="frames/frame_000000008.png", # Assuming this is a valid frame in the repo
            model_repo_id=pred_config.dataset_repo_id,
            output_dir=pred_config.output_dir,
            repo_type="dataset"
        )
        if not first_frame_path:
            raise HTTPException(status_code=500, detail="Could not find starting frame.")

        print(f"Starting frame downloaded to {first_frame_path}")

        pil_image = Image.open(first_frame_path).convert("RGB")

        # Initialize histories
        with torch.no_grad():
            initial_frame_tensor = transform(pil_image).unsqueeze(0).to(device)
            initial_latent = engine.vae.encode(initial_frame_tensor).latent_dist.sample()
        
        session_state["frame_history"] = deque([initial_latent] * model_config.history_len, maxlen=model_config.history_len)
        
        noop_action = torch.tensor(action_map["noop"], dtype=torch.float32, device=device).unsqueeze(0)
        session_state["action_history"] = deque([noop_action] * model_config.history_len, maxlen=model_config.history_len)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        print("Game session started, initial history created.")
        return {"frame": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@torch.inference_mode()
@app.post("/api/predict")
def predict(request: PredictRequest):
    """Predicts the next frame based on the current server-side state and received action."""
    global session_state
    
    # Check if session is initialized
    if session_state["frame_history"] is None:
        raise HTTPException(status_code=400, detail="Game session not started. Please call /api/start first.")

    # Get action tensor from request
    action_list = action_map.get(request.action)
    if action_list is None:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
    action_tensor = torch.tensor(action_list, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Inference ---
    # 1. Get the history of frame latents for channel-wise concatenation
    history_latents = torch.cat(list(session_state["frame_history"]), dim=1)

    # 2. Get conditioning embedding from the action
    action_conditioning = action_encoder(action_tensor)
    conditioning_batch = action_conditioning.unsqueeze(1)

    # 3. DDIM sampling
    # Initialize latents for the new frame we want to predict.
    # This tensor will be iteratively denoised.
    out_channels = 4  # The number of channels in the VAE latent space
    current_latents = torch.randn(
        (1, out_channels, model_config.image_size[0] // 8, model_config.image_size[1] // 8),
        device=device
    )

    for t in engine.scheduler.timesteps:
        # Concatenate the current noisy latents with the frame history to create the model input
        model_input = torch.cat([current_latents, history_latents], dim=1)

        # Predict the noise for the current timestep
        noise_pred = engine(model_input, t, conditioning_batch)

        # Denoise the current_latents based on the noise prediction
        current_latents = engine.scheduler.step(noise_pred, t, current_latents).prev_sample

    # 4. Decode the final denoised latents to an image
    predicted_latent_unscaled = current_latents / engine.vae.config.scaling_factor
    image = engine.vae.decode(predicted_latent_unscaled).sample

    # --- Update State ---
    session_state["frame_history"].append(predicted_latent_unscaled)
    session_state["action_history"].append(action_tensor)

    # --- Response ---
    next_frame_base64 = tensor_to_base64(image.cpu())
    return {"next_frame": next_frame_base64}


if __name__ == "__main__":
    import uvicorn
    # Note: Using a single worker is important for in-memory model setup
    uvicorn.run(app, host="0.0.0.0", port=8000) 