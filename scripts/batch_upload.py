import os
from pathlib import Path
from huggingface_hub import HfApi

# --- 1. CONFIGURE YOUR UPLOAD ---
REPO_ID = "RevanthGundala/tiny_engine"
LOCAL_DATA_FOLDER = "gamelogs/frames"
BATCH_SIZE = 5000  # Upload 5,000 files at a time
# --------------------------------

# Make sure you are logged in with a write token
# You can run `huggingface-cli login` in your terminal
api = HfApi()

# Get a list of all files to upload
all_files = [p for p in Path(LOCAL_DATA_FOLDER).rglob("*") if p.is_file()]
print(f"Found {len(all_files)} files to upload.")

# Loop through the files in batches
for i in range(0, len(all_files), BATCH_SIZE):
    batch = all_files[i : i + BATCH_SIZE]
    batch_paths = [str(p) for p in batch]

    # Create a clean list of file patterns for the current batch
    allow_patterns = [os.path.relpath(p, Path(LOCAL_DATA_FOLDER).parent) for p in batch_paths]

    print(f"\nUploading batch {i//BATCH_SIZE + 1}: {len(batch)} files...")

    try:
        # Upload the current batch
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=Path(LOCAL_DATA_FOLDER).parent, # Upload from the parent directory
            allow_patterns=allow_patterns, # Only upload files in the current batch
            commit_message=f"Upload batch {i//BATCH_SIZE + 1}"
        )
        print(f"Batch {i//BATCH_SIZE + 1} uploaded successfully.")
    except Exception as e:
        print(f"Error uploading batch {i//BATCH_SIZE + 1}: {e}")
        break

print("\nAll batches uploaded.")