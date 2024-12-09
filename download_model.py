import argparse
import shutil
from huggingface_hub import snapshot_download
from pathlib import Path

# Fixed repo_id
REPO_ID = "auphong2707/machine-translation-en-vi"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download specific files from a fixed Hugging Face repository and flatten them.")
parser.add_argument("--hf_dir", type=str, required=True, help="The directory on Hugging Face to download (e.g., 'experiment_2_0/best_model/*').")
parser.add_argument("--local_dir", type=str, required=True, help="The local directory where the files will be saved (flattened).")

args = parser.parse_args()

# Temporary directory to store the original structure
temp_dir = "./temp_hf_download"

# Download the specified folder from the Hugging Face repository
snapshot_download(
    repo_id=REPO_ID,
    revision="main",
    local_dir=temp_dir,
    allow_patterns=[args.hf_dir]  # Only download files in the specified Hugging Face directory
)

# Flatten the directory and copy files to the target local_dir
local_dir = Path(args.local_dir)
local_dir.mkdir(parents=True, exist_ok=True)

for file in Path(temp_dir).glob("**/*"):
    if file.is_file():
        shutil.copy(file, local_dir / file.name)

# Cleanup temporary directory
shutil.rmtree(temp_dir)

print(f"Files from {REPO_ID}/{args.hf_dir} have been downloaded and saved to: {local_dir}")
