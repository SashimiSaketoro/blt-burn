import argparse
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

def upload_model(file_path, repo_id, token=None):
    """
    Uploads the model file to the specified Hugging Face repository.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    print(f"Preparing to upload {file_path.name} to {repo_id}...")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
        print(f"Repository {repo_id} ready.")
    except Exception as e:
        print(f"Error creating/accessing repository: {e}")
        sys.exit(1)

    # Upload file
    try:
        print("Uploading... this may take a while for large files.")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"✅ Successfully uploaded {file_path.name} to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload BLT model to Hugging Face Hub")
    parser.add_argument("--file", type=str, default="blt_entropy_model.mpk", help="Path to the model file")
    parser.add_argument("--repo", type=str, default="SashimiSaketoro/entropy_burn", help="Target Repository ID (default: SashimiSaketoro/entropy_burn)")
    parser.add_argument("--token", type=str, help="Hugging Face API Token (optional if logged in via CLI)")
    
    args = parser.parse_args()
    
    upload_model(args.file, args.repo, args.token)
