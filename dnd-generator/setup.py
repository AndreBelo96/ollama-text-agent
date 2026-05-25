from diffusers import StableDiffusionPipeline
from pathlib import Path
import subprocess
import sys

def install_dependencies():
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "requests", "python-dotenv", "diffusers",
        "transformers", "accelerate", "pillow",
        "huggingface_hub"
    ])
    print("Dependencies installed!")

def download_model():
    model_path = Path(__file__).parent / "models" / "stable-diffusion-v1-5"

    if model_path.exists():
        print("Model already downloaded, skipping.")
        return

    print("Downloading Stable Diffusion v1.5 (~4GB)...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.save_pretrained(str(model_path))
    print(f"Model saved in {model_path}")

if __name__ == "__main__":
    install_dependencies()
    download_model()
    print("\nSetup complete! Run main.py to start.")