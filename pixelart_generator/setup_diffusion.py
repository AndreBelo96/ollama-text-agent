import subprocess
import sys
from pathlib import Path


def install_dependencies():
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "requests", "python-dotenv", "diffusers",
        "transformers", "accelerate", "pillow",
        "huggingface_hub<1.0",  # pin: transformers non e' ancora compatibile con la major 1.x
        "torch", "safetensors",
        "gradio",
        "rembg",  # rimozione sfondo per l'immagine finale
    ])
    print("Dependencies installed!")


def download_model():
    # Import fatto qui, dopo l'installazione, cosi' il primo avvio funziona
    # anche se torch/diffusers non erano ancora presenti.
    import torch
    from diffusers import StableDiffusionPipeline

    model_path = Path(__file__).parent / "models" / "stable-diffusion-v1-5"

    if model_path.exists():
        print("Model already downloaded, skipping.")
        return

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32

    print(f"Downloading Stable Diffusion v1.5 (~4GB) - dtype={dtype}, cuda={has_cuda}...")

    # NB: il repo originale "runwayml/stable-diffusion-v1-5" e' stato rimosso
    # da HuggingFace (agosto 2024). Si usa il mirror ufficiale mantenuto da HF:
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.save_pretrained(str(model_path))
    print(f"Model saved in {model_path}")


if __name__ == "__main__":
    install_dependencies()
    download_model()
    print("\nSetup complete! Run main.py to start.")