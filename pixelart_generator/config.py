"""
config.py
---------
Costanti e configurazione centralizzata. Nessun import pesante qui:
questo file va importato ovunque senza costi.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_TEMPERATURE = 0.35  # bassa: vogliamo output JSON prevedibile e coerente, non creativo

# --- Stable Diffusion ---
MODEL_PATH = PROJECT_ROOT / "models" / "stable-diffusion-v1-5"
SD_BASE_RESOLUTION = 384  # su CPU ogni pixel in più costa caro in tempo

DEVICE = "cpu"  # forzato: GPU AMD, niente CUDA disponibile su questo setup

# --- IP-Adapter (stile di riferimento) ---
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "models"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
IP_ADAPTER_ENCODER_SUBFOLDER = "models/image_encoder"

STYLE_REFERENCES_DIR = PROJECT_ROOT / "style_references"
STYLE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# --- Fedeltà generazione (nessuno slider manuale: decide la pipeline) ---
# Se l'utente fornisce un'immagine base: la seguiamo da vicino (bassa strength).
BASE_IMAGE_FIDELITY_STRENGTH = 0.35
# Se non c'e' immagine base: generiamo prima un concept fedele al prompt (txt2img
# puro), poi lo ri-processiamo con questa strength per convertirlo in pixel art.
AUTO_PIXELART_CONVERSION_STRENGTH = 0.55

PIXEL_ART_STYLE_SUFFIX = "pixel art, retro game sprite, 16-bit style, limited color palette, sharp pixels"

# --- Pixel processing ---
DEFAULT_PIXEL_GRID = 64
DEFAULT_N_COLORS = 16
DEFAULT_PREVIEW_SIZE = 512