import subprocess
import sys

def install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

print("Setting up AI Agent Toolkit...\n")

print("Installing text-agent dependencies...")
install(["requests", "python-dotenv"])

print("Installing dnd-generator dependencies...")
install([
    "diffusers", "transformers", "accelerate",
    "pillow", "huggingface_hub"
])

print("Installing rag-assistant dependencies...")
install([
    "llama-index",
    "chromadb",
    "llama-index-vector-stores-chroma",
    "llama-index-llms-ollama",
    "llama-index-embeddings-ollama"
])

print("\nAll dependencies installed!")
print("Next steps:")
print("  - Run dnd-generator/setup.py to download the image model")
print("  - Make sure Ollama is running with: ollama pull qwen2.5:14b")
print("  - Make sure Ollama has embeddings: ollama pull nomic-embed-text")