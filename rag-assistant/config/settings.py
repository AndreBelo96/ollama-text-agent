from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_PATH = BASE_DIR / "data" / "documents"
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

OLLAMA_MODEL = "qwen2.5:14b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 5

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

OLLAMA_URL = "http://localhost:11434/api/chat"