"""
main.py
-------
Entry point. Esegui: python main.py
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # silenzia warning cosmetico su Windows

from ui import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
