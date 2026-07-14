"""
ollama_agent.py
---------------
STEP 1 della pipeline: trasforma un soggetto breve (es. "un guerriero nano")
in un prompt dettagliato per Stable Diffusion, tramite Ollama.
"""

import json
from dataclasses import dataclass

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE

SYSTEM_INSTRUCTIONS = """Sei un esperto prompt engineer per Stable Diffusion specializzato in pixel art.
Dato un breve soggetto in italiano, genera SOLO un oggetto JSON valido (nessun testo extra, nessun markdown)
con questa struttura esatta:

{
  "positive_prompt": "prompt dettagliato in inglese, MASSIMO 60 parole, include dettagli su posa, colori, illuminazione, inquadratura. Includi sempre uno sfondo semplice/uniforme (es. 'plain solid color background', 'simple flat background'), niente scene affollate.",
  "negative_prompt": "elementi da evitare in inglese, es. blurry, cluttered background, busy background, watermark, text, deformed"
}

Il positive_prompt NON deve menzionare "pixel art" ne' stile retro: descrive semplicemente il soggetto in modo chiaro e dettagliato,
come se fosse per un'illustrazione normale. Lo stile pixel art viene applicato in un secondo momento da un altro passaggio.
Resta entro 60 parole: CLIP tronca tutto oltre i 77 token.
"""


@dataclass
class PromptPair:
    positive: str
    negative: str


def generate_prompt_with_ollama(subject: str) -> PromptPair:
    """Chiede a Ollama di trasformare il soggetto in un prompt dettagliato per SD."""
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_INSTRUCTIONS,
        "prompt": f"Soggetto: {subject}",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    raw_text = data.get("response", "").strip()
    parsed = json.loads(raw_text)

    return PromptPair(
        positive=parsed["positive_prompt"],
        negative=parsed["negative_prompt"],
    )