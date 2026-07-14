"""
pixel_processing.py
--------------------
STEP 3 della pipeline: downscale + quantizzazione palette + (opzionale)
rimozione sfondo con canale alpha pulito.
"""

import gradio as gr
from PIL import Image


def pixelate(
    image: Image.Image,
    pixel_grid: int = 64,
    n_colors: int = 16,
    upscale_for_preview: int = 512,
    remove_background: bool = False,
):
    """
    1) (opzionale) rimozione sfondo SUBITO, sull'immagine grezza ad alta risoluzione
       -> i bordi sono molto più puliti che rimuovendo lo sfondo dopo la pixelizzazione
    2) downscale in DUE passaggi:
       a) prima una riduzione "morbida" (LANCZOS) a una risoluzione intermedia
          -> fa la media delle zone, elimina il rumore/bordi sfumati di SD
       b) poi lo scatto netto a nearest-neighbor alla griglia finale bassa
          -> senza questo doppio passaggio, il downscale diretto "pesca" un
          singolo pixel per blocco, che puo' cadere su un bordo sfumato e dare
          un risultato rumoroso/sfocato invece di blocchi di colore netti
    3) quantizzazione colori per ridurre la palette
    4) se c'e' un canale alpha, lo si sogliona (0 o 255): evita pixel semi-trasparenti
       "fantasma" ai bordi, che in pixel art a bassa risoluzione si notano moltissimo
    5) upscale nearest-neighbor per una preview nitida a schermo
    """
    alpha_channel = None

    working = image
    if remove_background:
        try:
            from rembg import remove
        except ImportError:
            raise gr.Error("Per la rimozione sfondo installa 'rembg': pip install rembg")

        rgba = remove(image.convert("RGB"))
        alpha_channel = rgba.split()[3]
        working = rgba.convert("RGB")

    small_rgb = working.resize((pixel_grid * 4, pixel_grid * 4), Image.LANCZOS)
    small_rgb = small_rgb.resize((pixel_grid, pixel_grid), Image.NEAREST)
    quantized_rgb = small_rgb.quantize(
        colors=n_colors, method=Image.MEDIANCUT, dither=Image.Dither.NONE
    ).convert("RGB")

    if alpha_channel is not None:
        small_alpha = alpha_channel.resize((pixel_grid * 4, pixel_grid * 4), Image.LANCZOS)
        small_alpha = small_alpha.resize((pixel_grid, pixel_grid), Image.NEAREST)
        # binarizza l'alpha: niente semi-trasparenze ai bordi, o a bassa risoluzione
        # si vede uno "sporco" evidente attorno al soggetto
        small_alpha = small_alpha.point(lambda a: 255 if a > 128 else 0)
        final = quantized_rgb.convert("RGBA")
        final.putalpha(small_alpha)
    else:
        final = quantized_rgb

    preview = final.resize((upscale_for_preview, upscale_for_preview), Image.NEAREST)
    return preview, final  # preview grande per UI, final a bassa risoluzione per export