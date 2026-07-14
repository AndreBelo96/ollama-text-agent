"""
utils.py
--------
Helper condivisi tra i vari moduli.
"""

from PIL import Image


def flatten_rgba(image: Image.Image, background=(127, 127, 127)) -> Image.Image:
    """SD non capisce il canale alpha: compone l'immagine su uno sfondo neutro."""
    if image is None:
        return None
    if image.mode != "RGBA":
        return image.convert("RGB")
    bg = Image.new("RGB", image.size, background)
    bg.paste(image, mask=image.split()[3])
    return bg


def fit_to_square(image: Image.Image, size: int, background=(127, 127, 127)) -> Image.Image:
    """
    Ridimensiona mantenendo le proporzioni originali e aggiunge bordi (letterbox)
    per ottenere un quadrato size x size, INVECE di stirare/schiacciare l'immagine
    con un resize diretto (che deforma qualsiasi cosa non sia già quadrata).
    """
    if image is None:
        return None

    has_alpha = image.mode == "RGBA"
    ratio = min(size / image.width, size / image.height)
    new_w = max(1, round(image.width * ratio))
    new_h = max(1, round(image.height * ratio))
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    if has_alpha:
        canvas = Image.new("RGBA", (size, size), background + (0,))  # bordi trasparenti
        canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2), mask=resized.split()[3])
    else:
        canvas = Image.new("RGB", (size, size), background)
        canvas.paste(resized.convert("RGB"), ((size - new_w) // 2, (size - new_h) // 2))

    return canvas