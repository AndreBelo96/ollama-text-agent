"""
sd_pipeline.py
--------------
STEP 2 della pipeline: caricamento dei modelli (una sola volta) e generazione
immagine, con supporto a:
  - img2img (se viene fornita un'immagine base/target)
  - IP-Adapter (stile mediato dalle immagini in style_references/), caricato
    SOLO se ci sono davvero immagini nella cartella (altrimenti si risparmia
    tempo/spazio, e si evita di dover gestire l'IP-Adapter "vuoto")
"""

import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from config import (
    AUTO_PIXELART_CONVERSION_STRENGTH,
    BASE_IMAGE_FIDELITY_STRENGTH,
    DEVICE,
    IP_ADAPTER_ENCODER_SUBFOLDER,
    IP_ADAPTER_REPO,
    IP_ADAPTER_SUBFOLDER,
    IP_ADAPTER_WEIGHT,
    MODEL_PATH,
    PIXEL_ART_STYLE_SUFFIX,
    SD_BASE_RESOLUTION,
    STYLE_IMAGE_EXTENSIONS,
    STYLE_REFERENCES_DIR,
)
from utils import fit_to_square, flatten_rgba

DTYPE = torch.float32  # fp16 non è supportato/efficiente su CPU


def _find_style_image_paths():
    if not STYLE_REFERENCES_DIR.exists():
        return []
    return sorted(
        p for p in STYLE_REFERENCES_DIR.iterdir() if p.suffix.lower() in STYLE_IMAGE_EXTENSIONS
    )


_STYLE_IMAGE_PATHS = _find_style_image_paths()
_HAS_STYLE = len(_STYLE_IMAGE_PATHS) > 0


def _load_models(load_ip_adapter: bool):
    image_encoder = None
    if load_ip_adapter:
        print("Caricamento image encoder per IP-Adapter...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_REPO, subfolder=IP_ADAPTER_ENCODER_SUBFOLDER, torch_dtype=DTYPE
        )

    print(f"Caricamento modello Stable Diffusion da {MODEL_PATH} su device={DEVICE}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=DTYPE,
        image_encoder=image_encoder,
        safety_checker=None,  # altrimenti i falsi positivi vengono sostituiti da immagini nere
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)

    if load_ip_adapter:
        print("Caricamento pesi IP-Adapter...")
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder=IP_ADAPTER_SUBFOLDER, weight_name=IP_ADAPTER_WEIGHT)
    # NB: enable_attention_slicing() e' incompatibile con l'IP-Adapter caricato
    # (bug noto di diffusers, issue #7263) quindi non lo attiviamo, ne' qui ne' altrove.

    # Pipeline img2img che RIUSA gli stessi componenti (stesso UNet, stessa VAE, ecc.)
    img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)

    print("Modello caricato.")
    return pipe, img2img_pipe


_pipe, _img2img_pipe = _load_models(load_ip_adapter=_HAS_STYLE)


# ----------------------------------------------------------------------------
# Stile: embedding mediati da tutte le immagini in style_references/
# ----------------------------------------------------------------------------

def _load_averaged_style_embeds():
    """
    Calcola gli embedding IP-Adapter per ogni immagine in style_references/ e li
    media insieme, cosi' il 'tuo stile' e' rappresentato da piu' esempi invece
    che da una singola immagine instabile. Calcolato una sola volta all'avvio.
    Ritorna None se non ci sono immagini (in quel caso l'IP-Adapter non e'
    nemmeno stato caricato sul modello).
    """
    if not _HAS_STYLE:
        return None

    print(f"Calcolo embedding di stile da {len(_STYLE_IMAGE_PATHS)} immagini in {STYLE_REFERENCES_DIR}...")
    per_image_embeds = []
    for path in _STYLE_IMAGE_PATHS:
        img = flatten_rgba(Image.open(path)).resize((224, 224))
        # Ritorna una lista con un tensore per ogni IP-Adapter caricato (qui ne carichiamo uno solo).
        embeds = _pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=img,
            ip_adapter_image_embeds=None,
            device=DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        per_image_embeds.append(embeds[0])

    stacked = torch.stack(per_image_embeds, dim=0)
    averaged = stacked.mean(dim=0)
    print("Embedding di stile pronti.")
    return [averaged]  # stessa forma attesa da ip_adapter_image_embeds


_STYLE_EMBEDS = _load_averaged_style_embeds()


def has_style_reference() -> bool:
    return _HAS_STYLE


# ----------------------------------------------------------------------------
# Generazione immagine
# ----------------------------------------------------------------------------

def generate_image_sd(
    positive_prompt: str,
    negative_prompt: str,
    base_image: Image.Image = None,
    steps: int = 25,
    cfg_scale: float = 7.0,
    seed: int = -1,
    style_strength: float = 0.6,
    progress: gr.Progress = None,
) -> Image.Image:
    """
    Due percorsi, decisi automaticamente (nessuno slider manuale di 'fedeltà'):

      - C'e' un'immagine base -> la seguiamo da vicino (img2img, strength bassa
        fissa). Il prompt/stile hanno un ruolo di rifinitura, non di comando.

      - Nessuna immagine base -> 100% guidato dal prompt:
          Stage 1: txt2img puro con il prompt cosi' com'e', stile a peso ZERO
                    (se l'IP-Adapter e' caricato va comunque "alimentato" con
                    degli embedding, ma con scale=0 non ha alcuna influenza)
                    -> un concept fedele e pulito.
          Stage 2: img2img sul concept, aggiungendo i termini di stile pixel art
                    e l'IP-Adapter al peso scelto -> lo converte in pixel art
                    mantenendo il contenuto del concept.
    """
    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    total_steps = int(steps)

    def _make_callback(label):
        def _on_step_end(pipe, step_index, timestep, callback_kwargs):
            if progress is not None:
                progress((step_index + 1) / total_steps, desc=f"{label} (step {step_index + 1}/{total_steps})")
            return callback_kwargs
        return _on_step_end

    # Una volta che l'IP-Adapter e' caricato sul modello, l'UNet SI ASPETTA
    # sempre un embedding di stile: non si puo' passare None. Per "disattivarlo"
    # si passano comunque gli embedding ma con scale a 0 (nessuna influenza).
    ip_kwargs = {}
    if _HAS_STYLE:
        ip_kwargs["ip_adapter_image_embeds"] = _STYLE_EMBEDS

    if base_image is not None:
        # Percorso 1: c'e' un'immagine -> la seguiamo da vicino.
        if _HAS_STYLE:
            _img2img_pipe.set_ip_adapter_scale(style_strength)

        init_image = flatten_rgba(fit_to_square(base_image, SD_BASE_RESOLUTION))
        result = _img2img_pipe(
            prompt=f"{positive_prompt}, {PIXEL_ART_STYLE_SUFFIX}",
            negative_prompt=negative_prompt,
            image=init_image,
            strength=BASE_IMAGE_FIDELITY_STRENGTH,
            num_inference_steps=total_steps,
            guidance_scale=float(cfg_scale),
            generator=generator,
            callback_on_step_end=_make_callback("Adattamento all'immagine base"),
            **ip_kwargs,
        )
        return result.images[0].convert("RGB")

    # Percorso 2: nessuna immagine -> Stage 1 (concept fedele al prompt, stile a zero).
    if _HAS_STYLE:
        _pipe.set_ip_adapter_scale(0.0)

    if progress is not None:
        progress(0.0, desc="Stage 1/2: generazione concept dal prompt...")
    concept_result = _pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=SD_BASE_RESOLUTION,
        height=SD_BASE_RESOLUTION,
        num_inference_steps=total_steps,
        guidance_scale=float(cfg_scale),
        generator=generator,
        callback_on_step_end=_make_callback("Stage 1/2: concept"),
        **ip_kwargs,
    )
    concept_image = concept_result.images[0].convert("RGB")

    # Stage 2: conversione in pixel art, stile applicato qui al peso scelto.
    if _HAS_STYLE:
        _img2img_pipe.set_ip_adapter_scale(style_strength)

    final_result = _img2img_pipe(
        prompt=f"{positive_prompt}, {PIXEL_ART_STYLE_SUFFIX}",
        negative_prompt=negative_prompt,
        image=concept_image,
        strength=AUTO_PIXELART_CONVERSION_STRENGTH,
        num_inference_steps=total_steps,
        guidance_scale=float(cfg_scale),
        generator=generator,
        callback_on_step_end=_make_callback("Stage 2/2: conversione pixel art"),
        **ip_kwargs,
    )
    return final_result.images[0].convert("RGB")