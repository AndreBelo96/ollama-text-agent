"""
ui.py
-----
Interfaccia Gradio e orchestrazione della pipeline completa.
"""

import random

import gradio as gr

from ollama_agent import generate_prompt_with_ollama
from pixel_processing import pixelate
from sd_pipeline import generate_image_sd, has_style_reference

MAX_SEED = 2**31 - 1


def full_pipeline(
    subject: str,
    base_image,
    num_images: int,
    pixel_grid: int,
    n_colors: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    style_strength: float,
    remove_background: bool,
    progress=gr.Progress(),
):
    if not subject or not subject.strip():
        raise gr.Error("Inserisci un soggetto per generare l'immagine.")

    num_images = int(num_images)

    # Prompt generato UNA sola volta: il batch riusa lo stesso prompt/stile/parametri,
    # variando solo il seed -> immagini comparabili tra loro, non "mele con pere".
    progress(0.0, desc="Generazione prompt con Ollama...")
    try:
        prompts = generate_prompt_with_ollama(subject)
    except Exception as e:
        raise gr.Error(f"Errore nella chiamata a Ollama: {e}")

    # Seed di partenza per il batch: se l'utente ne ha fissato uno, le immagini
    # successive derivano da quello (riproducibile); altrimenti tutte casuali.
    base_seed = int(seed) if seed is not None and int(seed) >= 0 else None

    raw_images, previews, lowres_images = [], [], []

    for i in range(num_images):
        if base_seed is not None:
            image_seed = base_seed + i
        else:
            image_seed = random.randint(0, MAX_SEED)

        progress(
            (i / num_images) * 0.85 + 0.05,
            desc=f"Generazione immagine {i + 1}/{num_images} (seed {image_seed})...",
        )
        try:
            raw_image = generate_image_sd(
                prompts.positive,
                prompts.negative,
                base_image=base_image,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=image_seed,
                style_strength=style_strength,
                progress=None,  # il progresso per-step non ha senso con piu' immagini in coda
            )
        except Exception as e:
            raise gr.Error(f"Errore nella chiamata a Stable Diffusion (immagine {i + 1}): {e}")

        preview, final_lowres = pixelate(
            raw_image, pixel_grid=pixel_grid, n_colors=n_colors, remove_background=remove_background
        )

        raw_images.append(raw_image)
        previews.append(preview)
        lowres_images.append(final_lowres)

    progress(1.0, desc="Completato!")

    info_text = (
        f"**Prompt positivo:** {prompts.positive}\n\n"
        f"**Prompt negativo:** {prompts.negative}\n\n"
        f"**Immagini generate:** {num_images}"
    )

    return raw_images, previews, lowres_images, info_text


def build_ui() -> gr.Blocks:
    style_note = (
        "✅ Stile di riferimento caricato da `style_references/`"
        if has_style_reference()
        else "⚠️ Nessuna immagine trovata in `style_references/`: la generazione userà solo il prompt testuale."
    )

    with gr.Blocks(title="Pixel Art Generator") as demo:
        gr.Markdown("# 🎮 Pixel Art Generator\nInserisci un soggetto, il resto lo fa la pipeline.")
        gr.Markdown(style_note)

        with gr.Row():
            with gr.Column(scale=1):
                subject_input = gr.Textbox(
                    label="Soggetto",
                    placeholder="Es. Un guerriero nano con ascia e armatura blu",
                    lines=2,
                )

                base_image_input = gr.Image(
                    label="Immagine base / target (opzionale)",
                    type="pil",
                    image_mode="RGBA",
                )

                num_images_slider = gr.Slider(
                    1, 6, value=1, step=1,
                    label="Numero di immagini da generare (stesso prompt/stile, seed diversi)",
                )

                with gr.Accordion("Parametri avanzati", open=False):
                    pixel_grid_slider = gr.Slider(16, 128, value=64, step=16, label="Griglia pixel (risoluzione finale)")
                    n_colors_slider = gr.Slider(4, 64, value=16, step=1, label="Numero colori palette")
                    steps_slider = gr.Slider(10, 50, value=20, step=1, label="Step diffusione (SD)")
                    cfg_slider = gr.Slider(1, 15, value=7.5, step=0.5, label="CFG scale (SD)")
                    seed_input = gr.Number(
                        value=-1,
                        label="Seed di partenza (-1 = casuale; con batch, le successive sono seed+1, seed+2, ...)",
                    )
                    style_strength_slider = gr.Slider(
                        0.0, 1.5, value=0.6, step=0.05,
                        label="Intensità stile di riferimento (IP-Adapter)",
                    )
                    remove_bg_checkbox = gr.Checkbox(
                        value=False, label="Rimuovi sfondo nell'immagine finale (richiede 'rembg')"
                    )

                generate_btn = gr.Button("Genera immagine/i", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("Risultato finale (preview)"):
                    preview_output = gr.Gallery(label="Pixel art (anteprima ingrandita)", columns=3)
                with gr.Tab("Export bassa risoluzione"):
                    lowres_output = gr.Gallery(label="Immagini finali reali (bassa risoluzione, per uso in-game)", columns=3)
                with gr.Tab("Immagine grezza SD"):
                    raw_output = gr.Gallery(label="Output grezzo di Stable Diffusion (pre-pixelizzazione)", columns=3)

                prompt_info = gr.Markdown()

        generate_btn.click(
            fn=full_pipeline,
            inputs=[
                subject_input,
                base_image_input,
                num_images_slider,
                pixel_grid_slider,
                n_colors_slider,
                steps_slider,
                cfg_slider,
                seed_input,
                style_strength_slider,
                remove_bg_checkbox,
            ],
            outputs=[raw_output, preview_output, lowres_output, prompt_info],
        )

    return demo