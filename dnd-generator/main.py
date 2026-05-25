import tkinter as tk
from tkinter import scrolledtext
import requests
import os
import json
import threading
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image, ImageTk

pipe = None

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"


# ── AGENT 1: Campaign Generator ─────────────────────────────
def generate_campaign():
    print("Agent 1: Generating campaign...")

    prompt = """Generate a D&D campaign. Respond ONLY with a valid JSON object, no markdown, no backticks, no explanation.

    The JSON must have exactly this structure:
    {
      "title": "campaign title",
      "setting": "world description in 2-3 sentences",
      "villain": {
        "name": "villain name",
        "description": "villain description in 1-2 sentences",
        "visual": "visual description for image generation: appearance, clothing, expression"
      },
      "scenes": [
        {
          "name": "scene name",
          "description": "scene description in 2-3 sentences",
          "visual": "visual description for image generation: location, atmosphere, lighting, key elements"
        }
      ],
      "npcs": [
        {
          "name": "npc name",
          "role": "their role",
          "visual": "visual description for image generation: appearance, clothing, personality shown through looks"
        }
      ]
    }
    
    Generate 3 scenes and 2 NPCs. Be creative and detailed."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]

    # Clean up in case model adds backticks anyway
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)

# ── AGENT 2: Prompt Engineer ─────────────────────────────────
def generate_image_prompts(campaign):
    print("Agent 2: Engineering image prompts...")

    subjects = []

    subjects.append({
        "key": "villain",
        "name": campaign["villain"]["name"],
        "visual": campaign["villain"]["visual"]
    })

    for i, scene in enumerate(campaign["scenes"]):
        subjects.append({
            "key": f"scene_{i+1}",
            "name": scene["name"],
            "visual": scene["visual"]
        })

    for i, npc in enumerate(campaign["npcs"]):
        subjects.append({
            "key": f"npc_{i+1}",
            "name": npc["name"],
            "visual": npc["visual"]
        })

    prompt = f"""You are an expert at writing Stable Diffusion image generation prompts for fantasy D&D art.

    For each subject below, write an optimized image generation prompt.
    Rules:
    - Style prefix: always start with "fantasy art, detailed, epic, high quality, "
    - Be specific about colors, lighting, composition
    - Add relevant artistic keywords: "dramatic lighting", "intricate details", "concept art", etc.
    - Keep each prompt under 100 words
    - Respond ONLY with a valid JSON object, no markdown, no backticks
    
    Subjects:
    {json.dumps(subjects, indent=2)}
    
    Respond with:
    {{
      "villain": "prompt here",
      "scene_1": "prompt here",
      "scene_2": "prompt here",
      "scene_3": "prompt here",
      "npc_1": "prompt here",
      "npc_2": "prompt here"
    }}"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"]

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)

# ── AGENT 3: Image Generator ─────────────────────────────────
def load_pipeline():
    global pipe
    if pipe is None:
        print("Loading image model...")
        model_path = Path(__file__).parent / "models" / "stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(str(model_path))
        pipe = pipe.to("cpu")
    return pipe

def generate_images(prompts, output_dir, on_image_saved):
    print("Agent 3: Generating images...")
    pipeline = load_pipeline()

    for key, prompt in prompts.items():
        print(f"  Generating image: {key}...")
        try:
            image = pipeline(prompt).images[0]
            image_path = output_dir / f"{key}.png"
            image.save(image_path)
            on_image_saved(key, image_path)
            print(f"  Saved: {image_path}")
        except Exception as e:
            print(f"  Failed {key}: {e}")

# ── AGENT 4: Narrator ────────────────────────────────────────
def generate_narrative(campaign):
    print("Agent 4: Writing campaign narrative...")

    prompt = f"""You are a master Dungeon Master and storyteller. 
    Based on this D&D campaign data, write a rich, immersive narrative document that a DM can read and use at the table.
    
    Campaign data:
    {json.dumps(campaign, indent=2)}
    
    Write the narrative with this structure:
    
    1. INTRODUCTION — A dramatic opening paragraph that sets the tone and world, written as if narrating to players.
    
    2. THE THREAT — A full description of the villain, their backstory, motivations, and what they are planning. Make it dramatic and detailed.
    
    3. THE ADVENTURE — For each scene, write:
       - A narrative description the DM reads aloud to players
       - What the players can discover or do here
       - Atmosphere, sounds, smells, tension
    
    4. THE CAST — For each NPC, write:
       - Their personality and mannerisms
       - How they speak and behave with players
       - What they know and what they hide
       - How they can help or hinder the party
    
    Rules:
    - Write in second person for player-facing text ("You enter...", "You see...")
    - Write in third person for DM notes
    - Be vivid, atmospheric, and detailed
    - Minimum 600 words
    - Do NOT use JSON, just flowing narrative text"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

# ── SAVE CAMPAIGN TEXT ────────────────────────────────────────
def save_campaign_text(narrative, campaign, output_dir):
    lines = []

    lines.append(f"{'='*60}")
    lines.append(f"  {campaign['title'].upper()}")
    lines.append(f"{'='*60}\n")
    lines.append(narrative)
    lines.append(f"\n{'='*60}")

    text_path = output_dir / "campaign.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Campaign text saved: {text_path}")

def save_prompts_md(prompts, campaign, output_dir):
    """File tecnico per Agent 2 — prompt SD strutturati"""
    lines = []

    lines.append(f"# Image Prompts — {campaign['title']}")
    lines.append(f"\nGenerated by Agent 2 — Prompt Engineer\n")
    lines.append(f"{'='*60}\n")

    for key, prompt in prompts.items():
        lines.append(f"## {key}")
        lines.append(f"```")
        lines.append(prompt)
        lines.append(f"```\n")

    md_path = output_dir / "prompts.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Prompts saved: {md_path}")

def save_files(narrative, prompts, campaign, output_dir):
    # campaign.txt
    lines = [
        f"{'='*60}",
        f"  {campaign['title'].upper()}",
        f"{'='*60}\n",
        narrative,
        f"\n{'='*60}"
    ]
    (output_dir / "campaign.txt").write_text("\n".join(lines), encoding="utf-8")

    # prompts.md
    md = [f"# Image Prompts — {campaign['title']}\n"]
    for key, prompt in prompts.items():
        md.append(f"## {key}\n```\n{prompt}\n```\n")
    (output_dir / "prompts.md").write_text("\n".join(md), encoding="utf-8")

# ── GUI ───────────────────────────────────────────────────────

class DnDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DnD Campaign Generator")
        self.root.geometry("900x750")
        self.image_refs = []
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.build_ui()

    def build_ui(self):
        # Header
        tk.Label(self.root, text="⚔️  DnD Campaign Generator", font=("Arial", 20, "bold")).pack(pady=15)

        # Generate button
        self.gen_button = tk.Button(
            self.root, text="Generate Campaign",
            command=self.start_generation,
            font=("Arial", 13), width=20, height=2,
            bg="#4a4a8a", fg="white"
        )
        self.gen_button.pack(pady=5)

        # Agent status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=10)

        self.agent_labels = {}
        agents = [
            ("agent1", "Agent 1 — Campaign Generator"),
            ("agent2", "Agent 2 — Prompt Engineer"),
            ("agent3", "Agent 3 — Image Generator"),
            ("agent4", "Agent 4 — Narrator"),
        ]
        for key, text in agents:
            row = tk.Frame(status_frame)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=text, font=("Arial", 11), width=30, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="⏸ Waiting", font=("Arial", 11), fg="gray", width=20, anchor="w")
            lbl.pack(side=tk.LEFT)
            self.agent_labels[key] = lbl

        # Narrative text
        tk.Label(self.root, text="Campaign Narrative", font=("Arial", 12, "bold")).pack(pady=(10, 2))
        self.narrative_box = scrolledtext.ScrolledText(
            self.root, height=12, wrap=tk.WORD,
            font=("Arial", 10), state=tk.DISABLED
        )
        self.narrative_box.pack(fill=tk.X, padx=15, pady=5)

        # Images frame
        tk.Label(self.root, text="Generated Images", font=("Arial", 12, "bold")).pack(pady=(10, 2))
        self.images_frame = tk.Frame(self.root)
        self.images_frame.pack(pady=5)

    def set_agent_status(self, key, status, color):
        self.root.after(0, lambda: self.agent_labels[key].config(text=status, fg=color))

    def set_narrative(self, text):
        def update():
            self.narrative_box.config(state=tk.NORMAL)
            self.narrative_box.delete("1.0", tk.END)
            self.narrative_box.insert(tk.END, text)
            self.narrative_box.config(state=tk.DISABLED)
        self.root.after(0, update)

    def add_image(self, key, path):
        def update():
            try:
                img = Image.open(path).resize((130, 130))
                photo = ImageTk.PhotoImage(img)
                self.image_refs.append(photo)

                frame = tk.Frame(self.images_frame)
                frame.pack(side=tk.LEFT, padx=5)
                tk.Label(frame, image=photo).pack()
                tk.Label(frame, text=key, font=("Arial", 9)).pack()
            except Exception as e:
                print(f"Could not display image {key}: {e}")
        self.root.after(0, update)

    def start_generation(self):
        self.gen_button.config(state=tk.DISABLED)
        self.image_refs.clear()
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        self.set_narrative("")
        for key in self.agent_labels:
            self.set_agent_status(key, "⏸ Waiting", "gray")

        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        try:
            # Agent 1
            self.set_agent_status("agent1", "⏳ Running...", "orange")
            campaign = generate_campaign()
            self.set_agent_status("agent1", "✅ Done", "green")

            # Agent 2
            self.set_agent_status("agent2", "⏳ Running...", "orange")
            prompts = generate_image_prompts(campaign)
            self.set_agent_status("agent2", "✅ Done", "green")

            # Agent 3 e 4 in parallelo
            self.set_agent_status("agent3", "⏳ Running...", "orange")
            self.set_agent_status("agent4", "⏳ Running...", "orange")

            narrative_result = [None]
            narrative_done = threading.Event()

            def run_agent4():
                narrative_result[0] = generate_narrative(campaign)
                self.set_agent_status("agent4", "✅ Done", "green")
                self.set_narrative(narrative_result[0])
                narrative_done.set()

            def run_agent3():
                generate_images(prompts, self.output_dir, self.add_image)
                self.set_agent_status("agent3", "✅ Done", "green")

            t3 = threading.Thread(target=run_agent3, daemon=True)
            t4 = threading.Thread(target=run_agent4, daemon=True)
            t3.start()
            t4.start()
            t3.join()
            narrative_done.wait()

            # Salva file
            save_files(narrative_result[0], prompts, campaign, self.output_dir)

            self.root.after(0, lambda: self.gen_button.config(state=tk.NORMAL))
            print(f"\nDone! Files saved in {self.output_dir}")

        except Exception as e:
            print(f"Pipeline error: {e}")
            self.root.after(0, lambda: self.gen_button.config(state=tk.NORMAL))


root = tk.Tk()
app = DnDApp(root)
root.mainloop()