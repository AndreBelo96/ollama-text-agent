import requests
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


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
def generate_images(prompts, output_dir):
    print("Agent 3: Generating images...")

    client = InferenceClient(
        provider="wavespeed",
        api_key=HF_TOKEN,
    )

    for key, prompt in prompts.items():
        print(f"  Generating image: {key}...")
        try:
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev",
            )
            image_path = output_dir / f"{key}.png"
            image.save(image_path)
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

# ── MAIN ──────────────────────────────────────────────────────
def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    campaign = generate_campaign()
    print(f"\nCampaign: {campaign['title']}\n")

    prompts = generate_image_prompts(campaign)
    narrative = generate_narrative(campaign)

    save_campaign_text(narrative, campaign, output_dir)
    save_prompts_md(prompts, campaign, output_dir)

    #generate_images(prompts, output_dir)

    print("\nDone! Check the output/ folder.")

if __name__ == "__main__":
    main()