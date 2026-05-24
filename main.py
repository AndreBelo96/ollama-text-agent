import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import requests
import threading

SYSTEM_PROMPT = """You are a text processing assistant with exactly two capabilities:

1. CORRECTION: Fix grammar, punctuation and improve readability of text.
   - Do NOT change meaning
   - Do NOT invent details
   - Keep the text as close to the original as possible

2. TRANSLATION: Translate a text from the given language to another specified by the user.
   - Do NOT summarize
   - Do NOT add content
   - Maintain original meaning faithfully

Rules:
- If the user requests a correction → output ONLY the corrected text. No preamble, no explanation.
- If the user requests a translation → output ONLY the translated text. No preamble, no explanation.
- If the user requests anything else → respond with exactly: "I can only correct or translate the loaded text."
- Never process text unless explicitly asked.
- The text to process is always marked as [TEXT]."""


class TextAgent:
    def __init__(self):
        self.file_path = None
        self.file_content = None

    def load_file(self, path):
        self.file_path = path
        with open(path, "r", encoding="utf-8") as f:
            self.file_content = f.read()

    def detect_suffix(self, request):
        r = request.lower()
        if any(w in r for w in ["translat", "traduc"]):
            return "translated"
        return "corrected"

    def process(self, user_request):
        message = f"[TEXT]:\n{self.file_content}\n\n[REQUEST]:\n{user_request}"
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "qwen2.5:14b",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": message}
                    ],
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Ollama failed: {e}")
            return None

    def save(self, content, suffix):
        output_path = self.file_path.replace(".txt", f"_{suffix}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def clear(self):
        self.file_path = None
        self.file_content = None


agent = TextAgent()


def append_to_display(sender, message):
    output_display.config(state=tk.NORMAL)
    output_display.insert(tk.END, f"{sender}:\n{message}\n\n{'─'*60}\n\n")
    output_display.config(state=tk.DISABLED)
    output_display.see(tk.END)


def load_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not path:
        return
    agent.load_file(path)
    filename = path.split("/")[-1]
    file_label.config(text=f"📄 {filename}", fg="green")
    send_button.config(state=tk.NORMAL)
    append_to_display("System", f"File loaded: {filename}\nType your request (e.g. 'correct it', 'translate it to English').")


def send_request(event=None):
    request = user_input.get("1.0", tk.END).strip()
    if not request:
        return

    if not agent.file_content:
        messagebox.showwarning("No file", "Please load a .txt file first.")
        return

    user_input.delete("1.0", tk.END)
    append_to_display("You", request)
    send_button.config(state=tk.DISABLED)
    status_label.config(text="Processing...")

    # Detect suffix before threading
    suffix = agent.detect_suffix(request)

    def run():
        result = agent.process(request)
        root.after(0, lambda: on_result(result, suffix))

    threading.Thread(target=run, daemon=True).start()


def on_result(result, suffix):
    status_label.config(text="")
    send_button.config(state=tk.NORMAL)

    if not result:
        messagebox.showerror("Error", "Ollama is not responding. Make sure it's running.")
        return

    if result.strip().startswith("I can only"):
        append_to_display("Agent", result)
        return

    # Salva il file
    output_path = agent.save(result, suffix)

    # Mostra solo conferma in chat, non il testo intero
    filename = output_path.split("/")[-1]
    append_to_display("Agent", f"Done! File saved as: {filename}")


def clear_all():
    agent.clear()
    output_display.config(state=tk.NORMAL)
    output_display.delete("1.0", tk.END)
    output_display.config(state=tk.DISABLED)
    file_label.config(text="No file loaded", fg="gray")
    send_button.config(state=tk.DISABLED)
    status_label.config(text="")


# ── GUI ────────────────────────────────────────────────────────
root = tk.Tk()
root.title("AI Text Agent")
root.geometry("800x700")

# Top bar
top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X, padx=10, pady=8)

tk.Button(top_frame, text="📂 Load TXT", command=load_file, width=12).pack(side=tk.LEFT)
file_label = tk.Label(top_frame, text="No file loaded", fg="gray")
file_label.pack(side=tk.LEFT, padx=10)
tk.Button(top_frame, text="Clear", command=clear_all, width=8).pack(side=tk.RIGHT)

# Output display
output_display = scrolledtext.ScrolledText(
    root, state=tk.DISABLED, wrap=tk.WORD, font=("Arial", 11)
)
output_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Input area
input_frame = tk.Frame(root)
input_frame.pack(fill=tk.X, padx=10, pady=5)

user_input = tk.Text(input_frame, height=3, font=("Arial", 11), wrap=tk.WORD)
user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
user_input.bind("<Return>", lambda e: (send_request(), "break")[1])
user_input.bind("<Shift-Return>", lambda e: None)

send_button = tk.Button(
    input_frame, text="Send", command=send_request,
    width=8, height=3, state=tk.DISABLED
)
send_button.pack(side=tk.RIGHT, padx=(5, 0))

# Bottom bar
bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.X, padx=10, pady=5)

status_label = tk.Label(bottom_frame, text="", fg="gray")
status_label.pack(side=tk.LEFT)

root.mainloop()