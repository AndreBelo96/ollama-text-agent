import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import requests
import threading

SYSTEM_PROMPT = """You are a helpful text assistant with three capabilities:
1. Correct Italian text: fix grammar, punctuation, improve readability. Do NOT change meaning or invent details.
2. Translate Italian text to English: faithfully, without summarizing or adding content.
3. Chat normally about anything else.

Always respond in the same language the user writes in."""


class ChatAgent:
    def __init__(self):
        self.history = []
        self.file_path = None
        self.last_response = None

    def load_file(self, path):
        self.file_path = path
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})

        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "mistral",
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()["message"]["content"]
            self.history.append({"role": "assistant", "content": result})
            self.last_response = result
            return result

        except Exception as e:
            print(f"Ollama failed: {e}")
            return None

    def save_response(self, suffix):
        if not self.file_path or not self.last_response:
            return None
        output_path = self.file_path.replace(".txt", f"_{suffix}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.last_response)
        return output_path

    def clear(self):
        self.history = []
        self.last_response = None
        self.file_path = None


agent = ChatAgent()


def append_message(sender, message):
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, f"{sender}: {message}\n\n")
    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)


def load_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not path:
        return

    content = agent.load_file(path)
    filename = path.split("/")[-1]
    file_label.config(text=f"📄 {filename}")

    # Inject file content into conversation history
    file_message = f"[File loaded: {filename}]\n\n{content}"
    agent.history.append({"role": "user", "content": file_message})
    append_message("You", f"[Loaded file: {filename}]")

    # Agent acknowledges
    send_to_agent(None, prefill=file_message, show_user=False)


def send_to_agent(event=None, prefill=None, show_user=True):
    message = prefill or user_input.get("1.0", tk.END).strip()
    if not message:
        return

    if show_user:
        append_message("You", message)
        user_input.delete("1.0", tk.END)
        # Add to history only if not already added (prefill case handles it)
        agent.history.append({"role": "user", "content": message})

    send_button.config(state=tk.DISABLED)
    status_label.config(text="Thinking...")

    def run():
        # Call Ollama directly (history already updated)
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "mistral",
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + agent.history,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()["message"]["content"]
            agent.history.append({"role": "assistant", "content": result})
            agent.last_response = result
            root.after(0, lambda: on_response(result))
        except Exception as e:
            root.after(0, lambda: on_error(str(e)))

    threading.Thread(target=run, daemon=True).start()


def on_response(result):
    append_message("Agent", result)
    status_label.config(text="")
    send_button.config(state=tk.NORMAL)
    save_button.config(state=tk.NORMAL)


def on_error(error):
    append_message("Agent", f"Error: {error}")
    status_label.config(text="")
    send_button.config(state=tk.NORMAL)


def save_response():
    if not agent.file_path:
        messagebox.showwarning("Warning", "No file loaded — nothing to save.")
        return

    # Ask user what suffix to use
    suffix = tk.simpledialog.askstring(
        "Save as",
        "File suffix (e.g. corrected, translated):",
        initialvalue="corrected"
    )
    if not suffix:
        return

    path = agent.save_response(suffix)
    if path:
        messagebox.showinfo("Saved", f"File saved:\n{path}")


def clear_chat():
    agent.clear()
    chat_display.config(state=tk.NORMAL)
    chat_display.delete("1.0", tk.END)
    chat_display.config(state=tk.DISABLED)
    file_label.config(text="No file loaded")
    save_button.config(state=tk.DISABLED)
    status_label.config(text="")


# ── GUI ──────────────────────────────────────────────────
root = tk.Tk()
root.title("AI Text Agent")
root.geometry("700x600")

from tkinter import simpledialog  # needed for save dialog

# Top bar
top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X, padx=10, pady=8)

tk.Button(top_frame, text="Load TXT", command=load_file, width=12).pack(side=tk.LEFT)
file_label = tk.Label(top_frame, text="No file loaded", fg="gray")
file_label.pack(side=tk.LEFT, padx=10)
tk.Button(top_frame, text="Clear", command=clear_chat, width=8).pack(side=tk.RIGHT)

# Chat display
chat_display = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, font=("Arial", 11))
chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# Input area
input_frame = tk.Frame(root)
input_frame.pack(fill=tk.X, padx=10, pady=5)

user_input = tk.Text(input_frame, height=3, font=("Arial", 11), wrap=tk.WORD)
user_input.pack(fill=tk.X, side=tk.LEFT, expand=True)
user_input.bind("<Return>", lambda e: (send_to_agent(), "break")[1])
user_input.bind("<Shift-Return>", lambda e: None)  # Shift+Enter = newline

send_button = tk.Button(input_frame, text="Send", command=send_to_agent, width=8, height=3)
send_button.pack(side=tk.RIGHT, padx=(5, 0))

# Bottom bar
bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.X, padx=10, pady=5)

status_label = tk.Label(bottom_frame, text="", fg="gray")
status_label.pack(side=tk.LEFT)

save_button = tk.Button(bottom_frame, text="💾 Save response", command=save_response, state=tk.DISABLED)
save_button.pack(side=tk.RIGHT)

root.mainloop()