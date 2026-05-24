import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import requests
import threading

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "correct_text",
            "description": "Correct grammar, punctuation and improve readability of the loaded text",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate the loaded text to another language",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_language": {
                        "type": "string",
                        "description": "The language to translate to, e.g. English, French, Spanish"
                    }
                },
                "required": ["target_language"]
            }
        }
    }
]

ROUTER_PROMPT = """You are a text processing assistant. A text file has been loaded.
Based on the user's request, decide which tool to call:
- correct_text: if the user wants grammar/punctuation fixes
- translate_text: if the user wants the text translated (extract the target language)
- No tool: if the request is about something else entirely

Only call a tool if the request clearly maps to one of the two capabilities."""

CORRECTION_PROMPT = """Fix grammar, punctuation and improve readability of this text.
Rules:
- Do NOT change meaning
- Do NOT invent details  
- Keep the text as close to the original as possible
Output ONLY the corrected text, no preamble or explanation."""

TRANSLATION_PROMPT = """Translate this text to {target_language}.
Rules:
- Do NOT summarize
- Do NOT add content
- Maintain original meaning faithfully
Output ONLY the translated text, no preamble or explanation."""


class TextAgent:
    def __init__(self):
        self.file_path = None
        self.file_content = None

    def load_file(self, path):
        self.file_path = path
        with open(path, "r", encoding="utf-8") as f:
            self.file_content = f.read()

    def route(self, user_request):
        """Call 1: LLM decides which tool to use"""
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:14b",
                "messages": [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": user_request}
                ],
                "tools": TOOLS,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["message"]

    def execute(self, tool_name, tool_args):
        """Call 2: LLM actually executes the chosen operation"""
        if tool_name == "correct_text":
            prompt = f"{CORRECTION_PROMPT}\n\n{self.file_content}"
            suffix = "corrected"
        elif tool_name == "translate_text":
            target = tool_args.get("target_language", "English")
            prompt = f"{TRANSLATION_PROMPT.format(target_language=target)}\n\n{self.file_content}"
            suffix = "translated"
        else:
            return None, None

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:14b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()["message"]["content"]
        return result, suffix

    def process(self, user_request):
        # Step 1: route
        message = self.route(user_request)
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            # LLM didn't call any tool → out of scope
            return None, None, "I can only correct or translate the loaded text."

        # Step 2: execute
        tool_name = tool_calls[0]["function"]["name"]
        tool_args = tool_calls[0]["function"]["arguments"]
        result, suffix = self.execute(tool_name, tool_args)
        return result, suffix, None

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
    append_to_display("System", f"File loaded: {filename}\nTell me what to do with it.")


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
    status_label.config(text="Thinking...")

    def run():
        result, suffix, refusal = agent.process(request)
        root.after(0, lambda: on_result(result, suffix, refusal))

    threading.Thread(target=run, daemon=True).start()


def on_result(result, suffix, refusal):
    status_label.config(text="")
    send_button.config(state=tk.NORMAL)

    if refusal:
        append_to_display("Agent", refusal)
        return

    if not result:
        messagebox.showerror("Error", "Ollama is not responding. Make sure it's running.")
        return

    output_path = agent.save(result, suffix)
    filename = output_path.split("/")[-1]
    append_to_display("Agent", f"✅ Done! File saved as: {filename}")


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
root.title("AI Text Agent — Tool Calling")
root.geometry("800x700")

top_frame = tk.Frame(root)
top_frame.pack(fill=tk.X, padx=10, pady=8)

tk.Button(top_frame, text="📂 Load TXT", command=load_file, width=12).pack(side=tk.LEFT)
file_label = tk.Label(top_frame, text="No file loaded", fg="gray")
file_label.pack(side=tk.LEFT, padx=10)
tk.Button(top_frame, text="Clear", command=clear_all, width=8).pack(side=tk.RIGHT)

output_display = scrolledtext.ScrolledText(
    root, state=tk.DISABLED, wrap=tk.WORD, font=("Arial", 11)
)
output_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

bottom_frame = tk.Frame(root)
bottom_frame.pack(fill=tk.X, padx=10, pady=5)

status_label = tk.Label(bottom_frame, text="", fg="gray")
status_label.pack(side=tk.LEFT)

root.mainloop()