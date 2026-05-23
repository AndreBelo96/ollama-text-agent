import tkinter as tk
from tkinter import filedialog, messagebox
import requests

class TextCorrectorAgent:

    def define_prompt(self, text, mode):

        if mode == "Correction":
            return f"""
                Correggi il seguente testo italiano.

                Regole:
                - NON inventare dettagli
                - NON cambiare il significato
                - Mantieni il testo il più vicino possibile all'originale
                - Correggi SOLO errori grammaticali e di punteggiatura
                - correggi grammatica
                - correggi punteggiatura
                - migliora leggibilità

                Testo:
                {text}
                """
        else:
            return f"""
                Traduci il seguente testo dall'italiano all'inglese.

                Regole:
                - NON riassumere
                - NON aggiungere testo
                - Mantieni il significato originale

                Testo:
                {text}
                """

    def manage_agent(self, text, mode):

        prompt = self.define_prompt(text, mode)

        print("Calling local Ollama...")

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False
                }
            )

            response.raise_for_status()
            return response.json()["response"]

        except Exception as ollama_error:
            print("Ollama failed")
            print(ollama_error)
            return None


agent = TextCorrectorAgent()
selected_file = None

def load_file():
    global selected_file

    selected_file = filedialog.askopenfilename(
        filetypes=[("Text files", "*.txt")]
    )

    if selected_file:
        file_label.config(text=f"Selected file:\n{selected_file}")


def process_file():
    global selected_file

    if not selected_file:
        messagebox.showerror("Error", "Please select a .txt file first")
        return

    try:
        # Leggi file
        with open(selected_file, "r", encoding="utf-8") as file:
            original_text = file.read()

        mode = mode_var.get()
        status_label.config(text="Processing...")
        root.update()

        result = agent.manage_agent(original_text, mode)

        if result is None:
            messagebox.showerror("Error", "Ollama is not responding. Make sure it's running.")
            status_label.config(text="")
            return


        suffix = "_corrected" if mode == "Correction" else "_translated"
        output_path = selected_file.replace(".txt", f"{suffix}.txt")

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(result)

        status_label.config(text="Operation completed!")

        messagebox.showinfo("Success", f"File saved:\n{output_path}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI
root = tk.Tk()
root.title("Mini AI Agent")
root.geometry("640x480")

mode_var = tk.StringVar(value="Correction")

title = tk.Label(
    root,
    text="Mini AI Text Modifier",
    font=("Arial", 18, "bold")
)

title.pack(pady=20)

load_button = tk.Button(
    root,
    text="Load TXT file",
    command=load_file,
    width=25,
    height=2
)

load_button.pack(pady=10)

file_label = tk.Label(root, text="No file selected")
file_label.pack(pady=10)

mode_menu = tk.OptionMenu(
    root,
    mode_var,
    "Correction",
    "Translation ITA -> ENG"
)

mode_menu.pack(pady=10)

process_button = tk.Button(
    root,
    text="Process text",
    command=process_file,
    width=25,
    height=2
)

process_button.pack(pady=10)

status_label = tk.Label(root, text="")
status_label.pack(pady=20)

root.mainloop()