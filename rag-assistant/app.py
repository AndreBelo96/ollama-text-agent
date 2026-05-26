import tkinter as tk
from tkinter import scrolledtext
import threading
from main import setup_rag


class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ACME RAG Assistant")
        self.root.geometry("800x700")
        self.index = None
        self.build_ui()
        self.load_rag()

    def build_ui(self):
        # Header
        tk.Label(
            self.root,
            text="🔍 ACME RAG Assistant",
            font=("Arial", 18, "bold")
        ).pack(pady=10)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Arial", 11),
            state=tk.DISABLED,
            height=18
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Tag colori per chat
        self.chat_display.tag_config("you", foreground="#2255cc", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground="#1a7a1a", font=("Arial", 11))
        self.chat_display.tag_config("sources", foreground="#888888", font=("Arial", 9, "italic"))
        self.chat_display.tag_config("separator", foreground="#cccccc")
        self.chat_display.tag_config("valid", foreground="#1a7a1a", font=("Arial", 10))
        self.chat_display.tag_config("warning", foreground="#cc7700", font=("Arial", 10))
        self.chat_display.tag_config("error", foreground="#cc0000", font=("Arial", 10))

        # Pipeline status frame
        status_outer = tk.LabelFrame(
            self.root,
            text="Pipeline Status",
            font=("Arial", 10)
        )
        status_outer.pack(fill=tk.X, padx=15, pady=5)

        self.pipeline_labels = {}
        steps = [
            ("queries", "Generating queries"),
            ("retrieval", "Retrieving chunks"),
            ("reranking", "Reranking"),
            ("generation", "Generating answer"),
            ("validation", "Validating answer"),
        ]

        for key, label in steps:
            row = tk.Frame(status_outer)
            row.pack(anchor="w", padx=10, pady=1)
            tk.Label(row, text=label, font=("Arial", 10), width=22, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="⏸ Idle", font=("Arial", 10), fg="gray", width=15, anchor="w")
            lbl.pack(side=tk.LEFT)
            self.pipeline_labels[key] = lbl

        # Input area
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=15, pady=10)

        self.user_input = tk.Text(
            input_frame,
            height=2,
            font=("Arial", 11),
            wrap=tk.WORD
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", lambda e: (self.send(), "break")[1])
        self.user_input.bind("<Shift-Return>", lambda e: None)

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send,
            width=8,
            height=2,
            state=tk.DISABLED
        )
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

    def set_step(self, key, status, color):
        self.root.after(0, lambda: self.pipeline_labels[key].config(text=status, fg=color))

    def reset_steps(self):
        for key in self.pipeline_labels:
            self.set_step(key, "⏸ Idle", "gray")

    def append_chat(self, text, tag):
        def update():
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.insert(tk.END, text, tag)
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)
        self.root.after(0, update)

    def load_rag(self):
        self.append_chat("Setting up RAG pipeline...\n", "sources")

        def run():
            self.index = setup_rag()
            self.root.after(0, self.on_rag_ready)

        threading.Thread(target=run, daemon=True).start()

    def on_rag_ready(self):
        self.append_chat("RAG Assistant ready! Ask me anything about ACME Tech.\n\n", "sources")
        self.append_chat("─" * 60 + "\n\n", "separator")
        self.send_button.config(state=tk.NORMAL)
        self.user_input.focus()

    def send(self):
        question = self.user_input.get("1.0", tk.END).strip()
        if not question or self.index is None:
            return

        self.user_input.delete("1.0", tk.END)
        self.send_button.config(state=tk.DISABLED)
        self.reset_steps()

        self.append_chat(f"You: {question}\n\n", "you")

        threading.Thread(
            target=self.run_pipeline,
            args=(question,),
            daemon=True
        ).start()

    def run_pipeline(self, question):
        try:
            # Step 1: queries
            self.set_step("queries", "⏳ Running...", "orange")
            from main import generate_queries, retrieve, rerank, build_context, build_prompt, generate

            queries = generate_queries(question)
            queries.insert(0, question)
            self.set_step("queries", f"✅ {len(queries)} queries", "green")

            # Step 2: retrieval
            self.set_step("retrieval", "⏳ Running...", "orange")
            from main import deduplicate_nodes
            retriever = self.index.as_retriever(similarity_top_k=3)
            all_nodes = []
            for q in queries:
                all_nodes.extend(retriever.retrieve(q))
            nodes = deduplicate_nodes(all_nodes)
            self.set_step("retrieval", f"✅ {len(nodes)} chunks", "green")

            # Step 3: reranking
            self.set_step("reranking", "⏳ Running...", "orange")
            nodes = rerank(nodes, question)
            self.set_step("reranking", f"✅ {len(nodes)} chunks", "green")

            # Step 4: generation
            self.set_step("generation", "⏳ Running...", "orange")
            context = build_context(nodes)
            prompt = build_prompt(question, context)
            answer = generate(prompt)
            self.set_step("generation", "✅ Done", "green")

            # Step 5: validation
            self.set_step("validation", "⏳ Running...", "orange")
            from main import validate
            verdict, reason = validate(answer, context, question)

            color = "green" if verdict == "VALID" else "orange" if verdict == "INCOMPLETE" else "red"
            self.set_step("validation", f"✅ {verdict}", color)

            # Mostra risposta
            self.root.after(0, lambda: self.show_answer(answer, nodes, verdict, reason))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def show_answer(self, answer, nodes, verdict, reason):
        self.append_chat(f"Assistant:\n{answer}\n\n", "assistant")

        # mostra verdict
        if verdict == "VALID":
            self.append_chat(f"✅ Validated\n", "valid")
        elif verdict == "INCOMPLETE":
            self.append_chat(f"⚠️ Incomplete: {reason}\n", "warning")
        else:
            self.append_chat(f"❌ Invalid: {reason}\n", "error")

        sources = ", ".join([
            f"{n.metadata.get('file_name', 'unknown')} ({round(n.score, 2)})"
            for n in nodes
        ])
        self.append_chat(f"Sources: {sources}\n", "sources")
        self.append_chat("\n" + "─" * 60 + "\n\n", "separator")

        self.send_button.config(state=tk.NORMAL)
        self.reset_steps()

    def show_error(self, error):
        self.append_chat(f"Error: {error}\n\n", "sources")
        self.send_button.config(state=tk.NORMAL)
        self.reset_steps()

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()