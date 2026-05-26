from dotenv import load_dotenv
from config.settings import *

import requests
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────

load_dotenv()

# ─────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────

def setup_models():
    """
    Configura:
    - LLM
    - embedding model
    - chunking globale
    """

    print("Loading models...")

    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        request_timeout=120
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL
    )

    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

# ─────────────────────────────────────────────────────────────
# VECTOR STORE SETUP
# ─────────────────────────────────────────────────────────────

def setup_vector_store():
    """ Configura ChromaDB. """

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    chroma_collection = chroma_client.get_or_create_collection(
        "company_docs"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    return (
        chroma_collection,
        vector_store,
        storage_context
    )

# ─────────────────────────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────

def load_documents():
    """ Carica documenti dalla cartella. """

    print("Loading documents...")

    documents = SimpleDirectoryReader(
        str(DOCUMENTS_PATH)
    ).load_data()

    print(f"Loaded {len(documents)} documents")

    return documents

# ─────────────────────────────────────────────────────────────
# INDEX BUILDING
# ─────────────────────────────────────────────────────────────

def build_index(storage_context):
    """
    Costruisce l'indice: documenti -> chunk -> embedding -> ChromaDB
    """

    print("Building index...")

    documents = load_documents()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    return index

# ─────────────────────────────────────────────────────────────
# LOAD EXISTING INDEX
# ─────────────────────────────────────────────────────────────

def load_existing_index( vector_store, storage_context ):
    """ Carica indice già esistente. """

    print("Loading existing index...")

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )

# ─────────────────────────────────────────────────────────────
# MAIN RAG SETUP
# ─────────────────────────────────────────────────────────────

def setup_rag():
    """ Setup completo pipeline RAG. """

    print("Setting up RAG pipeline...\n")

    # setup modelli
    setup_models()

    # setup db vettoriale
    (
        chroma_collection,
        vector_store,
        storage_context
    ) = setup_vector_store()

    # se indice già presente
    if chroma_collection.count() > 0:

        print(
            f"Found existing DB with "
            f"{chroma_collection.count()} chunks\n"
        )

        index = load_existing_index(
            vector_store,
            storage_context
        )

    # altrimenti crea tutto
    else:

        print("No existing index found\n")

        index = build_index(
            storage_context
        )

        print(
            f"\nIndex built successfully!"
            f"\nSaved {chroma_collection.count()} chunks\n"
        )

    return index

# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

def retrieve(index, question):
    """ Recupera chunk semanticamente rilevanti. """

    retriever = index.as_retriever(
        similarity_top_k=TOP_K
    )

    # query alternative
    queries = generate_queries(question)

    # includi anche originale
    queries.insert(0, question)

    all_nodes = []

    for query in queries:

        nodes = retriever.retrieve(query)

        all_nodes.extend(nodes)

    return deduplicate_nodes(all_nodes)

def generate_queries(question):
    prompt = f"""
    You are an AI retrieval assistant.

    Your task is to generate multiple different
    versions of the user's question for semantic
    document retrieval.

    The goal is to overcome limitations of
    vector similarity search.

    Rules:
    - Generate 5 alternative queries
    - Preserve the original meaning
    - Use different wording and phrasing
    - Include both technical and natural variants
    - Keep queries concise
    - Output ONLY the queries
    - One query per line

    User question:
    {question}
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    text = response.json()["message"]["content"]

    queries = [
        q.strip()
        for q in text.split("\n")
        if q.strip()
    ]

    return queries

def deduplicate_nodes(nodes):
    unique = {}

    for node in nodes:

        key = node.text.strip()

        # tieni score migliore
        if (
                key not in unique
                or node.score > unique[key].score
        ):
            unique[key] = node

    return list(unique.values())

# ─────────────────────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────────────────────

def rerank(nodes, question, top_n=3):
    """
    Riordina i chunk per rilevanza reale.
    Chiede all'LLM di valutare ogni chunk.
    """
    if not nodes:
        return nodes

    scored = []

    for node in nodes:
        prompt = f"""Rate how relevant this text is to answer the question.
        Answer with ONLY a number from 0 to 10.
        
        Question: {question}
        
        Text: {node.text[:500]}
        
        Relevance score (0-10):"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        raw = response.json()["message"]["content"].strip()

        # estrai numero dalla risposta
        try:
            score = float(''.join(c for c in raw if c.isdigit() or c == '.'))
        except:
            score = 0.0

        scored.append((node, score))

    # ordina per score decrescente
    scored.sort(key=lambda x: x[1], reverse=True)

    return [node for node, _ in scored[:top_n]]

# ─────────────────────────────────────────────────────────────
# CONTEXT BUILDING
# ─────────────────────────────────────────────────────────────

def build_context(nodes):
    """
    Costruisce contesto leggibile per il prompt.
    """

    context_parts = []

    for i, node in enumerate(nodes):

        # metadata
        source = node.metadata.get(
            "file_name",
            "unknown"
        )

        # similarity score
        score = round(node.score, 3)

        context_parts.append(
            f"""
[Source {i+1}]

File:
{source}

Relevance Score:
{score}

Content:
{node.text}
"""
        )

    return "\n\n---\n\n".join(context_parts)

# ─────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────

def build_prompt(question, context):
    """ Costruisce prompt finale. """

    return f"""
    You are a strict RAG assistant.
    
    Rules:
    - Answer ONLY using the provided context
    - Never invent information
    - If information is missing say:
      "I don't have that information."
    - Be concise and precise
    - Cite sources when useful

    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

def generate(prompt):
    """ Chiamata diretta a Ollama. """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]

# ─────────────────────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────────────────────

def validate(answer, context, question):
    """
    Valida la risposta generata.
    Controlla se è basata sul contesto o ha inventato.
    """

    prompt = f"""You are a strict answer validator for a RAG system.

    Your job is to check if the answer is grounded in the provided context.
    
    Rules:
    - Reply with VALID if the answer is fully supported by the context
    - Reply with INVALID if the answer contains information not in the context
    - Reply with INCOMPLETE if the question cannot be answered with the context
    - After the verdict, add a brief reason (one line)
    
    Format:
    VERDICT: <VALID|INVALID|INCOMPLETE>
    REASON: <brief explanation>
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer to validate:
    {answer}
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()

    # parsing verdict
    verdict = "UNKNOWN"
    reason = ""

    for line in raw.split("\n"):
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return verdict, reason
