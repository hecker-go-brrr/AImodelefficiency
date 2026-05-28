import os
import time
import re
import psutil
import chromadb
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

DATA_PATH = "data"  # 👈 LOCAL DATASET FOLDER

DB_PATH = "chroma_db"
COLLECTION_NAME = "vivekananda_docs"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 8

# -----------------------------
# INIT MODELS + DB
# -----------------------------

print("\n[INFO] Initializing models...\n")

embedder = SentenceTransformer(EMBED_MODEL)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------
# LOAD TEXT FROM LOCAL DATA FOLDER
# -----------------------------

def load_text(folder):
    text = ""

    for root, _, files in os.walk(folder):
        print("[DEBUG] Files found:", files)
        for f in files:
            path = os.path.join(root, f)

            # ----------------------
            # TXT FILES
            # ----------------------
            if f.endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    text += file.read() + "\n"

            # ----------------------
            # EXCEL FILES (NEW FIX)
            # ----------------------
            elif f.endswith(".xlsx"):
                try:
                    excel_file = pd.ExcelFile(path)

                    for sheet in excel_file.sheet_names:
                        df = excel_file.parse(sheet)

                        # convert all cells to text
                        for col in df.columns:
                            text += " ".join(df[col].astype(str).fillna("")) + "\n"

                except Exception as e:
                    print(f"[WARN] Failed to read {f}: {e}")
            # ----------------------
            # CSV FILES
            # ----------------------
            elif f.endswith(".csv"):
                try:
                    df = pd.read_csv(path)

                    for col in df.columns:
                        text += " ".join(df[col].astype(str).fillna("")) + "\n"

                except Exception as e:
                    print(f"[WARN] Failed to read CSV {f}: {e}")

    return text
# -----------------------------
# CHUNKING (SENTENCE SAFE)
# -----------------------------

def chunk_text(text, chunk_size=1500, overlap=300):

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        # if paragraph fits
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para

        else:
            # save current chunk
            chunks.append(current_chunk.strip())

            # preserve overlap
            overlap_text = current_chunk[-overlap:]

            # start new chunk
            current_chunk = overlap_text + "\n\n" + para

    # add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# -----------------------------
# INGESTION (ONLY ON FIRST RUN)
# -----------------------------

if collection.count() == 0:
    print("\n[INFO] Building vector database from local data...\n")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data folder '{DATA_PATH}' not found. Please add dataset inside it."
        )

    raw_text = load_text(DATA_PATH)

    if len(raw_text.strip()) == 0:
        raise ValueError("No text files found in data folder")

    print("[DEBUG] Sample text:\n", raw_text[:400])

    chunks = chunk_text(
        raw_text,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    print(f"[INFO] Total chunks: {len(chunks)}")

    print("[INFO] Generating embeddings (this may take time)...")

    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    BATCH_SIZE = 5000

    for i in range(0, len(chunks), BATCH_SIZE):

        batch_docs = chunks[i:i+BATCH_SIZE]
        batch_embeds = embeddings[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeds,
            ids=batch_ids
        )

        print(f"[INFO] Added batch {i // BATCH_SIZE + 1}")

    print("[DONE] Vector DB created.\n")

else:
    print("\n[INFO] Using existing vector database\n")

# -----------------------------
# METRICS
# -----------------------------

metrics_log = []
process = psutil.Process(os.getpid())
psutil.cpu_percent(interval=None)

# -----------------------------
# QUERY LOOP
# -----------------------------

while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # -------------------------
    # RETRIEVAL
    # -------------------------

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    retrieved = results.get("documents", [[]])[0]

    if not retrieved:
        retrieved = ["No relevant context found"]

    print("\n--- RETRIEVED CONTEXT ---")
    for i, r in enumerate(retrieved):
        print(f"\n[Chunk {i}]\n{r[:250]}")

    context = "\n\n".join(retrieved)

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
    You are a helpful RAG assistant.

    Answer using the provided context.

    Rules:
    - Prefer concise answers
    - Combine information across chunks
    - If the answer is partially present, summarize it
    - ONLY say 'Not found in provided documents'
      if the context is completely unrelated

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # -------------------------
    # METRICS START
    # -------------------------

    start_time = time.time()
    first_token_time = None
    response = ""

    # -------------------------
    # GENERATION (OLLAMA STREAM)
    # -------------------------

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print("\n--- ANSWER ---\n")

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()

        token = chunk["message"]["content"]
        response += token
        print(token, end="", flush=True)

    end_time = time.time()

    # -------------------------
    # METRICS CALCULATION
    # -------------------------

    ttft = max(0, (first_token_time or end_time) - start_time)
    latency = end_time - start_time

    token_count = len(response.split())
    tokens_per_sec = token_count / (latency + 1e-6)

    ram = process.memory_info().rss / 1e9
    cpu = psutil.cpu_percent(interval=0.5)

    metrics = {
        "ttft": ttft,
        "latency": latency,
        "tps": tokens_per_sec,
        "ram": ram,
        "cpu": cpu
    }

    metrics_log.append(metrics)

    # -------------------------
    # PRINT METRICS
    # -------------------------

    print("\n\n--- METRICS (THIS QUERY) ---")
    print(f"TTFT: {ttft:.3f}s")
    print(f"Latency: {latency:.3f}s")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print(f"RAM (process): {ram:.2f} GB")
    print(f"CPU: {cpu:.1f}%")

    # -------------------------
    # AVERAGE LAST 10
    # -------------------------

    recent = metrics_log[-10:]

    avg_ttft = sum(m["ttft"] for m in recent) / len(recent)
    avg_lat = sum(m["latency"] for m in recent) / len(recent)
    avg_tps = sum(m["tps"] for m in recent) / len(recent)

    print("\n--- AVERAGE (LAST 10) ---")
    print(f"Avg TTFT: {avg_ttft:.3f}s")
    print(f"Avg Latency: {avg_lat:.3f}s")
    print(f"Avg TPS: {avg_tps:.2f}")