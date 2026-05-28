import os
import time
import psutil
import chromadb
import ollama

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

DATA_FOLDER = "data"
DB_PATH = "chroma_db"
COLLECTION_NAME = "docs"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 3

# -----------------------------
# INIT
# -----------------------------

embedder = SentenceTransformer(EMBED_MODEL)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------
# PDF LOADER
# -----------------------------

def load_pdfs(folder):
    text = ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    return text

# -----------------------------
# IMPROVED CHUNKING
# -----------------------------

def chunk_text(text, size=800, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks

# -----------------------------
# INGESTION (RUN ONCE)
# -----------------------------

if collection.count() == 0:
    print("\n[INFO] Ingesting documents...\n")

    raw_text = load_pdfs(DATA_FOLDER)

    print("[DEBUG] Sample text:")
    print(raw_text[:500])

    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"[INFO] Total chunks: {len(chunks)}")

    embeddings = embedder.encode(chunks).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print("[DONE] Vector DB created.\n")

else:
    print("\n[INFO] Using existing vector DB\n")

# -----------------------------
# QUERY LOOP
# -----------------------------

metrics_log = []

while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # -------------------------
    # RETRIEVAL
    # -------------------------
    query_embedding = embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    retrieved = results["documents"][0]

    print("\n--- RETRIEVED CHUNKS ---")
    for i, r in enumerate(retrieved):
        print(f"\n[Chunk {i}]\n{r[:300]}")

    context = "\n\n".join(retrieved)


    prompt = f"""
    You are a precise QA assistant for technical documents.

    INSTRUCTIONS:
    - Use ONLY the context below
    - If the answer is partially present, infer carefully from context
    - Do NOT say "Not found" if related information exists
    - If unsure, give best possible answer based on context
    
    Context:
    {context}

    Question:
    {query}
    """

    # -------------------------
    # METRICS START
    # -------------------------
    start = time.time()
    first_token_time = None
    response = ""

    ram_before = psutil.virtual_memory().used / 1e9

    # -------------------------
    # GENERATION (STREAM)
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

    end = time.time()

    # -------------------------
    # METRICS CALCULATION
    # -------------------------
    ttft = first_token_time - start
    latency = end - start

    token_count = len(response.split())
    tokens_per_sec = token_count / (latency - ttft + 1e-6)

    ram_after = psutil.virtual_memory().used / 1e9
    cpu = psutil.cpu_percent(interval=0.5)

    metrics = {
        "ttft": ttft,
        "latency": latency,
        "tps": tokens_per_sec,
        "ram": ram_after,
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
    print(f"RAM: {ram_after:.2f} GB")
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