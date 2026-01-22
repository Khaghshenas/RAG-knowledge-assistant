import json
import os
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CHUNK_FILE = (BASE_DIR / "../data/processed/chunks.jsonl").resolve() 
EMBEDDINGS_FILE = (BASE_DIR / "../data/processed/chunk_embeddings.npz").resolve() 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

chunks = []
chunk_ids = []
metadata = []

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        chunks.append(obj["text"])
        chunk_ids.append(obj["chunk_id"])
        metadata.append(obj["metadata"])

print(f"Loaded {len(chunks)} chunks")


embeddings = []

for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch_texts = chunks[i:i+BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.append(batch_embeddings)

embeddings = np.vstack(embeddings)
print(f"Embeddings shape: {embeddings.shape}")


np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings, chunk_ids=np.array(chunk_ids), metadata=np.array(metadata))
print(f"Saved embeddings to {EMBEDDINGS_FILE}")
