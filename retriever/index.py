import faiss
import numpy as np
import pickle
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = (BASE_DIR / "../data/processed/chunk_embeddings.npz").resolve()
INDEX_PATH = (BASE_DIR / "../data/processed/faiss.index").resolve()
METADATA_PATH = (BASE_DIR / "../data/processed/faiss_metadata.pkl").resolve()


def load_embeddings():
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data["embeddings"].astype("float32")
    chunk_ids = data["chunk_ids"]
    texts = data["texts"]
    metadata = data["metadata"]
    return embeddings, chunk_ids, texts, metadata


def build_faiss_index(embeddings):
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index using cosine similarity (inner product).
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


def save_index(index, chunk_ids, texts, metadata):
    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(
            {
                "chunk_ids": chunk_ids,
                "texts": texts,
                "metadata": metadata,
            },
            f,
        )


def load_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Build it first.")
    
    index = faiss.read_index(str(INDEX_PATH).replace("\\", "/"))

    with open(METADATA_PATH, "rb") as f:
        data = pickle.load(f)

    return (
        index,
        data["chunk_ids"],
        data["texts"],
        data["metadata"],
    )


if __name__ == "__main__":
    
    print("Building FAISS index...")

    embeddings, chunk_ids, texts, metadata = load_embeddings()
    index = build_faiss_index(embeddings)
    save_index(index, chunk_ids, texts, metadata)

    print(f"Index built with {index.ntotal} vectors")