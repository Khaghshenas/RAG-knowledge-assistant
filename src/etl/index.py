import logging
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

from src.scripts.utils import load_config, setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)


def load_embeddings(config=None):
    config = config or load_config()
    embedding_path = Path(config['paths']['embeddings_path'])
    data = np.load(embedding_path, allow_pickle=True)
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


def save_index(index, chunk_ids, texts, metadata, config=None):

    config = config or load_config()
    index_path = Path(config['paths']['index_path'])
    metadata_path = Path(config['paths']['metadata_path'])
    faiss.write_index(index, str(index_path))

    with open(metadata_path, "wb") as f:
        pickle.dump(
            {
                "chunk_ids": chunk_ids,
                "texts": texts,
                "metadata": metadata,
            },
            f,
        )


def load_index(config=None):
    config = config or load_config()
    metadata_path = Path(config['paths']['metadata_path'])
    index_path = Path(config['paths']['index_path'])
    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found. Build it first.")
    
    index = faiss.read_index(str(index_path).replace("\\", "/"))

    with open(metadata_path, "rb") as f:
        data = pickle.load(f)

    return (
        index,
        data["chunk_ids"],
        data["texts"],
        data["metadata"],
    )


if __name__ == "__main__":
    
    logger.info("Building FAISS index...")

    embeddings, chunk_ids, texts, metadata = load_embeddings()
    index = build_faiss_index(embeddings)
    save_index(index, chunk_ids, texts, metadata)

    logger.info("Index built with %d vectors", index.ntotal)