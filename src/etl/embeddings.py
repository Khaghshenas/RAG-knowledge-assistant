import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.scripts.utils import load_config, setup_logging


def generate_embeddings(config=None):
    
    # Configuration Setup
    config = config or load_config()
    chunk_file = Path(config['paths']['chunks_path'])
    embeddings_file = Path(config['paths']['embeddings_path'])
    model_name = config['models']['retriever']['model_name']
    batch_size = config['models']['retriever']['batch_size']


    if not chunk_file.exists():
        logger.error(f"Chunk file not found at {chunk_file}. Run chunking first.")
        return False

    # Load Model
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer: {e}")
        raise

    # Load Chunks
    chunks = []
    chunk_ids = []
    metadata = []

    logger.info(f"Reading chunks from {chunk_file}...")
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                chunks.append(obj["text"])
                chunk_ids.append(obj["chunk_id"])
                metadata.append(obj["metadata"])
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(chunks)} chunks for embedding.")

    # Batch Processing
    all_embeddings = []
    logger.info(f"Starting embedding with batch_size={batch_size}...")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding Progress"):
        batch_texts = chunks[i : i + batch_size]
        try:
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            logger.error(f"Error encoding batch at index {i}: {e}")
            raise

    # Finalize and Save
    final_embeddings = np.vstack(all_embeddings)
    logger.info(f"Final embeddings shape: {final_embeddings.shape}")

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        embeddings_file,
        embeddings=final_embeddings,
        chunk_ids=np.array(chunk_ids),
        texts=np.array(chunks),
        metadata=np.array(metadata),
    )
    
    logger.info(f"Saved compressed embeddings to {embeddings_file}")
    return embeddings_file

if __name__ == "__main__":
    generate_embeddings()