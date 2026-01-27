import sys
import os

# Add Retriever root to PYTHONPATH
RETRIEVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../retriever/"))
sys.path.insert(0, RETRIEVER_PATH)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from index import load_index

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class FaissRetriever:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.model = SentenceTransformer(MODEL_NAME)

        self.index, self.chunk_ids, self.texts, self.metadata = load_index()

    def encode_query(self, query: str):
        embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding

    def search(self, query: str, topk):
        query_embedding = self.encode_query(query)

        scores, indices = self.index.search(query_embedding, topk)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.texts[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })

        return results
