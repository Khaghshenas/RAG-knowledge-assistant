import faiss
from sentence_transformers import SentenceTransformer

from src.etl.index import load_index
from src.scripts.utils import load_config

class FaissRetriever:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.model_name = self.config['models']['retriever']['model_name']
        self.top_k = self.config['models']['retriever']['top_k_retrieval']

        self.model = SentenceTransformer(self.model_name)

        self.index, self.chunk_ids, self.texts, self.metadata = load_index()

    def encode_query(self, query: str):
        embedding = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding

    def search(self, query: str, top_k=None):
        k = top_k or self.top_k
        query_embedding = self.encode_query(query)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.texts[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })

        return results
