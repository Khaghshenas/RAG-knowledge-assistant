from sentence_transformers import CrossEncoder
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, question, chunks):
        """
        question: str
        chunks: list of dicts, each with "text" key
        returns: chunks sorted by relevance (highest first)
        """
        pairs = [(question, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["score"] = float(score)

        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        return sorted_chunks
