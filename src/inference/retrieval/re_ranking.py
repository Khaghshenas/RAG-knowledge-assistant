import logging
import sys
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from src.scripts.utils import load_config, setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, config=None, device=None):
        self.config = config or load_config()
        self.model_name = self.config['models']['reranker']['model_name']
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, question: str, chunks: list):
        """
        question: str
        chunks: list of dicts, each with "text" key
        returns: chunks sorted by relevance (highest first)
        """
        if not chunks:
            return []
            
        pairs = [(question, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["score"] = float(score)

        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        return sorted_chunks
