import os
import sys
from pathlib import Path

from src.inference.rag_pipeline import RAGPipeline
from src.scripts.utils import load_config, setup_logging

config = load_config()
rag = RAGPipeline(config)

sample_questions = [
    "What is the official language of Brazil?",
    "What is the capital of France?",
    "Who is the author of the Harry Potter series?",
    "What is the name of Beyonce's younger sister?",
    "What is the capital of Japan?"
]

predictions = []

for q in sample_questions:
    result = rag.answer(q)
    predictions.append({
        "question": q,
        "answer": result["answer"],
        "sources": [c["metadata"] for c in result["contexts"]]
    })

# Display predictions
for p in predictions:
    logger.info("Q: %s", p["question"])
    logger.info("A: %s", p["answer"])
    logger.info("%s", "-" * 50)