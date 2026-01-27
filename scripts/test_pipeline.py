import sys
import os

# Add Pipeline root to PYTHONPATH
PIPELINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipeline/"))
sys.path.insert(0, PIPELINE_PATH)

from rag_pipeline import RAGPipeline

rag = RAGPipeline(top_k=5)

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
    print("Q:", p["question"])
    print("A:", p["answer"])
    #print("Sources:", p["sources"])
    print("-" * 50)