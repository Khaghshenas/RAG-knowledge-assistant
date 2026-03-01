import sys
import os

from src.inference.rag_pipeline import RAGPipeline
from src.evaluation.retrieval_metrics import recall_at_k
from src.evaluation.rag_metrics import evaluate_rag

def run_evaluation(eval_set):
    rag = RAGPipeline(top_k=5)

    recall = recall_at_k(eval_set, rag.retriever, rag.reranker, k=5)
    rag_scores = evaluate_rag(rag, eval_set, max_samples=100)

    print("Recall@5:", recall)
    print("RAG EM:", rag_scores["EM"])
    print("RAG F1:", rag_scores["F1"])
