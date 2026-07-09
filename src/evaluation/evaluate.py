import logging

from src.inference.rag_pipeline import RAGPipeline
from src.evaluation.retrieval_metrics import recall_at_k
from src.evaluation.rag_metrics import evaluate_rag
from src.scripts.utils import setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)


def run_evaluation(eval_set):
    rag = RAGPipeline(top_k=5)

    recall = recall_at_k(eval_set, rag.retriever, rag.reranker, k=5)
    rag_scores = evaluate_rag(rag, eval_set, max_samples=100)

    logger.info("Recall@5: %s", recall)
    logger.info("RAG EM: %s", rag_scores["EM"])
    logger.info("RAG F1: %s", rag_scores["F1"])
