import json
from pathlib import Path
import logging

from src.inference.rag_pipeline import RAGPipeline
from src.evaluation.retrieval_metrics import recall_at_k
from src.evaluation.rag_metrics import evaluate_rag
from src.scripts.utils import setup_logging, load_config

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)


def run_evaluation(eval_set):
    rag = RAGPipeline()

    recall = recall_at_k(eval_set, rag.retriever, rag.reranker, k=5)
    rag_scores = evaluate_rag(rag, eval_set, max_samples=100)

    logger.info("Recall@5: %s", recall)
    logger.info("RAG EM: %s", rag_scores["EM"])
    logger.info("RAG F1: %s", rag_scores["F1"])

def load_eval_data(path):
    eval_set = []

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)

        # JSONL format
        if first_line.startswith("{"):

            for line in f:
                if line.strip():
                    item = json.loads(line)

                    eval_set.append(item)


    return eval_set


if __name__ == "__main__":

    config = load_config()

    eval_path = Path(config['paths']['raw_data_path']) / "train.json"
    eval_set = load_eval_data(eval_path)

    logger.info("Loaded evaluation samples: %d", len(eval_set))
    
    print("hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    run_evaluation(eval_set)