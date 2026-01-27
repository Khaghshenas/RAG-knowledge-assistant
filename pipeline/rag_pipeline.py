import sys
import os

# Add Generator and Retriever pathes to PYTHONPATH
GENERATOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../generator/"))
RETRIEVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../retriever/"))
sys.path.insert(0, GENERATOR_PATH)
sys.path.insert(0, RETRIEVER_PATH)

from search import FaissRetriever
from llm import HFGenerator
from prompt import build_prompt
from re_ranking import CrossEncoderReranker


class RAGPipeline:
    def __init__(
        self,
        retriever: FaissRetriever = None,
        generator: HFGenerator = None,
        top_k: int = 5,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    ):
        self.retriever = retriever or FaissRetriever(top_k=top_k)
        self.generator = generator or HFGenerator()
        self.top_k = top_k
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)

    def answer(self, question: str):
        # 1. Retrieve documents, FAISS retrieval
        retrieved_chunks = self.retriever.search(question, 20)

        # 2. Cross-encoder re-ranking
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks)

        # 3. Keep only top-k after re-ranking
        final_chunks = reranked_chunks[:self.top_k]

        # 4. Build prompt
        prompt = build_prompt(question, final_chunks)

        # 5 Generate answer
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "contexts": final_chunks,
        }
