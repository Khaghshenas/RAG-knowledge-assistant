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


class RAGPipeline:
    def __init__(
        self,
        retriever: FaissRetriever = None,
        generator: HFGenerator = None,
        top_k: int = 5,
    ):
        self.retriever = retriever or FaissRetriever(top_k=top_k)
        self.generator = generator or HFGenerator()
        self.top_k = top_k

    def answer(self, question: str):
        # 1. Retrieve documents
        retrieved_chunks = self.retriever.search(question)

        # 2. Build prompt
        prompt = build_prompt(question, retrieved_chunks)

        # 3. Generate answer
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved_chunks,
        }
