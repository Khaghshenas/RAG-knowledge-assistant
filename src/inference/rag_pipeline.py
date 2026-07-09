from src.inference.generation.llm import HFGenerator
from src.inference.generation.prompt import build_prompt
from src.inference.retrieval.re_ranking import CrossEncoderReranker
from src.inference.retrieval.search import FaissRetriever
from src.scripts.utils import load_config


class RAGPipeline:
    def __init__(
        self,
        config=None,
        retriever: FaissRetriever = None,
        generator: HFGenerator = None,
        reranker: CrossEncoderReranker = None
        ):

        self.config = config or load_config()
        self.retriever = retriever or FaissRetriever(config)
        self.reranker = reranker or CrossEncoderReranker(config)
        self.generator = generator or HFGenerator(config)
        
        self.initial_k = config['models']['retriever']['top_k_retrieval']
        self.final_k = config['models']['retriever']['top_k_final']
        

    def answer(self, question: str):
        # 1. Retrieve documents, FAISS retrieval
        retrieved_chunks = self.retriever.search(question)

        # 2. Cross-encoder re-ranking
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks)

        # 3. Keep only top-k after re-ranking
        final_chunks = reranked_chunks[:self.final_k]

        # 4. Build prompt
        prompt = build_prompt(question, final_chunks)

        # 5 Generate answer
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "contexts": final_chunks,
            "metadata": {
                "initial_retrieval_count": len(retrieved_chunks),
                "final_context_count": len(final_chunks)
            }
        }
