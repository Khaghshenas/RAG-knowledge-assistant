from src.inference.rag_pipeline import RAGPipeline
from src.scripts.utils import load_config

def test_pipeline_returns_answer(monkeypatch):

    config = load_config()

    class DummyRetriever:
        def __init__(self, config):
            pass

        def search(self, question):
            return [
                {
                    "text": "The capital of France is Paris.",
                    "metadata": {"source": "test_document"}
                }
            ]

    class DummyReranker:
        def __init__(self, config):
            pass

        def rerank(self, question, chunks):
            return chunks

    class DummyGenerator:
        def __init__(self, config):
            pass

        def generate(self, prompt):
            return "The capital of France is Paris."

    monkeypatch.setattr("src.inference.rag_pipeline.FaissRetriever", DummyRetriever)

    monkeypatch.setattr("src.inference.rag_pipeline.CrossEncoderReranker", DummyReranker)

    monkeypatch.setattr("src.inference.rag_pipeline.HFGenerator", DummyGenerator)

    rag = RAGPipeline(config=config)

    result = rag.answer("What is the capital of France?")

    assert result["answer"] == "The capital of France is Paris."
    assert len(result["contexts"]) == 1
    assert result["contexts"][0]["metadata"]["source"] == "test_document"