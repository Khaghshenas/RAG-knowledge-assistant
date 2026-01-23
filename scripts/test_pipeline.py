import sys
import os

# Add Pipeline root to PYTHONPATH
PIPELINE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pipeline/"))
sys.path.insert(0, PIPELINE_PATH)

from rag_pipeline import RAGPipeline

rag = RAGPipeline(top_k=5)

question = "What is the name of Beyonce's younger sister?"

result = rag.answer(question)

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for ctx in result["contexts"]:
    print("-", ctx["metadata"])
