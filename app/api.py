import sys
import logging
import yaml
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.rag_pipeline import RAGPipeline
from src.scripts.utils import setup_logging, load_config

# Setup
setup_logging()
logger = logging.getLogger("api")
app = FastAPI(
    title="RAG QA API",
    description="Retrieval-Augmented Generation Question Answering Service",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, example="What is the capital of France?")


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list
    metadata: dict

rag_pipeline: RAGPipeline = None

@app.on_event("startup")
def startup_event():
    """Load configurations and initialize the inference pipeline once when the server starts."""
    global rag_pipeline

    logger.info("Loading configuration...")
    config = load_config()

    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(config=config)

    logger.info("RAG pipeline ready.")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        logger.info("Received query: %s", request.question)

        result = rag_pipeline.answer(request.question)

        return result

    except Exception as e:
        logger.exception("Error during inference")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
