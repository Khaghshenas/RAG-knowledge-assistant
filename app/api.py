import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_pipeline_manager
from src.scripts.utils import setup_logging

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


@app.on_event("startup")
def startup_event():
    """Initialize the RAG pipeline once when the server starts."""
    logger.info("FastAPI startup event triggered")
    manager = get_pipeline_manager()
    manager.initialize()
    logger.info("RAG pipeline is ready for requests")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Process a query using the shared RAG pipeline."""
    try:
        logger.info("Received query: %s", request.question)
        
        manager = get_pipeline_manager()
        result = manager.pipeline.answer(request.question)
        
        return result

    except Exception as e:
        logger.exception("Error during inference")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.get("/health")
def health():
    """Health check endpoint."""
    manager = get_pipeline_manager()
    return {
        "status": "ok",
        "pipeline_ready": manager.is_ready()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )