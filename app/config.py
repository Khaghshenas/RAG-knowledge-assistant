import logging
from typing import Optional
from src.inference.rag_pipeline import RAGPipeline
from src.scripts.utils import load_config

logger = logging.getLogger(__name__)


class PipelineManager:
    
    _instance: Optional['PipelineManager'] = None
    _initialized: bool = False
    
    _config: Optional[dict] = None
    _pipeline: Optional[RAGPipeline] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self) -> None:
        """Initialize the pipeline and configuration once at startup."""
        if self._initialized:
            logger.warning("Pipeline already initialized, skipping re-initialization")
            return
        
        try:
            logger.info("Loading configuration...")
            self._config = load_config()
            
            logger.info("Initializing RAG pipeline...")
            self._pipeline = RAGPipeline(config=self._config)
            
            self._initialized = True
            logger.info("RAG pipeline successfully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise
    
    @property
    def pipeline(self) -> RAGPipeline:
        """Get the initialized pipeline. Raises if not yet initialized."""
        if not self._initialized or self._pipeline is None:
            raise RuntimeError(
                "Pipeline not initialized. Call PipelineManager().initialize() first."
            )
        return self._pipeline
    
    @property
    def config(self) -> dict:
        """Get the loaded configuration. Raises if not yet initialized."""
        if self._config is None:
            raise RuntimeError(
                "Configuration not loaded. Call PipelineManager().initialize() first."
            )
        return self._config
    
    def is_ready(self) -> bool:
        """Check if the pipeline is initialized and ready."""
        return self._initialized and self._pipeline is not None


def get_pipeline_manager() -> PipelineManager:
    """Factory function to get the pipeline manager singleton."""
    return PipelineManager()