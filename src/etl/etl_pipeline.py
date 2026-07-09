import logging
import sys
import time
import argparse

from src.etl.download_data import download_squad
from src.etl.extract_documents import extract_contexts
from src.etl.chunk_documents import process_documents
from src.etl.embeddings import generate_embeddings
from src.etl.index import build_faiss_index
from src.scripts.utils import setup_logging, load_config

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

def run_pipeline(config_path: str) -> None:
    
    config = load_config(config_path)

    
    logger.info("--- Starting Offline ETL Pipeline ---")
    start_time = time.time()

    try:
        # 0. Download
        logger.info("Stage 0: Downloading SQuAD_V2 dataset...")
        raw_data_path = download_squad(config)
        logger.info(f"Successfully downloaded and saved to {raw_data_path}.")

        # 1. Extraction
        logger.info("Stage 1: Extracting data from SQuAD source...")
        documents_path, documents_num = extract_contexts(config)
        logger.info(f"Successfully extracted {documents_num} records at {documents_path}.")

        # 2. Chunking
        logger.info("Stage 2: Chunking text for RAG...")
        chunks_path = process_documents(config)
        logger.info(f"Created text chunks at {chunks_path}.")

        # 3. Embedding
        logger.info("Stage 3: Generating embeddings (this may take a while)...")
        embeddings_file_path = generate_embeddings(config)
        logger.info(f"Embeddings generated and saved to {embeddings_file_path} successfully.")

        # 4. Indexing
        logger.info("Stage 4: Building and saving the vector index...")
        index = build_faiss_index(config)
        logger.info(f"Index built and saved to {config['paths']['index_path']}.")

        logger.info("--- ETL Pipeline Completed Successfully ---")

    except Exception as e:
        logger.critical(f"Pipeline failed at a critical step: {e}", exc_info=True)
        sys.exit(1)

    total_time = (time.time() - start_time) / 60
    logger.info(f"--- ETL Completed in {total_time:.2f} minutes ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG ETL Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    run_pipeline(args.config)