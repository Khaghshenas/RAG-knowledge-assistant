import logging
import sys
from pathlib import Path

from datasets import load_dataset

from src.scripts.utils import load_config
from src.scripts.utils import setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

def download_squad(config=None):
    config = config or load_config()
    output_path = Path(config['paths']['raw_data_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset = load_dataset("squad_v2")

    # Save each split as a JSON file
    for split in dataset:
        path = output_path / f"{split}.json"
        dataset[split].to_json(path)
        print(f"Saved {split} to {path}")
    
    return output_path   

if __name__ == "__main__":
    download_squad()
