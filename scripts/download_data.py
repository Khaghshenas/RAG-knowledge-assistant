from datasets import load_dataset
from pathlib import Path

def download_squad(output_dir: Path):
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset = load_dataset("squad_v2")

    # Save each split as a JSON file
    for split in dataset:
        path = output_dir / f"{split}.json"
        dataset[split].to_json(path)
        print(f"Saved {split} to {path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    output_dir = (BASE_DIR / "../data/raw/squad_v2").resolve()
    download_squad(output_dir)
