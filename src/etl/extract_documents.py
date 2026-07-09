import json
from pathlib import Path

from tqdm import tqdm

from src.scripts.utils import load_config


def extract_contexts(config=None):

    config = config or load_config()
    input_file = Path(config['paths']['raw_data_path']) / "train.json"
    output_file = Path(config['paths']['documents_path'])

    seen_contexts = set()
    docs = []

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load JSON Lines
    data = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # Extract contexts
    for item in tqdm(data, desc="Processing examples"):
        context = item["context"].strip()

        if context not in seen_contexts:
            seen_contexts.add(context)
            docs.append({
                "text": context,
                "source": "squad_v2",
                "title": item["title"]
            })

    # Write to JSON Lines
    with output_file.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return output_file, len(docs)
    
if __name__ == "__main__":
    extract_contexts()