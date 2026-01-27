import json
from pathlib import Path
from tqdm import tqdm

def extract_contexts(input_file: Path, output_file: Path):
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

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON Lines
    with output_file.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent
    input_path = (BASE_DIR / "../data/raw/squad_v2/train.json").resolve()
    output_path = (BASE_DIR / "../data/processed/documents.jsonl").resolve()

    extract_contexts(input_path, output_path)