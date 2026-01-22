# TODO: Check chunk size and overlap later
# NOTE: Might need to deduplicate chunks if many repeats

import json
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import re
from clean_text import clean_text

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

def chunk_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def process_documents(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as out:

        for line in tqdm(f, desc="Processing documents"):
            doc = json.loads(line)
            text = clean_text(doc["text"])
            chunks = chunk_text(text)

            safe_title = re.sub(r"\W+", "_", doc["title"])
            for i, chunk in enumerate(chunks):
                out.write(json.dumps({
                    "chunk_id": f"{safe_title}_{i}",
                    "text": chunk,
                    "metadata": {
                        "title": doc["title"],
                        "source": doc["source"]
                    }
                }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_path = (BASE_DIR / "../data/processed/documents.jsonl").resolve()
    output_path = (BASE_DIR / "../data/processed/chunks.jsonl").resolve()

    process_documents(input_path, output_path)