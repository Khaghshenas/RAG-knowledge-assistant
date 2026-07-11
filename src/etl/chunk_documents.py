# TODO: Check chunk size and overlap later
# NOTE: Might need to deduplicate chunks if many repeats

import json
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from src.etl.clean_text import clean_text
from src.scripts.utils import load_config



def chunk_text(text, config, tokenizer):
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + config['preprocessing']['chunk_size']
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += config['preprocessing']['chunk_size'] - config['preprocessing']['chunk_overlap']

    return chunks

def process_documents(config=None):
    config = config or load_config()
    input_path = Path(config['paths']['documents_path'])
    output_path = Path(config['paths']['chunks_path'])

    with input_path.open("r", encoding="utf-8") as f, output_path.open("w", encoding="utf-8") as out:
        
        tokenizer_name = config['preprocessing']['tokenizer_name']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        for line in tqdm(f, desc="Processing documents"):
            doc = json.loads(line)
            text = clean_text(doc["text"])
            chunks = chunk_text(text, config, tokenizer)

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
    
    return output_path 

if __name__ == "__main__":
    process_documents()