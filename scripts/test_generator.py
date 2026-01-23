import sys
import os

# Add Generator root to PYTHONPATH
GENERATOR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../generator/"))
sys.path.insert(0, GENERATOR_PATH)

from llm import HFGenerator
from prompt import build_prompt

fake_chunks = [
    {"text": "The Super Bowl XLIV was won by the New Orleans Saints."},
    {"text": "The game was played in 2010 in Miami."}
]

question = "Who won the Super Bowl in 2010?"

prompt = build_prompt(question, fake_chunks)

generator = HFGenerator()
answer = generator.generate(prompt)

print("Answer:", answer)
