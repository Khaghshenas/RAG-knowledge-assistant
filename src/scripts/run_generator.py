import logging

from src.inference.generation.llm import HFGenerator
from src.inference.generation.prompt import build_prompt
from src.scripts.utils import setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

fake_chunks = [
    {"text": "The Super Bowl XLIV was won by the New Orleans Saints."},
    {"text": "The game was played in 2010 in Miami."}
]

question = "Who won the Super Bowl in 2010?"

prompt = build_prompt(question, fake_chunks)

generator = HFGenerator()
answer = generator.generate(prompt)

logger.info("Answer: %s", answer)