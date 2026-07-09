from src.inference.generation.llm import HFGenerator
from src.inference.generation.prompt import build_prompt

def test_generator_returns_answer(monkeypatch):
    fake_chunks = [
        {"text": "The Super Bowl XLIV was won by the New Orleans Saints."},
        {"text": "The game was played in 2010 in Miami."}
    ]

    question = "Who won the Super Bowl in 2010?"

    prompt = build_prompt(question, fake_chunks)

    def fake_generate(self, prompt):
        return "The New Orleans Saints"

    monkeypatch.setattr(HFGenerator, "generate", fake_generate)

    generator = HFGenerator()
    answer = generator.generate(prompt)

    assert isinstance(answer, str)
    assert answer == "The New Orleans Saints"