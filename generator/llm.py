import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class HFGenerator:
    def __init__(
        self,
        model_name="google/flan-t5-large",
        max_new_tokens=128,
        device=None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading LLM: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        answer = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        return answer.strip()
