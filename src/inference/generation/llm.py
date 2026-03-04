import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.scripts.utils import load_config, setup_logging

# Logging Setup
setup_logging()
logger = logging.getLogger(__name__)

class HFGenerator:
    def __init__(
        self,
        config=None,
        device=None,
    ):

        self.config = config or load_config()
        self.model_name = self.config['models']['generator']['model_name']
        self.max_new_tokens = self.config['models']['generator']['max_new_tokens']
        self.temperature = self.config['models']['generator']['temperature']
        self.do_sample = self.config['models']['generator']['do_sample']
        self.max_length = self.config['models']['generator']['max_length']
        self.truncation = self.config['models']['generator']['truncation']
        self.skip_special_tokens = self.config['models']['generator']['skip_special_tokens']
        self.return_tensors = self.config['models']['generator']['return_tensors']

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading LLM: %s on %s", self.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors=self.return_tensors,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
            )

        answer = self.tokenizer.decode(
            outputs[0], skip_special_tokens=self.skip_special_tokens
        )

        return answer.strip()
