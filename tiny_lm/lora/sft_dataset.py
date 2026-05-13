import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class SFTJsonlDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, context_length: int, split: str = "train", max_samples: int | None = None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.split = split
        self.max_samples = max_samples

        if os.path.isdir(self.data_path):
            if self.split == "train":
                self.data_file = os.path.join(self.data_path, "sft_data.jsonl")
            elif self.split == "valid":
                self.data_file = os.path.join(self.data_path, "sft_data_test.jsonl")
            else:
                raise ValueError(f"Unsupported split: {self.split}")
        else:
            self.data_file = self.data_path

        if not os.path.isfile(self.data_file):
            raise ValueError(f"Data file not found: {self.data_file}")

        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.user_token_id = self._get_special_token_id("<|user|>")
        self.assistant_token_id = self._get_special_token_id("<|assistant|>")

        self.samples: List[Tuple[str, str]] = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                question = str(item.get("question", ""))
                answer = str(item.get("answer", ""))
                if not question or not answer:
                    continue

                # Lazy tokenize in __getitem__ to avoid upfront tokenization cost.
                self.samples.append((question, answer))
                if self.max_samples is not None and len(self.samples) >= self.max_samples:
                    break

        if not self.samples:
            raise ValueError(f"No valid samples loaded from {self.data_file}")

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _get_special_token_id(self, token: str) -> int:
        token_ids = self._encode_text(token)
        assert len(token_ids) == 1, f"{token} should map to exactly one token id"
        return token_ids[0]

    def __len__(self):
        return len(self.samples)

    def _build_xy(self, question: str, answer: str) -> Tuple[List[int], List[int]]:
        question_ids = self._encode_text(question)
        answer_ids = self._encode_text(answer)
        token_ids = [self.user_token_id] + question_ids + [self.assistant_token_id] + answer_ids + [self.eos_token_id]

        # data: all tokens before eos
        x_ids = token_ids[:-1]

        # label: next-token targets with prompt/question region masked.
        y_ids = token_ids[1:]
        # Mask from index 0 to the position where question-last-token predicts <|assistant|> (inclusive).
        # token layout: [<|user|>] + question + [<|assistant|>] + answer + [<eos>]
        # masked positions: x[0] ... x[len(question_ids)]
        mask_end_inclusive = len(question_ids)
        y_ids[:mask_end_inclusive + 1] = [-100] * (mask_end_inclusive + 1)

        if len(x_ids) > self.context_length:
            x_ids = x_ids[:self.context_length]
            y_ids = y_ids[:self.context_length]
        else:
            pad_len = self.context_length - len(x_ids)
            x_ids = x_ids + [self.pad_token_id] * pad_len
            y_ids = y_ids + [-100] * pad_len

        return x_ids, y_ids

    def __getitem__(self, idx):
        question, answer = self.samples[idx]
        x_ids, y_ids = self._build_xy(question, answer)
        return torch.LongTensor(x_ids), torch.LongTensor(y_ids)
