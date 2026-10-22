import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from textstyle.settings import STYLES, TONES

TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(text):
        if len(token) > 1 and token.isupper() and any(c.isalpha() for c in token):
            tokens.append("<upper>")
        tokens.append(token.casefold().replace("’", "'"))
    return tokens


class Vocabulary:
    pad_id = 0
    unk_id = 1
    kind = "vocabulary"

    def __init__(self, tokens: list[str], max_length: int = 256):
        self.tokens = tokens
        self.lookup = {token: i for i, token in enumerate(tokens)}
        self.max_length = max_length

    @classmethod
    def fit(cls, texts, max_size=12000, min_frequency=1, max_length=256):
        counts = Counter(token for text in texts for token in tokenize(text))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        tokens = ["<pad>", "<unk>"]
        tokens.extend(t for t in ordered if counts[t] >= min_frequency and t not in tokens)
        return cls(tokens[:max_size], max_length)

    def encode(self, text):
        return [self.lookup.get(t, self.unk_id) for t in tokenize(text)[: self.max_length]] or [self.unk_id]

    def inspect(self, text):
        all_tokens = tokenize(text)
        selected = all_tokens[: self.max_length]
        return {
            "token_count": len(all_tokens),
            "truncated": len(all_tokens) > self.max_length,
            "unknown_ratio": sum(t not in self.lookup for t in selected) / max(1, len(selected)),
        }

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "vocabulary.json").write_text(
            json.dumps(
                {
                    "tokens": self.tokens,
                    "max_length": self.max_length,
                    "kind": self.kind,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path):
        info = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
        return cls(info["tokens"], info["max_length"])


class TransformerTokenizer:
    kind = "transformer"

    def __init__(self, source: str | Path, max_length=256, local_only=False):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(source),
            local_files_only=local_only,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Choose an encoder tokenizer with a pad token")
        self.pad_id = self.tokenizer.pad_token_id
        self.max_length = max_length

    def encode(self, text):
        return self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]

    def inspect(self, text):
        ids = self.tokenizer(text, truncation=False)["input_ids"]
        selected = ids[: self.max_length]
        return {
            "token_count": len(ids),
            "truncated": len(ids) > self.max_length,
            "unknown_ratio": selected.count(self.tokenizer.unk_token_id) / max(1, len(selected)),
        }

    def save(self, folder: Path):
        self.tokenizer.save_pretrained(folder)


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.rows = [
            (tokenizer.encode(row.text), STYLES.index(row.style), TONES.index(row.tone)) for row in frame.itertuples()
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, pad_id=0, min_length=5):
        self.pad_id = pad_id
        self.min_length = min_length

    def __call__(self, rows):
        lengths = torch.tensor([len(row[0]) for row in rows], dtype=torch.long)
        size = max(self.min_length, int(lengths.max()))
        ids = torch.full((len(rows), size), self.pad_id, dtype=torch.long)
        for i, (tokens, _, _) in enumerate(rows):
            ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": ids,
            "lengths": lengths,
            "style_labels": torch.tensor([r[1] for r in rows], dtype=torch.long),
            "tone_labels": torch.tensor([r[2] for r in rows], dtype=torch.long),
        }
