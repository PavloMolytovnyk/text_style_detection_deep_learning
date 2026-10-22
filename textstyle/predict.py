import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch

from textstyle.data import fingerprint
from textstyle.models import build_model
from textstyle.settings import DEFAULT_ARTIFACTS, MAX_CHARS
from textstyle.tokenization import Collator, TransformerTokenizer, Vocabulary


def resolve_model_dir(artifact_root: Path = DEFAULT_ARTIFACTS) -> Path:
    override = os.getenv("MODEL_DIR")
    if override:
        return Path(override).expanduser().resolve()
    registry = artifact_root / "active_model.json"
    if not registry.exists():
        raise FileNotFoundError("Trained model is missing. Run: python -m textstyle.train --output artifacts")
    return (artifact_root / json.loads(registry.read_text(encoding="utf-8"))["model_dir"]).resolve()


def validate_text(text: str) -> str:
    text = text.strip()
    if not text or not any(character.isalpha() for character in text):
        raise ValueError("Enter text containing at least one word")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text must contain at most {MAX_CHARS} characters")
    return text


class Predictor:
    def __init__(self, model_dir: Path | None = None, device="cpu"):
        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))
        self.model_dir = Path(model_dir) if model_dir is not None else resolve_model_dir()
        self.device = torch.device(device)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported model artifact schema")
        if fingerprint(self.model_dir / "weights.pt") != self.metadata["weights_sha256"]:
            raise ValueError("Model weights checksum does not match metadata")
        if self.metadata["tokenizer_kind"] == "vocabulary":
            self.tokenizer = Vocabulary.load(self.model_dir / "tokenizer")
        else:
            self.tokenizer = TransformerTokenizer(
                self.model_dir / "tokenizer",
                max_length=self.metadata["max_length"],
                local_only=True,
            )
        self.model = build_model(self.metadata["architecture"], self.metadata["model_config"])
        self.model.load_state_dict(torch.load(self.model_dir / "weights.pt", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.collator = Collator(self.tokenizer.pad_id)
        self._lock = threading.Lock()

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if not texts or len(texts) > 32:
            raise ValueError("A batch must contain between 1 and 32 texts")
        texts = [validate_text(text) for text in texts]
        start = time.perf_counter()
        rows = [(self.tokenizer.encode(text), 0, 0) for text in texts]
        batch = self.collator(rows)
        with self._lock, torch.inference_mode():
            logits = self.model(batch["input_ids"].to(self.device), batch["lengths"].to(self.device))
            probabilities = [
                (values / self.metadata["temperatures"][target]).softmax(-1).cpu().tolist()
                for target, values in zip(["style", "tone"], logits, strict=True)
            ]
        elapsed_ms = round((time.perf_counter() - start) * 1000 / len(texts), 3)
        results = []
        for i, text in enumerate(texts):
            info = self.tokenizer.inspect(text)
            warnings = []
            if info["truncated"]:
                warnings.append(f"Only the first {self.metadata['max_length']} tokens were analyzed.")
            if info["unknown_ratio"] > 0.35:
                warnings.append("Many words are outside the training vocabulary; interpret the result cautiously.")
            letters = [c for c in text if c.isalpha()]
            if sum(c.isascii() for c in letters) / max(1, len(letters)) < 0.7:
                warnings.append("This model was trained on English text.")
            result = {
                "model": self.metadata["architecture"],
                "warnings": warnings,
                "token_count": info["token_count"],
                "truncated": info["truncated"],
                "unknown_ratio": round(info["unknown_ratio"], 4),
                "latency_ms": elapsed_ms,
            }
            for target, head_probabilities in zip(["style", "tone"], probabilities, strict=True):
                names = self.metadata["labels"][target]
                scores = {name: float(value) for name, value in zip(names, head_probabilities[i], strict=True)}
                label = max(scores, key=scores.get)
                result[target] = label
                result[f"{target}_confidence"] = scores[label]
                result[f"{target}_scores"] = scores
            if min(result["style_confidence"], result["tone_confidence"]) < 0.5:
                warnings.append("The model is uncertain about at least one prediction.")
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict English style and tone with the packaged neural model")
    parser.add_argument("text")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(Predictor(args.model_dir).predict(args.text), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
