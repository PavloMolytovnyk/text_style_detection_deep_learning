"""Re-evaluate an existing neural checkpoint without fitting anything."""

import argparse
from pathlib import Path

from textstyle.data import load_splits
from textstyle.evaluation import collect_logits, write_evaluation
from textstyle.predict import Predictor
from textstyle.settings import DEFAULT_DATA, ROOT
from textstyle.train import make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "reevaluation")
    args = parser.parse_args()
    frames, manifest = load_splits(args.data_dir)
    predictor = Predictor(args.model_dir)
    if manifest != predictor.metadata["data_manifest"]:
        raise ValueError("The checkpoint and prepared test data come from different experiments")
    loader = make_loader(frames["test"], predictor.tokenizer, 32)
    result = collect_logits(predictor.model, loader, predictor.device)
    print(write_evaluation(result, predictor.metadata["temperatures"], frames["test"], args.output))


if __name__ == "__main__":
    main()
