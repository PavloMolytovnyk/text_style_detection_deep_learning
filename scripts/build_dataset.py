"""Compile dataset/style/tone/*.txt into the three-column input CSV; no spaCy required."""

import argparse
from pathlib import Path

import pandas as pd

from textstyle.settings import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw" / "rebuilt_dataset.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.source.glob("*/*/*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            rows.append({"text": text, "style": path.parent.parent.name, "tone": path.parent.name})
    if not rows:
        raise ValueError("No non-empty text files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} source texts to {args.output}")


if __name__ == "__main__":
    main()
