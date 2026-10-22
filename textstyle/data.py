"""Data audit and one shared, reproducible, group-aware split for all models."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from textstyle.settings import DEFAULT_DATA, ROOT, SEED, STYLES, TONES


def text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_corpus(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"text", "style", "tone"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    frame = frame[["text", "style", "tone"]].copy()
    raw_rows = len(frame)
    frame["text"] = frame["text"].astype(str).str.strip()
    for target in ["style", "tone"]:
        frame[target] = frame[target].astype(str).str.strip().str.lower()
    aliases_fixed = int((frame["style"] == "literaly").sum())
    frame["style"] = frame["style"].replace({"literaly": "literary"})
    empty = frame["text"].str.len().eq(0)
    frame = frame.loc[~empty].copy()
    for target, allowed in [("style", STYLES), ("tone", TONES)]:
        invalid = set(frame[target]) - set(allowed)
        if invalid:
            raise ValueError(f"Unknown {target} labels: {sorted(invalid)}")
        if set(frame[target]) != set(allowed):
            raise ValueError(f"All five {target} classes are required for this project")
    frame["_key"] = frame["text"].map(text_key)
    conflicts = frame.groupby("_key")[["style", "tone"]].nunique().max(axis=1)
    conflict_keys = set(conflicts[conflicts > 1].index)
    excluded = frame.loc[frame["_key"].isin(conflict_keys), ["text", "style", "tone"]]
    # A conflicting annotation has no defensible automatic winner: exclude every copy.
    frame = frame.loc[~frame["_key"].isin(conflict_keys)].copy()
    before = len(frame)
    frame = frame.drop_duplicates("_key").reset_index(drop=True)
    frame["sample_id"] = frame["_key"].map(lambda t: hashlib.sha256(t.encode()).hexdigest()[:16])
    audit = {
        "source_sha256": fingerprint(path),
        "source_rows": raw_rows,
        "empty_removed": int(empty.sum()),
        "duplicates_removed": before - len(frame),
        "conflicting_groups_removed": len(conflict_keys),
        "conflicting_rows_removed": len(excluded),
        "excluded_conflicts": excluded.to_dict(orient="records"),
        "label_aliases_fixed": aliases_fixed,
        "clean_rows": len(frame),
    }
    return frame.drop(columns="_key"), audit


def similarity_groups(texts: list[str], threshold: float = 0.82) -> list[int]:
    """Conservatively group near-copies using word trigrams, before splitting.

    Also group a nearly contained passage when it is at least half as long.
    This is an unsupervised leakage safeguard, not a classifier.
    """
    parent = list(range(len(texts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    grams = []
    for text in texts:
        words = re.findall(r"\w+", text.casefold())
        grams.append({tuple(words[i : i + 3]) for i in range(len(words) - 2)})
    for i, a in enumerate(grams):
        if len(a) < 8:
            continue
        for j in range(i):
            b = grams[j]
            shorter, longer = min(len(a), len(b)), max(len(a), len(b))
            if shorter < 8 or shorter / longer < 0.5:
                continue
            intersection = len(a & b)
            union = len(a) + len(b) - intersection
            if intersection / union >= threshold or intersection / shorter >= 0.95:
                ra, rb = root(i), root(j)
                parent[max(ra, rb)] = min(ra, rb)
    return [root(i) for i in range(len(texts))]


def verify_splits(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    for i, name in enumerate(names):
        if frames[name].empty:
            raise ValueError(f"Empty {name} split")
        for other in names[:i]:
            for column in ["sample_id", "group_id"]:
                if set(frames[name][column]) & set(frames[other][column]):
                    raise ValueError(f"Leakage between {name} and {other}: {column}")
            if set(frames[name].text.map(text_key)) & set(frames[other].text.map(text_key)):
                raise ValueError(f"Duplicate text crosses {name}/{other}")


def prepare(source: Path, destination: Path = DEFAULT_DATA, seed: int = SEED) -> dict:
    frame, audit = read_corpus(source)
    frame["group_id"] = similarity_groups(frame.text.tolist())
    strata = frame["style"] + "__" + frame["tone"]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(frame, strata, frame.group_id))
    trainval = frame.iloc[trainval_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_idx, val_idx = next(inner.split(trainval, trainval["style"] + "__" + trainval["tone"], trainval.group_id))
    frames = {
        "train": trainval.iloc[train_idx].reset_index(drop=True),
        "validation": trainval.iloc[val_idx].reset_index(drop=True),
        "test": frame.iloc[test_idx].reset_index(drop=True),
    }
    verify_splits(frames)
    destination.mkdir(parents=True, exist_ok=True)
    if audit["excluded_conflicts"]:
        pd.DataFrame(audit["excluded_conflicts"]).to_csv(destination / "excluded_conflicts.csv", index=False)
    frame.to_csv(destination / "dataset.csv", index=False)
    audit.update(
        {
            "seed": seed,
            "near_duplicate_groups": int(frame.group_id.nunique()),
            "grouping": {"word_trigram_jaccard": 0.82, "containment": 0.95, "min_length_ratio": 0.5},
            "strategy": "Nested five-fold StratifiedGroupKFold, joint style/tone strata, first folds",
            "approximate_ratios": {"train": 0.64, "validation": 0.16, "test": 0.20},
            "splits": {},
        }
    )
    for name, part in frames.items():
        path = destination / f"{name}.csv"
        part.to_csv(path, index=False)
        audit["splits"][name] = {
            "rows": len(part),
            "sha256": fingerprint(path),
            "style": dict(Counter(part["style"])),
            "tone": dict(Counter(part["tone"])),
        }
    (destination / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def load_splits(directory: Path = DEFAULT_DATA) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name, info in manifest["splits"].items():
        path = directory / f"{name}.csv"
        if fingerprint(path) != info["sha256"]:
            raise ValueError(f"{path} changed. Re-run data preparation before training")
        frames[name] = pd.read_csv(path, keep_default_na=False)
    verify_splits(frames)
    return frames, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw" / "dataset.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
