"""Train neural models, select on validation only, then evaluate on held-out test data."""

import argparse
import copy
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from textstyle.data import fingerprint, load_splits
from textstyle.evaluation import calibrate, collect_logits, metrics_from_logits, plot_history, write_evaluation
from textstyle.models import build_model
from textstyle.settings import DEFAULT_ARTIFACTS, DEFAULT_DATA, SEED, STYLES, TONES
from textstyle.tokenization import Collator, TextDataset, TransformerTokenizer, Vocabulary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(frame, tokenizer, batch_size, shuffle=False, seed=SEED):
    return DataLoader(
        TextDataset(frame, tokenizer),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collator(tokenizer.pad_id),
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def train_one(
    architecture,
    frames,
    manifest,
    output: Path,
    *,
    epochs=35,
    patience=7,
    batch_size=32,
    seed=SEED,
    device="cpu",
    max_length=256,
    pretrained="distilbert/distilbert-base-cased",
    learning_rate=None,
):
    set_seed(seed)
    run_dir = output / architecture
    if (run_dir / "model" / "metadata.json").exists():
        raise FileExistsError(f"{run_dir} already contains a model. Use a new --output directory")
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if architecture == "transformer":
        from transformers import AutoConfig

        tokenizer = TransformerTokenizer(pretrained, max_length=max_length)
        config = {
            "encoder_config": AutoConfig.from_pretrained(pretrained, trust_remote_code=False).to_dict(),
            "dropout": 0.3,
        }
        learning_rate = learning_rate or 2e-5
    else:
        tokenizer = Vocabulary.fit(frames["train"].text, max_length=max_length)
        config = {"vocab_size": len(tokenizer.tokens), "embedding_dim": 128, "dropout": 0.4, "pad_id": tokenizer.pad_id}
        if architecture == "cnn":
            config.update({"channels": 96, "kernels": [3, 4, 5]})
        else:
            config["hidden_size"] = 96
        learning_rate = learning_rate or 0.001
    tokenizer.save(model_dir / "tokenizer")
    model = build_model(architecture, config, pretrained if architecture == "transformer" else None).to(device)
    train_loader = make_loader(frames["train"], tokenizer, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(frames["validation"], tokenizer, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            style, tone = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            loss = F.cross_entropy(style, batch["style_labels"].to(device)) + F.cross_entropy(
                tone, batch["tone_labels"].to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch["lengths"])
        validation = collect_logits(model, val_loader, device)
        metrics = metrics_from_logits(validation)
        val_loss = sum(
            float(F.cross_entropy(validation[f"{target}_logits"], validation[f"{target}_labels"]))
            for target in ["style", "tone"]
        )
        score = metrics["mean_macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(frames["train"]),
                "validation_loss": val_loss,
                "style_f1": metrics["style"]["f1_macro"],
                "tone_f1": metrics["tone"]["f1_macro"],
                "mean_macro_f1": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"{architecture:11} epoch {epoch:02d}/{epochs} loss={history[-1]['train_loss']:.4f} "
            f"val_style_f1={metrics['style']['f1_macro']:.4f} val_tone_f1={metrics['tone']['f1_macro']:.4f}",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step(score)
        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    validation = collect_logits(model, val_loader, device)
    temperatures = {
        target: calibrate(validation[f"{target}_logits"], validation[f"{target}_labels"])
        for target in ["style", "tone"]
    }
    validation_metrics = metrics_from_logits(validation, temperatures)
    torch.save(best_state, model_dir / "weights.pt")
    metadata = {
        "schema_version": 1,
        "architecture": architecture,
        "model_config": config,
        "tokenizer_kind": tokenizer.kind,
        "max_length": max_length,
        "labels": {"style": STYLES, "tone": TONES},
        "temperatures": temperatures,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": training_seconds,
        "training_device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "loss": "CrossEntropy(style) + CrossEntropy(tone)",
        "selection_metric": "mean validation macro F1",
        "validation": validation_metrics,
        "data_manifest": copy.deepcopy(manifest),
        "torch_version": str(torch.__version__),
        "weights_sha256": fingerprint(model_dir / "weights.pt"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pretrained_source": pretrained if architecture == "transformer" else None,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, run_dir / "learning_curves.png")
    return model, tokenizer, metadata


def train_experiment(
    architectures=("cnn", "bilstm"),
    data_dir=DEFAULT_DATA,
    output=DEFAULT_ARTIFACTS,
    epochs=35,
    patience=7,
    batch_size=32,
    seed=SEED,
    device="cpu",
    max_length=256,
    pretrained="distilbert/distilbert-base-cased",
    learning_rate=None,
):
    torch.set_num_threads(2)
    frames, manifest = load_splits(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    trained = {}
    for architecture in architectures:
        trained[architecture] = train_one(
            architecture,
            frames,
            manifest,
            output,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
            device=device,
            max_length=max_length,
            pretrained=pretrained,
            learning_rate=learning_rate,
        )
    winner = max(trained, key=lambda name: trained[name][2]["validation"]["mean_macro_f1"])
    summary = {
        "selected_model": winner,
        "selection_basis": "Validation mean macro F1 only",
        "data": manifest,
        "models": {},
    }
    previous_path = output / "experiment.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous["data"] != manifest:
            raise ValueError("Existing experiment uses different data; choose a new output directory")
        summary["models"].update(previous["models"])
    # Only after model selection is fixed do we read held-out test results.
    for name, (model, tokenizer, metadata) in trained.items():
        test_loader = make_loader(frames["test"], tokenizer, batch_size)
        collected = collect_logits(model, test_loader, device)
        test_report = write_evaluation(collected, metadata["temperatures"], frames["test"], output / name / "test")
        summary["models"][name] = {
            "validation_mean_macro_f1": metadata["validation"]["mean_macro_f1"],
            "test": test_report,
            "best_epoch": metadata["best_epoch"],
            "parameter_count": metadata["parameter_count"],
            "training_seconds": metadata["training_seconds"],
        }
    winner = max(summary["models"], key=lambda name: summary["models"][name]["validation_mean_macro_f1"])
    summary["selected_model"] = winner
    (output / "experiment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "active_model.json").write_text(json.dumps({"model_dir": f"{winner}/model"}, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": winner, "test": summary["models"][winner]["test"]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures", nargs="+", choices=["cnn", "bilstm", "transformer"], default=["cnn", "bilstm"]
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--pretrained", default="distilbert/distilbert-base-cased")
    args = vars(parser.parse_args())
    args["device"] = (
        "cuda"
        if args["device"] == "auto" and torch.cuda.is_available()
        else ("cpu" if args["device"] == "auto" else args["device"])
    )
    if any(args[key] < 1 for key in ["epochs", "patience", "batch_size", "max_length"]):
        parser.error("Epochs, patience, batch size and maximum length must be positive")
    if args["max_length"] < 5:
        parser.error("Maximum length must be at least 5 tokens")
    train_experiment(**args)


if __name__ == "__main__":
    main()
