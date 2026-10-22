import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.nn import functional as F

from textstyle.settings import STYLES, TONES


@torch.inference_mode()
def collect_logits(model, loader, device):
    model.eval()
    result = {key: [] for key in ["style_logits", "tone_logits", "style_labels", "tone_labels"]}
    for batch in loader:
        style, tone = model(batch["input_ids"].to(device), batch["lengths"].to(device))
        result["style_logits"].append(style.cpu())
        result["tone_logits"].append(tone.cpu())
        for key in ["style_labels", "tone_labels"]:
            result[key].append(batch[key].cpu())
    return {key: torch.cat(values) for key, values in result.items()}


def calibrate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a scalar temperature on VALIDATION logits only."""
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=60, line_search_fn="strong_wolfe")
    logits = logits.detach().clone()
    labels = labels.detach().clone()

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.25, 8.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.25, 8.0).item())


def expected_calibration_error(labels, probabilities, bins=10):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    result = 0.0
    for i in range(bins):
        mask = (confidence > i / bins) & (confidence <= (i + 1) / bins)
        if mask.any():
            result += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def metrics_from_logits(collected, temperatures=None):
    temperatures = temperatures or {"style": 1.0, "tone": 1.0}
    report = {}
    joint = []
    for target, names in [("style", STYLES), ("tone", TONES)]:
        labels = collected[f"{target}_labels"].numpy()
        probabilities = (collected[f"{target}_logits"] / temperatures[target]).softmax(dim=-1).numpy()
        predictions = probabilities.argmax(axis=1)
        joint.append(labels == predictions)
        auc = None
        if len(set(labels.tolist())) == len(names):
            auc = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
        report[target] = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "f1_macro": float(f1_score(labels, predictions, labels=list(range(5)), average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
            "roc_auc_ovr_macro": auc,
            "negative_log_likelihood": float(
                F.cross_entropy(
                    collected[f"{target}_logits"] / temperatures[target],
                    collected[f"{target}_labels"],
                )
            ),
            "ece_10_bins": expected_calibration_error(labels, probabilities),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(5))).tolist(),
            "classification_report": classification_report(
                labels,
                predictions,
                labels=list(range(5)),
                target_names=names,
                output_dict=True,
                zero_division=0,
            ),
        }
    report["mean_macro_f1"] = (report["style"]["f1_macro"] + report["tone"]["f1_macro"]) / 2
    report["joint_accuracy"] = float(np.logical_and(*joint).mean())
    return report


def write_evaluation(collected, temperatures, frame, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    report = metrics_from_logits(collected, temperatures)
    (destination / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = frame[["sample_id", "text", "style", "tone"]].copy()
    for target, names in [("style", STYLES), ("tone", TONES)]:
        probabilities = (collected[f"{target}_logits"] / temperatures[target]).softmax(-1).numpy()
        predictions[f"predicted_{target}"] = [names[i] for i in probabilities.argmax(1)]
        predictions[f"{target}_score"] = probabilities.max(1)
        matrix = np.array(report[target]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set(
            xticks=range(5),
            yticks=range(5),
            xticklabels=names,
            yticklabels=names,
            xlabel="Predicted class",
            ylabel="True class",
            title=f"{target.title()} on held-out test data",
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "#172B4D"
            )
        fig.savefig(destination / f"{target}_confusion.png", dpi=160)
        plt.close(fig)
        pd.DataFrame(report[target]["classification_report"]).T.to_csv(destination / f"{target}_report.csv")
    predictions.to_csv(destination / "predictions.csv", index=False)
    errors = predictions[
        (predictions["style"] != predictions["predicted_style"])
        | (predictions["tone"] != predictions["predicted_tone"])
    ]
    errors.to_csv(destination / "errors.csv", index=False)
    return report


def plot_history(history, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    frame = pd.DataFrame(history)
    axes[0].plot(frame.epoch, frame.train_loss, label="Training")
    axes[0].plot(frame.epoch, frame.validation_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Combined cross-entropy", title="Training progress")
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.style_f1, label="Style")
    axes[1].plot(frame.epoch, frame.tone_f1, label="Tone")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro F1", ylim=(0, 1), title="Validation quality")
    axes[1].legend()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
