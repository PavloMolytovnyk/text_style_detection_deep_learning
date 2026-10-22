import pytest
import torch
from torch.nn import functional as F

from textstyle.models import build_model
from textstyle.predict import Predictor
from textstyle.tokenization import Collator, Vocabulary


@pytest.mark.parametrize("architecture", ["cnn", "bilstm"])
def test_both_heads_and_embedding_receive_gradients(architecture):
    torch.manual_seed(17)
    model = build_model(architecture, {"vocab_size": 20, "embedding_dim": 16, "dropout": 0.0})
    batch = Collator()([([2, 3], 1, 3), ([4, 5, 6, 7, 8, 9], 2, 4)])
    style, tone = model(batch["input_ids"], batch["lengths"])
    assert style.shape == tone.shape == (2, 5)
    assert torch.isfinite(style).all() and torch.isfinite(tone).all()
    loss = F.cross_entropy(style, batch["style_labels"]) + F.cross_entropy(tone, batch["tone_labels"])
    loss.backward()
    for parameter in [model.embedding.weight, model.style_head.weight, model.tone_head.weight]:
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_real_checkpoint_predictions_are_batch_padding_invariant(cnn_predictor):
    short = "Thanks, friend!"
    long = "The quarterly committee will review the detailed financial report at its next scheduled meeting."
    individual = cnn_predictor.predict(short)
    batch = cnn_predictor.predict_batch([short, long])[0]
    assert individual["style"] == batch["style"]
    assert individual["tone"] == batch["tone"]
    for target in ["style", "tone"]:
        assert individual[f"{target}_scores"] == pytest.approx(batch[f"{target}_scores"], abs=1e-6)


def test_actual_model_reload_is_portable(cnn_predictor):
    reloaded = Predictor(cnn_predictor.model_dir)
    original = cnn_predictor.predict("Please reply before the deadline.")
    result = reloaded.predict("Please reply before the deadline.")
    assert original["tone_scores"] == result["tone_scores"]
    assert result["tone"] in reloaded.metadata["labels"]["tone"]


def test_vocabulary_round_trip(tmp_path):
    vocabulary = Vocabulary.fit(["Hello, WORLD!"])
    vocabulary.save(tmp_path)
    assert Vocabulary.load(tmp_path).encode("Hello, WORLD!") == vocabulary.encode("Hello, WORLD!")


def test_truncation_is_disclosed(cnn_predictor):
    result = cnn_predictor.predict("word " * 300)
    assert result["truncated"]
    assert result["warnings"]


def test_transformer_path_is_trainable_without_network_downloads():
    model = build_model(
        "transformer",
        {
            "encoder_config": {
                "model_type": "distilbert",
                "vocab_size": 30,
                "dim": 16,
                "hidden_dim": 32,
                "n_heads": 2,
                "n_layers": 1,
                "max_position_embeddings": 64,
            },
            "dropout": 0.0,
        },
    )
    batch = Collator()([([2, 3, 4], 0, 1), ([5, 6, 7, 8], 2, 3)])
    style, tone = model(batch["input_ids"], batch["lengths"])
    loss = F.cross_entropy(style, batch["style_labels"]) + F.cross_entropy(tone, batch["tone_labels"])
    loss.backward()
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
