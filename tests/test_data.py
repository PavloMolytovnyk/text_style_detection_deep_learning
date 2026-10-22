import json

from textstyle.data import load_splits, read_corpus, similarity_groups, text_key
from textstyle.settings import DEFAULT_DATA, ROOT
from textstyle.tokenization import Vocabulary, tokenize


def test_original_corpus_conflicts_are_excluded():
    frame, audit = read_corpus(ROOT / "data" / "raw" / "dataset.csv")
    assert audit["source_rows"] == 1000
    assert audit["conflicting_rows_removed"] == 2
    assert len(frame) == 998
    assert "literaly" not in set(frame["style"])
    assert frame.text.map(text_key).is_unique


def test_splits_are_reproducible_and_disjoint():
    frames, manifest = load_splits()
    assert sum(len(frame) for frame in frames.values()) == 998
    assert manifest == json.loads((DEFAULT_DATA / "manifest.json").read_text())
    assert not set(frames["train"].group_id) & set(frames["test"].group_id)
    assert not set(frames["validation"].group_id) & set(frames["test"].group_id)


def test_near_duplicate_passages_stay_together():
    paragraph = "The research committee reviewed the evidence and confirmed that the findings support our previous analysis of the problem."
    groups = similarity_groups([paragraph, paragraph + " Additional work is planned.", "Hello friend!"])
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_vocabulary_uses_training_texts_only():
    vocabulary = Vocabulary.fit(["Training phrase!"])
    assert vocabulary.encode("validationonly") == [1]
    assert "!" in vocabulary.lookup
    assert "<upper>" in tokenize("URGENT reply!")
