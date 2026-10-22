import pytest

from textstyle.predict import Predictor
from textstyle.settings import DEFAULT_ARTIFACTS


@pytest.fixture(scope="session")
def cnn_predictor():
    return Predictor(DEFAULT_ARTIFACTS / "cnn" / "model")
