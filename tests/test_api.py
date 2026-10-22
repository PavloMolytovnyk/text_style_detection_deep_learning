import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from textstyle.settings import DEFAULT_ARTIFACTS


@pytest.fixture()
def client():
    with TestClient(create_app(DEFAULT_ARTIFACTS / "cnn" / "model")) as value:
        yield value


def test_live_ready_and_model_info(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["model"] == "cnn"
    assert client.get("/model/info").json()["parameter_count"] > 0


def test_real_inference_and_single_target_endpoints(client):
    text = "Dear colleagues, please confirm the meeting schedule today."
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    for target in ["style", "tone"]:
        assert sum(result[f"{target}_scores"].values()) == pytest.approx(1.0, abs=1e-6)
        separate = client.post(f"/predict/{target}", json={"text": text}).json()
        assert separate[target] == result[target]
        assert separate["confidence"] == pytest.approx(result[f"{target}_confidence"], abs=1e-6)


@pytest.mark.parametrize("text", ["", "  ", "!!!", "x" * 8001])
def test_bad_input_returns_422(client, text):
    assert client.post("/predict", json={"text": text}).status_code == 422


def test_batch_and_batch_limit(client):
    response = client.post("/predict/batch", json={"texts": ["Hello there!", "We require an immediate response."]})
    assert response.status_code == 200 and len(response.json()["results"]) == 2
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["Hello"] * 33}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["Hello", ""]}).status_code == 422


def test_missing_checkpoint_returns_503(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"text": "Hello world"}).status_code == 503
