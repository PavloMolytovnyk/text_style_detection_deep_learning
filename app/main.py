import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.schemas import BatchRequest, BatchResponse, Prediction, StyleResponse, TextRequest, ToneResponse
from textstyle.predict import Predictor

logger = logging.getLogger(__name__)


def create_app(model_dir: Path | None = None):
    @asynccontextmanager
    async def lifespan(application):
        application.state.predictor = None
        try:
            application.state.predictor = Predictor(model_dir)
        except (OSError, ValueError, RuntimeError) as error:
            logger.exception("Model could not be loaded: %s", error)
        yield
        application.state.predictor = None

    application = FastAPI(
        title="Text Style & Tone Detection DL",
        version="1.0.0",
        lifespan=lifespan,
        description="English text classification with a trained PyTorch neural network. Scores are model estimates.",
    )

    def predictor(request: Request):
        value = getattr(request.app.state, "predictor", None)
        if value is None:
            raise HTTPException(503, "Trained model is unavailable. Check MODEL_DIR or run python -m textstyle.train")
        return value

    @application.get("/", tags=["Service"])
    def index():
        return {"service": "Text Style & Tone Detection DL", "docs": "/docs", "ui": "Streamlit on port 8501"}

    @application.get("/health/live", tags=["Service"])
    def live():
        return {"status": "alive"}

    @application.get("/health/ready", tags=["Service"])
    def ready(request: Request):
        value = predictor(request)
        return {"status": "ready", "model": value.metadata["architecture"]}

    @application.get("/model/info", tags=["Service"])
    def model_info(request: Request):
        metadata = predictor(request).metadata
        return {
            key: metadata[key]
            for key in [
                "architecture",
                "labels",
                "max_length",
                "parameter_count",
                "best_epoch",
                "temperatures",
                "validation",
                "created_at_utc",
            ]
        }

    @application.post("/predict", response_model=Prediction, tags=["Prediction"])
    def predict_both(body: TextRequest, request: Request):
        return predictor(request).predict(body.text)

    @application.post("/predict/style", response_model=StyleResponse, tags=["Prediction"])
    def predict_style(body: TextRequest, request: Request):
        result = predictor(request).predict(body.text)
        return {
            "style": result["style"],
            "confidence": result["style_confidence"],
            "scores": result["style_scores"],
            "model": result["model"],
            "warnings": result["warnings"],
        }

    @application.post("/predict/tone", response_model=ToneResponse, tags=["Prediction"])
    def predict_tone(body: TextRequest, request: Request):
        result = predictor(request).predict(body.text)
        return {
            "tone": result["tone"],
            "confidence": result["tone_confidence"],
            "scores": result["tone_scores"],
            "model": result["model"],
            "warnings": result["warnings"],
        }

    @application.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
    def predict_batch(body: BatchRequest, request: Request):
        return {"results": predictor(request).predict_batch(body.texts)}

    return application


app = create_app()
