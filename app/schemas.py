from pydantic import BaseModel, Field, field_validator

from textstyle.predict import validate_text
from textstyle.settings import MAX_CHARS


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, examples=["Dear team, please submit the report before noon."])

    @field_validator("text")
    @classmethod
    def validate_content(cls, value):
        return validate_text(value)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)

    @field_validator("texts")
    @classmethod
    def validate_contents(cls, values):
        return [validate_text(value) for value in values]


class Prediction(BaseModel):
    style: str
    tone: str
    style_confidence: float
    tone_confidence: float
    style_scores: dict[str, float]
    tone_scores: dict[str, float]
    model: str
    token_count: int
    truncated: bool
    unknown_ratio: float
    latency_ms: float
    warnings: list[str]


class StyleResponse(BaseModel):
    style: str
    confidence: float
    scores: dict[str, float]
    model: str
    warnings: list[str]


class ToneResponse(BaseModel):
    tone: str
    confidence: float
    scores: dict[str, float]
    model: str
    warnings: list[str]


class BatchResponse(BaseModel):
    results: list[Prediction]
