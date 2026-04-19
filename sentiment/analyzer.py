"""
Sentiment analysis module.

Supported models:
  - "xlm-roberta"   → cardiffnlp/twitter-xlm-roberta-base-sentiment
                       Multilingual (100 languages incl. Thai), 3 classes.
                       Labels: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive

  - "wangchanberta" → phoner45/wangchan-sentiment-thai-text-model
                       WangchanBERTa fine-tuned on Thai sentiment dataset.
                       ~94.57% accuracy on Thai text (Nokkaew et al. 2023).
                       Labels: pos=positive, neu=neutral, neg=negative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "neutral", "negative"]

MODEL_IDS: dict[str, str] = {
    "xlm-roberta":   "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "wangchanberta": "phoner45/wangchan-sentiment-thai-text-model",
}

# Unified label map for both models → canonical positive/neutral/negative
_LABEL_MAP: dict[str, SentimentLabel] = {
    # cardiffnlp XLM-RoBERTa outputs
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # phoner45 WangchanBERTa outputs
    "pos": "positive",
    "neu": "neutral",
    "neg": "negative",
    # Pass-through (already canonical)
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
}


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float   # model confidence 0.0 – 1.0


class SentimentAnalyzer:
    """
    Lazy-loading wrapper around a HuggingFace pipeline.

    Usage:
        analyzer = SentimentAnalyzer()
        result   = analyzer.predict("ฉันรักประเทศไทย")
        print(result.label, result.score)
    """

    def __init__(self, model_name: str = "xlm-roberta"):
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unknown model '{model_name}'. Choose: {list(MODEL_IDS)}")
        self._model_name = model_name
        self._model_id   = MODEL_IDS[model_name]
        self._pipeline: Pipeline | None = None

    def _load(self) -> Pipeline:
        if self._pipeline is None:
            logger.info("Loading sentiment model: %s", self._model_id)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_id,
                tokenizer=self._model_id,
                truncation=True,
                max_length=416,   # WangchanBERTa max is 416 tokens
            )
            logger.info("Model loaded.")
        return self._pipeline

    def predict(self, text: str) -> SentimentResult:
        pipe = self._load()
        output = pipe(text)[0]
        return SentimentResult(
            label=_LABEL_MAP.get(output["label"], "neutral"),
            score=float(output["score"]),
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        pipe = self._load()
        outputs = pipe(texts, batch_size=16)
        return [
            SentimentResult(
                label=_LABEL_MAP.get(out["label"], "neutral"),
                score=float(out["score"]),
            )
            for out in outputs
        ]


# Module-level singleton — one model loaded per process
_analyzer: SentimentAnalyzer | None = None


def get_analyzer(model_name: str = "xlm-roberta") -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None or _analyzer._model_name != model_name:
        _analyzer = SentimentAnalyzer(model_name)
    return _analyzer
