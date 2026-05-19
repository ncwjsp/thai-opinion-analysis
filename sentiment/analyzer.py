"""
Sentiment analysis module.

Supported models:
  - "xlm-roberta"   → cardiffnlp/twitter-xlm-roberta-base-sentiment
  - "wangchanberta" → phoner45/wangchan-sentiment-thai-text-model

Inference strategy (auto-selected):
  1. HuggingFace Inference API  — if HF_API_TOKEN env var is set (no local RAM needed)
  2. Local pipeline              — fallback (requires ~1GB RAM)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "neutral", "negative"]

MODEL_IDS: dict[str, str] = {
    "xlm-roberta":   "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "wangchanberta": "phoner45/wangchan-sentiment-thai-text-model",
}

_LABEL_MAP: dict[str, SentimentLabel] = {
    "LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive",
    "pos": "positive", "neu": "neutral", "neg": "negative",
    "positive": "positive", "neutral": "neutral", "negative": "negative",
}


@dataclass
class SentimentResult:
    label: SentimentLabel
    score: float


class SentimentAnalyzer:
    def __init__(self, model_name: str = "xlm-roberta"):
        if model_name not in MODEL_IDS:
            raise ValueError(f"Unknown model '{model_name}'. Choose: {list(MODEL_IDS)}")
        self._model_name = model_name
        self._model_id   = MODEL_IDS[model_name]
        self._pipeline   = None
        self._hf_token   = os.environ.get("HF_API_TOKEN", "").strip()
        self._use_api    = bool(self._hf_token)
        if self._use_api:
            logger.info("Sentiment: using HuggingFace Inference API for %s", self._model_id)
        else:
            logger.info("Sentiment: using local pipeline for %s", self._model_id)

    # ── HuggingFace Inference API ────────────────────────────────────────────

    def _api_predict(self, texts: list[str]) -> list[SentimentResult]:
        import requests as _req
        url = f"https://api-inference.huggingface.co/models/{self._model_id}"
        headers = {"Authorization": f"Bearer {self._hf_token}"}
        results = []
        # API accepts up to 100 inputs per call
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            for attempt in range(3):
                resp = _req.post(url, headers=headers, json={"inputs": batch}, timeout=30)
                if resp.status_code == 503:
                    # Model is loading on HF side — wait and retry
                    wait = resp.json().get("estimated_time", 20)
                    logger.info("HF model loading, waiting %.0fs...", wait)
                    time.sleep(min(float(wait), 30))
                    continue
                resp.raise_for_status()
                break
            raw = resp.json()
            # Response shape: [[{label, score}, ...], ...]  or [{label,score},...]
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                # multi-input: list of lists — take highest-score item per input
                for item_scores in raw:
                    best = max(item_scores, key=lambda x: x["score"])
                    results.append(SentimentResult(
                        label=_LABEL_MAP.get(best["label"], "neutral"),
                        score=float(best["score"]),
                    ))
            else:
                best = max(raw, key=lambda x: x["score"])
                results.append(SentimentResult(
                    label=_LABEL_MAP.get(best["label"], "neutral"),
                    score=float(best["score"]),
                ))
        return results

    # ── Local pipeline ───────────────────────────────────────────────────────

    def _load_local(self):
        if self._pipeline is None:
            from transformers import pipeline
            logger.info("Loading local model: %s", self._model_id)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_id,
                tokenizer=self._model_id,
                truncation=True,
                max_length=416,
            )
            logger.info("Local model loaded.")
        return self._pipeline

    # ── Public interface ─────────────────────────────────────────────────────

    def predict(self, text: str) -> SentimentResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        if self._use_api:
            return self._api_predict(texts)
        pipe = self._load_local()
        outputs = pipe(texts, batch_size=16)
        return [
            SentimentResult(
                label=_LABEL_MAP.get(out["label"], "neutral"),
                score=float(out["score"]),
            )
            for out in outputs
        ]


_analyzer: SentimentAnalyzer | None = None


def get_analyzer(model_name: str = "xlm-roberta") -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None or _analyzer._model_name != model_name:
        _analyzer = SentimentAnalyzer(model_name)
    return _analyzer
