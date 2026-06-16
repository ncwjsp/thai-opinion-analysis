"""
Subsystem: Sentiment analysis.
--------------------------------------------------------------------------
Function:   Classify each cleaned text as positive / neutral / negative with a
            confidence score, using transformer language models.
Algorithms & techniques:
  * Two pre-trained transformers are offered at runtime: XLM-RoBERTa (a
    100-language multilingual model) and WangchanBERTa (pre-trained purely on
    Thai text, higher accuracy on Thai-specific expressions).
  * The provider-specific raw labels are normalised through a single mapping
    table so the rest of the system only sees positive/neutral/negative.
  * Inference is batched (size 16 locally, up to 100 per remote call) to
    amortise model-call overhead, and the model is lazy-loaded once and cached
    as a process-wide singleton to avoid repeated multi-second start-ups.
Role in pipeline:
  * Consumes the cleaned texts emitted by preprocessing and returns one
    SentimentResult (label + confidence) per text, preserving order so callers
    can zip results back onto their source items.
  * Confidence is always surfaced, not hidden, so the API can show users that
    each label is a probabilistic judgement rather than ground truth.
  * Backend selection is transparent to callers: whether the prediction came
    from the remote API or a local pipeline, the return shape is identical.
--------------------------------------------------------------------------

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

from config import HF_API_BATCH, MODEL_MAX_LENGTH, SENTIMENT_BATCH_SIZE
from enums import ModelName, normalize_label
from exceptions import ModelError, ModelLoadError

logger = logging.getLogger(__name__)

MODEL_IDS: dict[str, str] = {
    ModelName.XLM_ROBERTA.value:   "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    ModelName.WANGCHANBERTA.value: "phoner45/wangchan-sentiment-thai-text-model",
}


@dataclass
class SentimentResult:
    """A single sentiment prediction.

    Attributes:
        label: One of ``"positive"``, ``"neutral"``, ``"negative"``.
        score: Model confidence for ``label``, in the range ``[0, 1]``.
    """
    label: str
    score: float


class SentimentAnalyzer:
    """Wrapper over a HuggingFace sentiment model with two inference backends.

    Inference is auto-selected at construction time: if the ``HF_API_TOKEN``
    environment variable is set, predictions are made through the HuggingFace
    Inference API (no local RAM needed); otherwise a local ``transformers``
    pipeline is lazily loaded on first use. Both backends return the same
    :class:`SentimentResult` shape, so callers are agnostic to the choice.
    """

    def __init__(self, model_name: str = "xlm-roberta"):
        """Initialise the analyzer for a named model.

        Args:
            model_name: Key into :data:`MODEL_IDS` (``"xlm-roberta"`` or
                ``"wangchanberta"``).

        Raises:
            ValueError: If ``model_name`` is not a recognised model.
        """
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
        """Predict sentiment via the HuggingFace Inference API.

        Inputs are sent in batches of :data:`HF_API_BATCH`. A ``503`` response
        means the model is still loading server-side, so the call waits for the
        suggested time and retries.

        Args:
            texts: Texts to classify.

        Returns:
            One :class:`SentimentResult` per input, in order.

        Raises:
            ModelError: If a batch cannot be classified after its retries.
        """
        import requests as _req
        url = f"https://router.huggingface.co/hf-inference/models/{self._model_id}"
        headers = {"Authorization": f"Bearer {self._hf_token}"}
        results = []
        # API accepts up to HF_API_BATCH inputs per call
        for i in range(0, len(texts), HF_API_BATCH):
            batch = texts[i:i + HF_API_BATCH]
            resp = None
            for attempt in range(3):
                try:
                    resp = _req.post(url, headers=headers, json={"inputs": batch}, timeout=30)
                except (_req.Timeout, _req.ConnectionError) as exc:
                    logger.warning("HF API attempt %d failed: %s", attempt + 1, exc)
                    if attempt == 2:
                        raise ModelError("HuggingFace API unreachable after retries") from exc
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 503:
                    # Model is loading on HF side — wait and retry
                    wait = resp.json().get("estimated_time", 20)
                    logger.info("HF model loading, waiting %.0fs...", wait)
                    time.sleep(min(float(wait), 30))
                    continue
                resp.raise_for_status()
                break
            if resp is None or not resp.ok:
                raise ModelError("HuggingFace API failed to classify a batch")
            raw = resp.json()
            # Response shape: [[{label, score}, ...], ...]  or [{label,score},...]
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                # multi-input: list of lists — take highest-score item per input
                for item_scores in raw:
                    best = max(item_scores, key=lambda x: x["score"])
                    results.append(SentimentResult(
                        label=normalize_label(best["label"]),
                        score=float(best["score"]),
                    ))
            else:
                best = max(raw, key=lambda x: x["score"])
                results.append(SentimentResult(
                    label=normalize_label(best["label"]),
                    score=float(best["score"]),
                ))
        return results

    # ── Local pipeline ───────────────────────────────────────────────────────

    def _load_local(self):
        """Lazily build and cache the local ``transformers`` pipeline.

        The model (~1GB) is loaded only on first use and reused thereafter.

        Returns:
            The cached sentiment-analysis pipeline.

        Raises:
            ModelLoadError: If the model cannot be loaded (e.g. missing weights
                or insufficient memory).
        """
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info("Loading local model: %s", self._model_id)
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self._model_id,
                    tokenizer=self._model_id,
                    truncation=True,
                    max_length=MODEL_MAX_LENGTH,
                )
                logger.info("Local model loaded.")
            except Exception as exc:
                logger.error("Failed to load local model %s: %s", self._model_id, exc)
                raise ModelLoadError(f"Could not load sentiment model {self._model_id}") from exc
        return self._pipeline

    # ── Public interface ─────────────────────────────────────────────────────

    def predict(self, text: str) -> SentimentResult:
        """Classify a single text.

        Args:
            text: The text to classify.

        Returns:
            The :class:`SentimentResult` for ``text``.
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Classify many texts at once.

        Routes to the HuggingFace API or the local pipeline depending on how
        the analyzer was configured. The local path batches inference at
        :data:`SENTIMENT_BATCH_SIZE`.

        Args:
            texts: Texts to classify.

        Returns:
            One :class:`SentimentResult` per input, in order. Empty if
            ``texts`` is empty.
        """
        if not texts:
            return []
        if self._use_api:
            return self._api_predict(texts)
        pipe = self._load_local()
        outputs = pipe(texts, batch_size=SENTIMENT_BATCH_SIZE)
        return [
            SentimentResult(
                label=normalize_label(out["label"]),
                score=float(out["score"]),
            )
            for out in outputs
        ]


_analyzer: SentimentAnalyzer | None = None


def get_analyzer(model_name: str = "xlm-roberta") -> SentimentAnalyzer:
    """Return a process-wide :class:`SentimentAnalyzer`, reusing when possible.

    A single analyzer is cached and shared so the (expensive) model is not
    reconstructed on every request. It is rebuilt only when a different
    ``model_name`` is requested than the one currently cached.

    Args:
        model_name: Which model the caller wants.

    Returns:
        A ready-to-use analyzer for ``model_name``.
    """
    global _analyzer
    if _analyzer is None or _analyzer._model_name != model_name:
        _analyzer = SentimentAnalyzer(model_name)
    return _analyzer
