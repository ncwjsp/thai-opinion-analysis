"""
Subsystem: Shared type definitions (enumerations).
--------------------------------------------------------------------------
Function:   Provide a single authoritative set of enumerated constants for the
            values that recur across the system — sentiment labels, source
            names, model names, and result-sort orders — so that string
            literals are not duplicated and mistyped across modules.
Algorithms & techniques:
  * Each enumeration subclasses ``(str, Enum)``, so members behave as plain
    strings (identical JSON output and database storage) while still giving
    static checkers and editors a closed set of valid values.
  * ``normalize_label`` centralises the mapping from the various raw labels the
    two transformer models emit (``LABEL_0``, ``pos``, ``neg``, …) to the three
    canonical labels used everywhere else.
Role in pipeline:
  * Imported by the sentiment, schema, and API layers; it has no dependencies
    of its own, so it sits at the bottom of the import graph.
--------------------------------------------------------------------------
"""

from enum import Enum


class SentimentLabel(str, Enum):
    """The three canonical sentiment classes used throughout the system."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SourceName(str, Enum):
    """Identifiers for the data sources the crawlers can target."""
    GOOGLE_NEWS = "google_news"
    PANTIP = "pantip"


class ModelName(str, Enum):
    """Selectable sentiment models, mapped to checkpoints in the analyzer."""
    XLM_ROBERTA = "xlm-roberta"
    WANGCHANBERTA = "wangchanberta"


class SortOrder(str, Enum):
    """Ordering options for returned article lists."""
    NEWEST = "newest"
    OLDEST = "oldest"
    CONFIDENCE = "confidence"


# Maps every raw label either transformer can emit to a canonical class.
_RAW_TO_LABEL: dict[str, SentimentLabel] = {
    "LABEL_0": SentimentLabel.NEGATIVE,
    "LABEL_1": SentimentLabel.NEUTRAL,
    "LABEL_2": SentimentLabel.POSITIVE,
    "neg": SentimentLabel.NEGATIVE,
    "neu": SentimentLabel.NEUTRAL,
    "pos": SentimentLabel.POSITIVE,
    "negative": SentimentLabel.NEGATIVE,
    "neutral": SentimentLabel.NEUTRAL,
    "positive": SentimentLabel.POSITIVE,
}


def normalize_label(raw: str) -> str:
    """Map a model-specific raw label to a canonical label string.

    Args:
        raw: The label string as produced by a transformer model.

    Returns:
        One of ``"positive"`` / ``"neutral"`` / ``"negative"``; unknown labels
        fall back to ``"neutral"``.

    Example:
        >>> normalize_label("LABEL_2")
        'positive'
        >>> normalize_label("neg")
        'negative'
        >>> normalize_label("???")
        'neutral'
    """
    return _RAW_TO_LABEL.get(raw, SentimentLabel.NEUTRAL).value
