"""
Subsystem: Reusable input validators.
--------------------------------------------------------------------------
Function:   Hold the validation logic used by the Pydantic request schemas as
            standalone, independently testable functions, rather than burying
            it inside the schema class.
Algorithms & techniques:
  * Each validator is a pure function that returns the cleaned value or raises
    ``ValueError`` with a clear message; Pydantic surfaces that message as an
    HTTP 422 response automatically.
Role in pipeline:
  * Imported by ``api/schemas.py``; keeping the rules here means the same check
    can be reused by other entry points (e.g. the schema's field validators)
    without duplication.
--------------------------------------------------------------------------
"""

from typing import Iterable


def clean_keyword(value: str) -> str:
    """Trim a keyword and reject it if empty after trimming.

    Args:
        value: Raw keyword string from the request.

    Returns:
        The trimmed keyword.

    Raises:
        ValueError: If the keyword is empty or only whitespace.

    Example:
        ``clean_keyword("  cat  ")`` returns ``"cat"``; a blank or
        whitespace-only input raises ``ValueError``.
    """
    value = value.strip()
    if not value:
        raise ValueError("keyword must not be empty or whitespace")
    return value


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """Remove duplicates from ``values`` while preserving first-seen order.

    Args:
        values: The incoming sequence (e.g. selected sources).

    Returns:
        A de-duplicated list in original order.

    Raises:
        ValueError: If the resulting list is empty.

    Example:
        ``dedupe_preserving_order(["a", "b", "a"])`` returns ``["a", "b"]``.
    """
    seen: set[str] = set()
    result = [v for v in values if not (v in seen or seen.add(v))]
    if not result:
        raise ValueError("at least one value must be provided")
    return result


def ensure_ratio(value: float, name: str = "value") -> float:
    """Validate that ``value`` lies within the inclusive range [0, 1].

    Args:
        value: The number to check.
        name: Field name used in the error message.

    Returns:
        ``value`` unchanged when valid.

    Raises:
        ValueError: If ``value`` is outside [0, 1].
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value
