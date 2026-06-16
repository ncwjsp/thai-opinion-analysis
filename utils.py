"""
Subsystem: Shared utility helpers.
--------------------------------------------------------------------------
Function:   Collect small, pure, reusable helpers — text normalisation,
            hashing, and character-n-gram generation — in one place so the
            crawler, preprocessing, and persistence layers do not each carry
            their own copy of the same logic.
Algorithms & techniques:
  * The text helpers wrap pre-compiled regular expressions; compiling once at
    import time keeps repeated cleaning cheap.
  * ``md5_hash`` is the single hashing function used both for exact
    deduplication and for the database's unique content hash, guaranteeing the
    two always agree.
  * ``char_ngrams`` yields the overlapping character n-grams used as the
    document fingerprint for MinHash-based near-duplicate detection.
Role in pipeline:
  * A leaf module with no project dependencies; imported wherever the same
    primitive operation is needed.
--------------------------------------------------------------------------
"""

import hashlib
import re
from typing import Iterator

# Pre-compiled patterns, compiled once at import time.
_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMOJI = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)
_RE_SPECIAL = re.compile("[^฀-๿a-zA-Z0-9\\s]")
_RE_SPACES = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML/XML tags from ``text``.

    Args:
        text: Input string possibly containing markup.

    Returns:
        ``text`` with every ``<...>`` tag replaced by a space.

    Example:
        >>> strip_html("<b>hi</b>")
        ' hi '
    """
    return _RE_HTML.sub(" ", text)


def strip_urls(text: str) -> str:
    """Remove ``http(s)://`` and ``www.`` URLs from ``text``.

    Args:
        text: Input string.

    Returns:
        ``text`` with URLs replaced by a space.
    """
    return _RE_URL.sub(" ", text)


def strip_emoji(text: str) -> str:
    """Remove emoji and pictographic symbols from ``text``.

    Args:
        text: Input string.

    Returns:
        ``text`` with emoji code points removed.
    """
    return _RE_EMOJI.sub(" ", text)


def strip_special(text: str) -> str:
    """Remove characters outside the Thai, Latin, digit, and space ranges.

    Args:
        text: Input string.

    Returns:
        ``text`` reduced to Thai/Latin letters, digits, and spaces.
    """
    return _RE_SPECIAL.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Collapse any run of whitespace to a single space and trim the ends.

    Args:
        text: Input string.

    Returns:
        The whitespace-normalised, stripped string.

    Example:
        >>> collapse_whitespace("a   b\\n c ")
        'a b c'
    """
    return _RE_SPACES.sub(" ", text).strip()


def md5_hash(text: str) -> str:
    """Return the 32-character hexadecimal MD5 digest of ``text``.

    Used both for exact-duplicate detection and for the database's unique
    ``content_hash`` column, so both rely on identical hashing.

    Args:
        text: Input string to hash.

    Returns:
        The hex MD5 digest.

    Example:
        >>> md5_hash("a")
        '0cc175b9c0f1b6a831c399e269772661'
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int) -> Iterator[bytes]:
    """Yield the overlapping character n-grams of ``text`` as UTF-8 bytes.

    Args:
        text: Source text.
        n: N-gram size.

    Yields:
        Each length-``n`` character window, encoded as bytes (the form the
        MinHash implementation consumes).

    Example:
        For ``text="abcd"`` and ``n=3`` this yields ``b"abc"`` then ``b"bcd"``.
        Strings shorter than ``n`` yield nothing.
    """
    for i in range(len(text) - n + 1):
        yield text[i:i + n].encode("utf-8")
