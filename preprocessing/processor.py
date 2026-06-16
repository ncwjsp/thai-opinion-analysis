"""
Subsystem: Thai NLP preprocessing pipeline.
--------------------------------------------------------------------------
Function:   Turn noisy crawled text into clean, deduplicated, region-tagged
            records ready for sentiment classification.
Algorithms & techniques:
  * Tokenization uses PyThaiNLP "newmm" — a dictionary-based maximum-matching
    segmenter, required because Thai is written without spaces between words.
  * Region extraction uses thainer NER for LOC entities, with a curated
    77-province keyword set as a deterministic fallback when NER is absent.
  * Deduplication is two-level: an MD5 hash set removes byte-identical text in
    O(1), and a MinHash + Locality-Sensitive-Hashing index removes near
    duplicates (syndicated/re-headlined stories) above an 85% similarity
    threshold, using character 3-grams as the document fingerprint.
Role in pipeline:
  * Sits between data collection and sentiment analysis. It receives the raw
    RawItem list produced by the crawlers and returns a smaller, cleaned list
    in which every item has non-empty text, is unique within the run, and (when
    detectable) carries a province in its region field.
  * Why character n-grams for MinHash: word tokens are unreliable for Thai
    because segmentation itself can differ between two otherwise-identical
    strings; character 3-grams sidestep that and stay robust to small edits.
  * Why a per-request Deduplicator: dedup state must not leak between searches,
    so the caller constructs a fresh instance for every request.
--------------------------------------------------------------------------

Steps (as described in the thesis proposal §3.2):
  1. Text cleaning  — HTML, URLs, emoji, special chars, whitespace
  2. Tokenization   — PyThaiNLP newmm (dictionary max-matching)
  3. Stopword removal
  4. NER            — extract Thai province names for the region field
  5. Deduplication  — exact (MD5) + near-duplicate (MinHash + LSH)
"""

import logging
from typing import Optional

from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

try:
    from pythainlp.tag import NER as _NERClass
    _NER_AVAILABLE = True
except ImportError:
    _NERClass = None
    _NER_AVAILABLE = False

from datasketch import MinHash, MinHashLSH

from config import MINHASH_NGRAM, MINHASH_NUM_PERM, NER_ENGINE
from crawler.base import RawItem
from database import Article
from utils import (
    char_ngrams,
    collapse_whitespace,
    md5_hash,
    strip_emoji,
    strip_html,
    strip_special,
    strip_urls,
)

logger = logging.getLogger(__name__)

# ── NER tagger (lazy-loaded once) ───────────────────────────────────────────
_ner = None


def _get_ner():
    """Lazily construct and cache the PyThaiNLP NER tagger.

    The tagger is heavy to build, so it is created once on first use and reused
    thereafter. If PyThaiNLP's NER is unavailable or fails to initialise, this
    returns ``None`` and callers fall back to the province keyword scan.

    Returns:
        The cached NER instance, or ``None`` when NER cannot be used.
    """
    global _ner
    if _ner is None and _NER_AVAILABLE:
        # "thainer" is the correct engine name in PyThaiNLP 4/5
        try:
            _ner = _NERClass(engine=NER_ENGINE)
            logger.info("Loaded PyThaiNLP NER engine %r", NER_ENGINE)
        except Exception as exc:
            # falls back to province keyword scan
            logger.warning("NER engine unavailable (%s); using keyword fallback", exc)
    return _ner


# ── Thai province names (used as fallback region extractor) ─────────────────
THAI_PROVINCES = {
    "กรุงเทพ", "กรุงเทพมหานคร", "เชียงใหม่", "เชียงราย", "ภูเก็ต",
    "ขอนแก่น", "อุดรธานี", "นครราชสีมา", "โคราช", "พัทยา", "ชลบุรี",
    "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "อยุธยา", "สุราษฎร์ธานี",
    "หาดใหญ่", "สงขลา", "นครศรีธรรมราช", "ระยอง", "สระบุรี",
    "ลำปาง", "ลำพูน", "แม่ฮ่องสอน", "พะเยา", "น่าน", "แพร่",
    "อุตรดิตถ์", "สุโขทัย", "พิษณุโลก", "เพชรบูรณ์", "กำแพงเพชร",
    "ตาก", "นครสวรรค์", "อุทัยธานี", "ชัยนาท", "สิงห์บุรี", "ลพบุรี",
    "สระแก้ว", "ปราจีนบุรี", "ฉะเชิงเทรา", "นครนายก", "สมุทรสาคร",
    "สมุทรสงคราม", "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม",
    "เพชรบุรี", "ประจวบคีรีขันธ์", "ชุมพร", "ระนอง", "พังงา", "กระบี่",
    "ตรัง", "สตูล", "พัทลุง", "ปัตตานี", "ยะลา", "นราธิวาส",
    "มุกดาหาร", "นครพนม", "สกลนคร", "หนองคาย", "บึงกาฬ", "หนองบัวลำภู",
    "เลย", "กาฬสินธุ์", "มหาสารคาม", "ร้อยเอ็ด", "ยโสธร", "อำนาจเจริญ",
    "อุบลราชธานี", "ศรีสะเกษ", "สุรินทร์", "บุรีรัมย์", "ชัยภูมิ",
}

_STOPWORDS = set(thai_stopwords())


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Step 1 — normalise raw text by stripping non-content noise.

    Applies, in order: HTML tag removal, URL removal, emoji removal, removal of
    characters outside the Thai/Latin/digit ranges, and whitespace collapsing.
    Each step delegates to the shared helpers in :mod:`utils`.

    Args:
        text: Raw text from a crawled item.

    Returns:
        The cleaned, whitespace-normalised string (possibly empty).
    """
    if not text:
        return ""
    text = strip_html(text)
    text = strip_urls(text)
    text = strip_emoji(text)
    text = strip_special(text)
    return collapse_whitespace(text)


def tokenize(text: str) -> list[str]:
    """Step 2+3 — Thai word tokenization (newmm) followed by stopword removal.

    Uses PyThaiNLP's ``newmm`` dictionary-based maximum-matching tokenizer,
    then drops empty tokens and Thai stopwords. Tokenization failures are
    contained: an empty list is returned rather than propagating the error.

    Args:
        text: Cleaned text to tokenize.

    Returns:
        A list of content tokens with stopwords removed.
    """
    try:
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    except Exception as exc:
        logger.warning("Tokenization failed (%s); returning no tokens", exc)
        return []
    return [t for t in tokens if t.strip() and t not in _STOPWORDS]


def extract_region(text: str) -> Optional[str]:
    """
    Step 4 — extract Thai province name via:
      a) PyThaiNLP NER (looks for LOC entities)
      b) Simple keyword match against known province list
    Returns the first match or None.
    """
    # Try NER first
    try:
        ner = _get_ner()
        entities = ner.get_ner(text)          # list of (word, tag)
        for word, tag in entities:
            if "LOC" in tag:
                province = _match_province(word)
                if province:
                    return province
    except Exception:
        pass

    # Fallback: plain keyword scan
    for province in THAI_PROVINCES:
        if province in text:
            return province
    return None


def _match_province(word: str) -> Optional[str]:
    """Match a candidate string against the known province set.

    Matching is bidirectional substring: it succeeds if the candidate contains
    a province name or a province name contains the candidate, which tolerates
    minor boundary noise from the tokenizer or NER.

    Args:
        word: A candidate location string.

    Returns:
        The matched canonical province name, or ``None``.
    """
    for prov in THAI_PROVINCES:
        if prov in word or word in prov:
            return prov
    return None


# ── Deduplication state (per-request; reset between searches) ───────────────

class Deduplicator:
    """Two-level, stateful deduplicator for a single search run.

    Crawled content frequently repeats: the same story is syndicated across
    outlets (exact duplicates) or re-headlined with small edits (near
    duplicates). This collapses both:

      1. Exact  — an MD5 hash set rejects byte-identical text in O(1).
      2. Near   — a MinHash + LSH index rejects text whose estimated Jaccard
                  similarity exceeds ``threshold`` (default 0.85).

    A new instance is created per request so its state never leaks between
    searches.

    Attributes:
        threshold: MinHash similarity cutoff for the near-duplicate stage.
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = MINHASH_NUM_PERM):
        self.threshold = threshold
        self._num_perm = num_perm
        self._hash_set: set[str] = set()
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0

    def is_duplicate(self, text: str) -> bool:
        """Test ``text`` against everything seen so far, recording it if new.

        Args:
            text: Cleaned text to check.

        Returns:
            ``True`` if ``text`` is an exact or near duplicate of an earlier
            item; ``False`` otherwise (in which case it is now indexed).
        """
        # Level 1 — exact
        digest = md5_hash(text)
        if digest in self._hash_set:
            return True
        self._hash_set.add(digest)

        # Level 2 — near-duplicate via MinHash
        mh = _make_minhash(text, self._num_perm)
        neighbors = self._lsh.query(mh)
        if neighbors:
            return True

        key = f"doc_{self._counter}"
        self._counter += 1
        self._lsh.insert(key, mh)
        return False


def _make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Build a MinHash signature for ``text`` from character n-grams.

    Character n-grams (rather than word tokens) are used as the set
    representation because they are robust to Thai's lack of word boundaries
    and to minor edits between near-duplicate headlines.

    Args:
        text: Cleaned text to fingerprint.
        num_perm: Number of MinHash permutations (must match the LSH index).

    Returns:
        A populated :class:`datasketch.MinHash`.
    """
    mh = MinHash(num_perm=num_perm)
    for gram in char_ngrams(text, MINHASH_NGRAM):
        mh.update(gram)
    return mh


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess(raw_items: list[RawItem], dedup: Deduplicator) -> list[RawItem]:
    """Run the full preprocessing pipeline over a batch of crawled items.

    For each item: clean the text, drop it if empty or a (near-)duplicate,
    otherwise write the cleaned text back in place and fill ``region`` via NER
    when it is not already set. The input items are mutated in place.

    Args:
        raw_items: Items straight from the crawlers.
        dedup: A per-request :class:`Deduplicator` holding dedup state.

    Returns:
        The subset of items that survived cleaning and deduplication.
    """
    processed = []
    for item in raw_items:
        cleaned = clean_text(item.text_content)
        if not cleaned:
            continue
        if dedup.is_duplicate(cleaned):
            continue
        item.text_content = cleaned
        if item.region is None:
            item.region = extract_region(cleaned)
        processed.append(item)
    logger.info(
        "Preprocess: %d in -> %d out (%d dropped)",
        len(raw_items), len(processed), len(raw_items) - len(processed),
    )
    return processed
