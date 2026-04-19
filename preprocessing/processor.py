"""
Thai NLP preprocessing pipeline.

Steps (as described in the thesis proposal §3.2):
  1. Text cleaning  — HTML, URLs, emoji, special chars, whitespace
  2. Tokenization   — PyThaiNLP newmm (dictionary max-matching)
  3. Stopword removal
  4. NER            — extract Thai province names for the region field
  5. Deduplication  — exact (MD5) + near-duplicate (MinHash + LSH)
"""

import hashlib
import re
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

from crawler.base import RawItem
from database import Article

# ── NER tagger (lazy-loaded once) ───────────────────────────────────────────
_ner = None


def _get_ner():
    global _ner
    if _ner is None and _NER_AVAILABLE:
        # "thainer" is the correct engine name in PyThaiNLP 4/5
        try:
            _ner = _NERClass(engine="thainer")
        except Exception:
            pass   # falls back to province keyword scan
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

# ── Regex patterns ───────────────────────────────────────────────────────────
_RE_HTML     = re.compile(r"<[^>]+>")
_RE_URL      = re.compile(r"https?://\S+|www\.\S+")
_RE_EMOJI    = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)
_RE_SPECIAL  = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s]")
_RE_SPACES   = re.compile(r"\s+")


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Step 1 — remove HTML, URLs, emoji, symbols."""
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMOJI.sub(" ", text)
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_SPACES.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Step 2+3 — Thai tokenization (newmm) then stopword removal."""
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
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
    for prov in THAI_PROVINCES:
        if prov in word or word in prov:
            return prov
    return None


# ── Deduplication state (per-request; reset between searches) ───────────────

class Deduplicator:
    """
    Two-level deduplication:
      1. Exact  — MD5 hash set
      2. Near   — MinHash + LSH (85% similarity threshold)
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.threshold = threshold
        self._hash_set: set[str] = set()
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0

    def is_duplicate(self, text: str) -> bool:
        # Level 1 — exact
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        if md5 in self._hash_set:
            return True
        self._hash_set.add(md5)

        # Level 2 — near-duplicate via MinHash
        mh = _make_minhash(text)
        neighbors = self._lsh.query(mh)
        if neighbors:
            return True

        key = f"doc_{self._counter}"
        self._counter += 1
        self._lsh.insert(key, mh)
        return False


def _make_minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    # Use character 3-grams as the set representation
    for i in range(len(text) - 2):
        mh.update(text[i:i+3].encode("utf-8"))
    return mh


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess(raw_items: list[RawItem], dedup: Deduplicator) -> list[RawItem]:
    """
    Run the full preprocessing pipeline on a list of RawItems.
    Mutates text_content in-place and fills region.
    Returns only the items that passed deduplication.
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
    return processed
