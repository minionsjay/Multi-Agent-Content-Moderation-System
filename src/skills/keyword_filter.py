import json
import os
import re
import logging

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS = {
    "toxic": [
        "傻逼", "SB", "傻比", "脑残", "弱智", "白痴",
        "废物", "垃圾", "狗屁", "去死", "滚蛋", "草泥马",
        "操你妈", "fuck", "shit", "damn", "idiot", "stupid",
    ],
    "politics": [
        "台独", "港独", "藏独", "疆独",
    ],
    "violence": [
        "杀了你", "弄死你", "砍死", "炸了", "枪支", "炸弹",
    ],
    "porn": [
        "裸照", "约炮", "一夜情", "援交",
    ],
    "gambling": [
        "赌博", "赌场", "博彩", "下注", "盘口",
    ],
}

# Known false positive phrases → these are innocent texts that happen
# to contain keyword substrings. When matched, they auto-escalate to BERT.
WHITELIST_PHRASES = [
    "接口交换",    # contains "口交"  → technical term
    "性交朋友",    # contains "性交"  → actually "make friends"
    "操场上",      # contains "操"    → playground
    "操练",        # contains "操"    → drill/practice
    "人体艺术摄影",  # contains sensitive chars → photography
    "赌博式",      # contains "赌博"  → metaphorical usage
]


class KeywordFilter:
    def __init__(self, dict_path: str | None = None):
        self.keywords: dict[str, list[str]] = {}
        self._automaton = None
        self._jieba_loaded = False
        self._jieba = None
        self._whitelist_re = None
        self._load(dict_path)
        self._build_automaton()
        self._build_whitelist()

    def _load(self, dict_path: str | None):
        if dict_path and os.path.exists(dict_path):
            with open(dict_path) as f:
                self.keywords = json.load(f)
        else:
            self.keywords = DEFAULT_KEYWORDS
            logger.info("Using default keyword dictionary (%d categories)", len(self.keywords))

    def _build_automaton(self):
        try:
            import ahocorasick
            self._automaton = ahocorasick.Automaton()
            for category, words in self.keywords.items():
                for word in words:
                    self._automaton.add_word(word, (word, category))
            self._automaton.make_automaton()
        except ImportError:
            self._automaton = None
            logger.warning("pyahocorasick not installed, falling back to substring matching")

    def _build_whitelist(self):
        """Build regex pattern for whitelist phrases."""
        if WHITELIST_PHRASES:
            escaped = [re.escape(p) for p in WHITELIST_PHRASES]
            self._whitelist_re = re.compile("|".join(escaped))

    def _ensure_jieba(self):
        if self._jieba_loaded:
            return
        try:
            import jieba
            self._jieba = jieba
            self._jieba_loaded = True
        except ImportError:
            self._jieba_loaded = True  # Don't retry
            logger.warning("jieba not installed — keyword context validation disabled")

    def match(self, text: str) -> dict:
        """Match text against keywords with context-aware confidence scoring.

        Confidence levels:
          1.0  — standalone keyword (e.g., "傻逼" as a word) → direct block
          0.6  — embedded inside longer word (e.g., "口交" in "接口交换") → escalate to BERT
          0.0  — no match
        """
        if not text:
            return {"label": None, "confidence": 0.0, "matches": []}

        text_lower = text.lower()
        raw_matches: list[tuple[str, str]] = []

        if self._automaton:
            for end_idx, (word, category) in self._automaton.iter(text_lower):
                raw_matches.append((word, category))
        else:
            for category, words in self.keywords.items():
                for word in words:
                    if word.lower() in text_lower:
                        raw_matches.append((word, category))

        if not raw_matches:
            return {"label": None, "confidence": 0.0, "matches": []}

        # Check whitelist — only suppress keyword occurrences that fall INSIDE whitelist spans
        if self._whitelist_re and self._whitelist_re.search(text_lower):
            whitelist_hits = [(m.start(), m.end(), m.group())
                             for m in self._whitelist_re.finditer(text_lower)]
            filtered = []
            for word, cat in raw_matches:
                all_positions = self._find_all_positions(text_lower, word.lower())
                # A keyword is "real" if ANY occurrence is outside all whitelist spans
                has_real = False
                has_whitelisted = False
                for pos in all_positions:
                    in_whitelist = any(wh_start <= pos and pos + len(word) <= wh_end
                                      for wh_start, wh_end, _ in whitelist_hits)
                    if in_whitelist:
                        has_whitelisted = True
                    else:
                        has_real = True
                if has_real:
                    filtered.append((word, cat, "real"))
                elif has_whitelisted:
                    filtered.append((word, cat, "whitelist"))
            if not any(f[2] == "real" for f in filtered):
                # All matches are whitelisted → pass
                return {"label": None, "confidence": 0.0, "matches": [],
                       "whitelist_hit": True, "suppressed_matches": [{"word": f[0], "category": f[1]} for f in filtered]}
            raw_matches = [(w, c) for w, c, t in filtered if t == "real"]

        # Context-aware validation with jieba
        self._ensure_jieba()
        has_embedded = False
        has_standalone = False
        validated_matches = []

        for word, category in raw_matches:
            context = self._validate_context(text, word)
            validated_matches.append({
                "word": word,
                "category": category,
                "context": context,
            })
            if context == "standalone":
                has_standalone = True
            else:
                has_embedded = True

        # Pick most severe category
        severity_order = ["politics", "violence", "porn", "gambling", "toxic"]
        matched_categories = set(m["category"] for m in validated_matches)

        label = None
        for cat in severity_order:
            if cat in matched_categories:
                label = cat
                break
        if label is None and matched_categories:
            label = "toxic"

        # Confidence: 1.0 if ANY match is standalone, 0.6 if ALL are embedded
        if has_standalone:
            confidence = 1.0
        elif has_embedded:
            confidence = 0.6
        else:
            confidence = 0.0

        return {
            "label": label,
            "confidence": confidence,
            "matches": validated_matches,
        }

    @staticmethod
    def _find_all_positions(text: str, word: str) -> list[int]:
        """Find all start positions of word in text."""
        positions = []
        pos = 0
        while True:
            pos = text.find(word, pos)
            if pos < 0:
                break
            positions.append(pos)
            pos += 1
        return positions

    def _validate_context(self, text: str, word: str) -> str:
        """Check whether keyword is standalone or embedded.

        If keyword appears multiple times, return 'standalone' if ANY occurrence is standalone.
        """
        positions = self._find_all_positions(text, word)
        if not positions:
            return "embedded"

        # If any occurrence is standalone, the keyword is real
        for pos in positions:
            if self._check_position_standalone(text, word, pos):
                return "standalone"
        return "embedded"

    def _check_position_standalone(self, text: str, word: str, pos: int) -> bool:
        """Check whether keyword at given position is a standalone word."""
        kw_start, kw_end = pos, pos + len(word)

        # Method 1: jieba segmentation
        if self._jieba is not None:
            try:
                tokens = list(self._jieba.cut(text))
                token_spans = []
                idx = 0
                for token in tokens:
                    token_spans.append((idx, idx + len(token), token))
                    idx += len(token)

                # Exact single-token match
                for start, end, token in token_spans:
                    if start == kw_start and end == kw_end:
                        return True

                # Keyword fully inside a larger token → embedded
                for start, end, token in token_spans:
                    if start < kw_start and end > kw_end:
                        return False
                    if start < kw_start < end:  # starts inside a token
                        return False
                    if start < kw_end < end:    # ends inside a token
                        return False

                # Spans multiple tokens, starts + ends at boundaries → standalone
                starts_ok = any(start == kw_start for start, _, _ in token_spans)
                ends_ok = any(end == kw_end for _, end, _ in token_spans)
                return starts_ok or ends_ok
            except Exception:
                pass

        # Method 2: Heuristic — keyword surrounded by CJK chars → embedded
        before_char = text[pos - 1] if pos > 0 else ""
        after_char = text[pos + len(word)] if pos + len(word) < len(text) else ""
        if before_char and self._is_cjk(before_char):
            return False
        if after_char and self._is_cjk(after_char):
            return False
        return True

    def _is_cjk(self, char: str) -> bool:
        cp = ord(char)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF)


# Singleton
from src.config import KEYWORD_DICT
keyword_filter = KeywordFilter(KEYWORD_DICT)
