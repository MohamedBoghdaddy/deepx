"""
Multilingual text preprocessing utilities for Arabic ABSA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised in runtime environments with the dependency installed
    import emoji
except ImportError:  # pragma: no cover - graceful fallback before requirements are installed
    emoji = None

try:  # pragma: no cover - exercised in runtime environments with the dependency installed
    from lingua import LanguageDetectorBuilder
except ImportError:  # pragma: no cover - graceful fallback before requirements are installed
    LanguageDetectorBuilder = None

from transformers import AutoTokenizer

from franco import (
    DEFAULT_FRANCO_SEED_PATH,
    GENERIC_ENGLISH_FRANCO_ENTRIES,
    load_franco_map,
)


PRIMARY_ABSA_MODEL_NAME = "xlm-roberta-base"
AUXILIARY_SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
DEFAULT_FRANCO_THRESHOLD = 0.30

EMOJI_SENTIMENT = {
    ":smiling_face_with_heart-eyes:": "EMO_POS",
    ":red_heart:": "EMO_POS",
    ":thumbs_up:": "EMO_POS",
    ":enraged_face:": "EMO_NEG",
    ":thumbs_down:": "EMO_NEG",
    ":neutral_face:": "EMO_NEU",
}

EMOJI_NAME_TO_TOKEN = {
    alias.strip(":").lower(): token
    for alias, token in EMOJI_SENTIMENT.items()
}

FALLBACK_EMOJI_ALIAS_BY_CHAR = {
    "😍": "smiling_face_with_heart-eyes",
    "❤️": "red_heart",
    "❤": "red_heart",
    "👍": "thumbs_up",
    "😡": "enraged_face",
    "👎": "thumbs_down",
    "😐": "neutral_face",
    "😂": "face_with_tears_of_joy",
    "🤣": "rolling_on_the_floor_laughing",
    "🙃": "upside_down_face",
}


@dataclass
class PreprocessResult:
    """Structured preprocessing result for downstream explainability and rules."""

    original_text: str
    cleaned_text: str
    normalized_text: str
    emoji_tokens: List[str]
    emoji_names: List[str]
    language: str
    franco_token_ratio: float


@lru_cache(maxsize=1)
def _get_language_detector() -> Optional[object]:
    """Build the lingua detector lazily when the dependency is available."""
    if LanguageDetectorBuilder is None:
        return None

    builder_factory = getattr(LanguageDetectorBuilder, "from_all_spoken_languages", None)
    if builder_factory is None:
        builder_factory = getattr(LanguageDetectorBuilder, "from_all_languages", None)
    if builder_factory is None:
        return None
    return builder_factory().build()


def _iso_code_from_language(language: object) -> Optional[str]:
    """Extract an ISO 639-1 code from a lingua language object."""
    if language is None:
        return None

    iso_code = getattr(language, "iso_code_639_1", None)
    if iso_code is not None:
        for attribute in ("name", "value"):
            value = getattr(iso_code, attribute, None)
            if value:
                return str(value).lower()
        if isinstance(iso_code, str):
            return iso_code.lower()

    name = getattr(language, "name", None)
    if name:
        normalized = str(name).strip().lower()
        if len(normalized) == 2:
            return normalized
    return None


def detect_language(text: str) -> str:
    """Detect the dominant language and return an ISO 639-1 code."""
    sample = str(text or "").strip()
    if not sample:
        return "und"

    detector = _get_language_detector()
    if detector is not None:
        try:
            detected = detector.detect_language_of(sample)
            iso_code = _iso_code_from_language(detected)
            if iso_code:
                return iso_code
        except Exception:  # pragma: no cover - defensive fallback
            pass

    arabic_chars = sum(1 for char in sample if "\u0600" <= char <= "\u06FF")
    latin_chars = sum(1 for char in sample if char.isascii() and char.isalpha())
    if arabic_chars > latin_chars:
        return "ar"
    if re.search(r"\b(?:le|la|les|des|avec|tout|tr[eè]s)\b", sample.lower()):
        return "fr"
    if re.search(r"\b(?:molto|bello|grande|con|tutto)\b", sample.lower()):
        return "it"
    if latin_chars > 0:
        return "en"
    return "und"


@lru_cache(maxsize=1)
def _load_sorted_franco_entries() -> List[Tuple[str, str]]:
    """Load Franco entries sorted by longest phrase first."""
    entries = load_franco_map(DEFAULT_FRANCO_SEED_PATH)
    return sorted(
        ((source.lower(), target) for source, target in entries.items()),
        key=lambda item: (-len(item[0].split()), -len(item[0])),
    )


@lru_cache(maxsize=1)
def _load_franco_detection_entries() -> List[Tuple[str, List[str]]]:
    """Load phrase-token pairs used for Franco ratio estimation."""
    return [
        (phrase, phrase.split())
        for phrase, _ in _load_sorted_franco_entries()
        if phrase not in GENERIC_ENGLISH_FRANCO_ENTRIES
    ]


def _latin_tokenize(text: str) -> List[str]:
    """Tokenize ASCII words and Arabizi digits for Franco matching."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text or "").lower())


def estimate_franco_ratio(text: str) -> float:
    """Estimate how much of the text looks like Franco-Arabic."""
    tokens = _latin_tokenize(text)
    if not tokens:
        return 0.0

    phrase_entries = _load_franco_detection_entries()
    matched_tokens = 0
    index = 0
    while index < len(tokens):
        matched_length = 0
        for _, phrase_tokens in phrase_entries:
            phrase_length = len(phrase_tokens)
            if tokens[index : index + phrase_length] == phrase_tokens:
                matched_length = phrase_length
                break
        if matched_length:
            matched_tokens += matched_length
            index += matched_length
        else:
            index += 1

    return matched_tokens / max(len(tokens), 1)


def contains_significant_franco(text: str, threshold: float = DEFAULT_FRANCO_THRESHOLD) -> bool:
    """Return True when enough tokens look like Franco-Arabic to justify conversion."""
    return estimate_franco_ratio(text) > threshold


def normalize_franco(text: str) -> str:
    """Replace Franco-Arabic phrases with Arabic-script equivalents."""
    normalized = str(text or "").lower()
    if not normalized:
        return normalized

    for source_phrase, target_phrase in _load_sorted_franco_entries():
        pattern = re.compile(rf"(?<![\w]){re.escape(source_phrase)}(?![\w])", flags=re.IGNORECASE)
        normalized = pattern.sub(target_phrase, normalized)

    normalized = re.sub(r"\bemo_pos\b", "EMO_POS", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bemo_neg\b", "EMO_NEG", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bemo_neu\b", "EMO_NEU", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_emojis(text: str) -> str:
    """Convert emojis into sentiment-bearing tokens while preserving other aliases."""
    sample = str(text or "")
    if not sample:
        return ""

    if emoji is None:
        normalized = sample
        for emoji_char, emoji_name in FALLBACK_EMOJI_ALIAS_BY_CHAR.items():
            alias = f":{emoji_name}:"
            token = EMOJI_SENTIMENT.get(alias)
            replacement = f" {token} " if token else f" {alias} "
            normalized = normalized.replace(emoji_char, replacement)
        return re.sub(r"\s+", " ", normalized).strip()

    demojized = emoji.demojize(sample, language="en")
    normalized = demojized
    for alias, token in EMOJI_SENTIMENT.items():
        normalized = normalized.replace(alias, f" {token} ")
    return re.sub(r"\s+", " ", normalized).strip()


def preprocess_text(text: str) -> str:
    """Preprocess raw text into multilingual model-ready text."""
    return get_preprocessor().normalize(text)


class ArabicPreprocessor:
    """Multilingual, transformer-friendly preprocessing with Franco and emoji support."""

    TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
    TATWEEL = re.compile(r"\u0640+")
    ALEF_NORMALIZE = re.compile(r"[\u0622\u0623\u0625\u0671]")
    YEH_NORMALIZE = re.compile(r"\u0649")
    TEH_MARBUTA_NORMALIZE = re.compile(r"\u0629")
    EXTRA_WHITESPACE = re.compile(r"\s+")
    REPEATED_CHAR = re.compile(r"([\u0600-\u06FFA-Za-z])\1{2,}")

    URL_PATTERN = re.compile(r"http[s]?://\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
    HASHTAG_PATTERN = re.compile(r"#([^\s#]+)")
    DEMOJIZED_ALIAS_PATTERN = re.compile(r":([a-zA-Z0-9_\-&+]+):")
    EMOJI_TOKEN_PATTERN = re.compile(r"\bEMO_(?:POS|NEG|NEU)\b")
    PUNCT_SPACING_PATTERN = re.compile(r"\s*([،؛؟!,.:\-\(\)\[\]{}])\s*")

    POSITIVE_EMOJI_HINTS = (
        "heart",
        "smiling",
        "smile",
        "grinning",
        "kiss",
        "thumbs_up",
        "clap",
        "fire",
        "sparkles",
        "party",
        "ok_hand",
        "folded_hands",
        "face_with_tears_of_joy",
        "laugh",
    )
    NEGATIVE_EMOJI_HINTS = (
        "angry",
        "enraged",
        "rage",
        "pouting",
        "frown",
        "sad",
        "cry",
        "broken_heart",
        "thumbs_down",
        "nauseated",
        "vomit",
        "weary",
        "tired",
        "disappointed",
        "confounded",
        "skull",
        "face_with_symbols_on_mouth",
        "unamused",
    )
    NEUTRAL_EMOJI_HINTS = (
        "neutral",
        "expressionless",
        "thinking",
        "face_without_mouth",
        "zipper",
        "eyes",
        "rolling_eyes",
        "grimacing",
    )
    SARCASM_EMOJI_HINTS = (
        "face_with_tears_of_joy",
        "rolling_on_the_floor_laughing",
        "upside_down_face",
        "smirking_face",
    )

    def __init__(
        self,
        remove_noise: bool = True,
        remove_punctuation: bool = False,
        normalize_repeated_chars: bool = True,
        normalize_teh_marbuta: bool = False,
        use_bert_tokenizer: bool = False,
        bert_model_name: str = PRIMARY_ABSA_MODEL_NAME,
        max_length: int = 128,
        franco_threshold: float = DEFAULT_FRANCO_THRESHOLD,
    ) -> None:
        self.remove_noise = remove_noise
        self.remove_punctuation = remove_punctuation
        self.normalize_repeated_chars = normalize_repeated_chars
        self.normalize_teh_marbuta = normalize_teh_marbuta
        self.use_bert_tokenizer = use_bert_tokenizer
        self.max_length = max_length
        self.franco_threshold = franco_threshold
        self.bert_tokenizer: Optional[AutoTokenizer] = None

        if self.use_bert_tokenizer:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    @classmethod
    def classify_emoji_name(cls, emoji_name: str) -> str:
        """Map a demojized emoji name to a coarse sentiment token."""
        alias = str(emoji_name or "").strip().strip(":").lower()
        if alias in EMOJI_NAME_TO_TOKEN:
            return EMOJI_NAME_TO_TOKEN[alias]
        if any(hint in alias for hint in cls.POSITIVE_EMOJI_HINTS):
            return "EMO_POS"
        if any(hint in alias for hint in cls.NEGATIVE_EMOJI_HINTS):
            return "EMO_NEG"
        if any(hint in alias for hint in cls.NEUTRAL_EMOJI_HINTS):
            return "EMO_NEU"
        return "EMO_UNK"

    @classmethod
    def is_sarcasm_emoji(cls, emoji_name: str) -> bool:
        """Return True when an emoji is commonly used sarcastically."""
        alias = str(emoji_name or "").strip().strip(":").lower()
        return any(hint in alias for hint in cls.SARCASM_EMOJI_HINTS)

    def clean(self, text: str) -> str:
        """Remove noisy web artifacts while preserving textual content."""
        sample = str(text or "")
        if not sample:
            return ""

        cleaned = sample
        if self.remove_noise:
            cleaned = self.URL_PATTERN.sub(" ", cleaned)
            cleaned = self.MENTION_PATTERN.sub(" ", cleaned)
            cleaned = self.HASHTAG_PATTERN.sub(r"\1", cleaned)
        return self.EXTRA_WHITESPACE.sub(" ", cleaned).strip()

    def demojize_and_tokenize_emojis(self, text: str) -> Tuple[str, List[str], List[str]]:
        """Replace supported emojis with tokens and keep all detected aliases."""
        normalized = normalize_emojis(text)
        emoji_tokens: List[str] = self.EMOJI_TOKEN_PATTERN.findall(normalized)
        emoji_names: List[str] = []

        def collect_alias(match: re.Match[str]) -> str:
            alias = match.group(1).strip().lower()
            emoji_names.append(alias)
            mapped_token = EMOJI_NAME_TO_TOKEN.get(alias)
            if mapped_token:
                emoji_tokens.append(mapped_token)
                return f" {mapped_token} "
            inferred_token = self.classify_emoji_name(alias)
            if inferred_token != "EMO_UNK":
                emoji_tokens.append(inferred_token)
            return f" :{alias}: "

        replaced = self.DEMOJIZED_ALIAS_PATTERN.sub(collect_alias, normalized)
        return self.EXTRA_WHITESPACE.sub(" ", replaced).strip(), emoji_tokens, emoji_names

    def normalize_arabic_script(self, text: str) -> str:
        """Apply conservative Arabic-script normalization."""
        normalized = self.TASHKEEL.sub("", text)
        normalized = self.TATWEEL.sub("", normalized)
        normalized = self.ALEF_NORMALIZE.sub("ا", normalized)
        normalized = self.YEH_NORMALIZE.sub("ي", normalized)
        if self.normalize_teh_marbuta:
            normalized = self.TEH_MARBUTA_NORMALIZE.sub("ه", normalized)
        if self.normalize_repeated_chars:
            normalized = self.REPEATED_CHAR.sub(r"\1", normalized)
        return normalized

    def normalize_spacing(self, text: str) -> str:
        """Normalize whitespace and punctuation spacing without dropping model hints."""
        spaced = self.PUNCT_SPACING_PATTERN.sub(r" \1 ", text)
        if self.remove_punctuation:
            spaced = re.sub(r"[^\w\s\u0600-\u06FF:]", " ", spaced)
        return self.EXTRA_WHITESPACE.sub(" ", spaced).strip()

    def analyze(self, text: str) -> PreprocessResult:
        """Return the normalized text together with language and emoji metadata."""
        cleaned = self.clean(text)
        normalized_with_emojis, emoji_tokens, emoji_names = self.demojize_and_tokenize_emojis(cleaned)
        detected_language = detect_language(normalized_with_emojis)
        franco_ratio = estimate_franco_ratio(normalized_with_emojis)

        normalized = normalized_with_emojis
        if detected_language != "ar" and franco_ratio > self.franco_threshold:
            normalized = normalize_franco(normalized)

        normalized = self.normalize_arabic_script(normalized)
        normalized = self.normalize_spacing(normalized)
        return PreprocessResult(
            original_text=str(text or ""),
            cleaned_text=cleaned,
            normalized_text=normalized,
            emoji_tokens=emoji_tokens,
            emoji_names=emoji_names,
            language=detected_language,
            franco_token_ratio=round(float(franco_ratio), 6),
        )

    def normalize(self, text: str) -> str:
        """Return only the normalized text for model input."""
        return self.analyze(text).normalized_text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into lightweight sentence segments."""
        sentences = re.split(r"[.!?؟\n]+", self.normalize(text))
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def tokenize(self, text: str) -> List[str]:
        """Whitespace tokenization after normalization."""
        return self.normalize(text).split()

    def bert_encode(self, text: str) -> Dict[str, object]:
        """Tokenize normalized text with a transformer tokenizer."""
        if self.bert_tokenizer is None:
            raise ValueError("BERT tokenizer is not initialized. Set use_bert_tokenizer=True.")

        normalized = self.normalize(text)
        return self.bert_tokenizer(
            normalized,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def preprocess_for_bert(self, text: str) -> Dict[str, object]:
        """Compatibility wrapper for existing training code."""
        return self.bert_encode(text)

    def __call__(self, text: str) -> str:
        return self.normalize(text)


def get_preprocessor() -> ArabicPreprocessor:
    """Factory function for the default multilingual preprocessor."""
    return ArabicPreprocessor(remove_noise=True, remove_punctuation=False)


def get_rule_preprocessor() -> ArabicPreprocessor:
    """Factory for the slightly more normalized rule-matching preprocessor."""
    return ArabicPreprocessor(
        remove_noise=True,
        remove_punctuation=False,
        normalize_teh_marbuta=True,
    )


def get_bert_preprocessor(
    model_name: str = PRIMARY_ABSA_MODEL_NAME,
    max_length: int = 128,
) -> ArabicPreprocessor:
    """Factory function for tokenizer-backed preprocessing."""
    return ArabicPreprocessor(
        remove_noise=True,
        remove_punctuation=False,
        use_bert_tokenizer=True,
        bert_model_name=model_name,
        max_length=max_length,
    )


if __name__ == "__main__":
    examples = [
        "الأكل تحفة 😍",
        "el service msh helw 😡",
        "Very nice shopping but expensive.",
        "Incroyablement grand avec des belles boutiques ❤️",
    ]
    preprocessor = get_preprocessor()
    for sample in examples:
        analysis = preprocessor.analyze(sample)
        print(sample)
        print(analysis.normalized_text)
        print(analysis.language, analysis.franco_token_ratio)
        print(analysis.emoji_tokens)
        print("---")
