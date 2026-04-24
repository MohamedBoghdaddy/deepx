"""
Arabic text preprocessing utilities with emoji-aware sentiment preservation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # pragma: no cover - exercised in runtime environments with the dependency installed
    import emoji
except ImportError:  # pragma: no cover - graceful fallback before requirements are installed
    emoji = None

from transformers import AutoTokenizer


@dataclass
class PreprocessResult:
    """Structured preprocessing result for downstream explainability and rules."""

    original_text: str
    cleaned_text: str
    normalized_text: str
    emoji_tokens: List[str]
    emoji_names: List[str]


class ArabicPreprocessor:
    """Strong but transformer-friendly Arabic preprocessing with emoji preservation."""

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
    EMOJI_ALIAS_PATTERN = re.compile(r":([a-zA-Z0-9_\-&+]+):")
    PUNCT_SPACING_PATTERN = re.compile(r"\s*([،؛؟!,.:\-\(\)\[\]{}])\s*")

    ENGLISH_NORMALIZATION = {
        "expensive": "غالي",
        "bad": "سيء",
        "good": "جيد",
        "great": "ممتاز",
        "excellent": "ممتاز",
        "clean": "نظيف",
        "dirty": "وسخ",
        "slow": "بطيء",
        "fast": "سريع",
    }

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
    FALLBACK_EMOJI_NAME_BY_CHAR = {
        "😍": "smiling_face_with_heart_eyes",
        "❤️": "red_heart",
        "❤": "red_heart",
        "💖": "sparkling_heart",
        "👍": "thumbs_up",
        "👏": "clapping_hands",
        "😡": "enraged_face",
        "😠": "angry_face",
        "👎": "thumbs_down",
        "💔": "broken_heart",
        "😐": "neutral_face",
        "😑": "expressionless_face",
        "🤔": "thinking_face",
        "😂": "face_with_tears_of_joy",
        "🤣": "rolling_on_the_floor_laughing",
        "🙃": "upside_down_face",
    }

    def __init__(
        self,
        remove_noise: bool = True,
        remove_punctuation: bool = False,
        normalize_repeated_chars: bool = True,
        normalize_teh_marbuta: bool = False,
        use_bert_tokenizer: bool = False,
        bert_model_name: str = "aubmindlab/bert-base-arabertv02",
        max_length: int = 128,
    ) -> None:
        self.remove_noise = remove_noise
        self.remove_punctuation = remove_punctuation
        self.normalize_repeated_chars = normalize_repeated_chars
        self.normalize_teh_marbuta = normalize_teh_marbuta
        self.use_bert_tokenizer = use_bert_tokenizer
        self.max_length = max_length
        self.bert_tokenizer: Optional[AutoTokenizer] = None

        if self.use_bert_tokenizer:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    @classmethod
    def classify_emoji_name(cls, emoji_name: str) -> str:
        """Map a demojized emoji name to a coarse sentiment token."""
        name = emoji_name.lower()
        if any(hint in name for hint in cls.POSITIVE_EMOJI_HINTS):
            return "EMO_POS"
        if any(hint in name for hint in cls.NEGATIVE_EMOJI_HINTS):
            return "EMO_NEG"
        if any(hint in name for hint in cls.NEUTRAL_EMOJI_HINTS):
            return "EMO_NEU"
        return "EMO_UNK"

    @classmethod
    def is_sarcasm_emoji(cls, emoji_name: str) -> bool:
        """Return True when an emoji is commonly used sarcastically."""
        lowered = emoji_name.lower()
        return any(hint in lowered for hint in cls.SARCASM_EMOJI_HINTS)

    def clean(self, text: str) -> str:
        """Remove noisy web artifacts while preserving emoji-derived tokens."""
        if not isinstance(text, str):
            return ""

        cleaned = text
        if self.remove_noise:
            cleaned = self.URL_PATTERN.sub(" ", cleaned)
            cleaned = self.MENTION_PATTERN.sub(" ", cleaned)
            cleaned = self.HASHTAG_PATTERN.sub(r"\1", cleaned)
        return self.EXTRA_WHITESPACE.sub(" ", cleaned).strip()

    def normalize_common_english_words(self, text: str) -> str:
        """Translate frequent sentiment-bearing English words into Arabic hints."""
        normalized = text
        for english_word, arabic_word in self.ENGLISH_NORMALIZATION.items():
            normalized = re.sub(
                rf"\b{re.escape(english_word)}\b",
                arabic_word,
                normalized,
                flags=re.IGNORECASE,
            )
        return normalized

    def demojize_and_tokenize_emojis(self, text: str) -> tuple[str, List[str], List[str]]:
        """Replace emojis with coarse sentiment tokens and keep the detected aliases."""
        if not isinstance(text, str):
            return "", [], []

        if emoji is None:
            replaced = text
            emoji_tokens: List[str] = []
            emoji_names: List[str] = []
            for emoji_char, emoji_name in self.FALLBACK_EMOJI_NAME_BY_CHAR.items():
                if emoji_char in replaced:
                    token = self.classify_emoji_name(emoji_name)
                    occurrences = replaced.count(emoji_char)
                    emoji_tokens.extend([token] * occurrences)
                    emoji_names.extend([emoji_name] * occurrences)
                    replaced = replaced.replace(emoji_char, f" {token} ")
            return replaced, emoji_tokens, emoji_names

        demojized = emoji.demojize(text, language="en", delimiters=(":", ":"))
        emoji_tokens: List[str] = []
        emoji_names: List[str] = []

        def replace_alias(match: re.Match[str]) -> str:
            emoji_name = match.group(1).strip().lower()
            token = self.classify_emoji_name(emoji_name)
            emoji_tokens.append(token)
            emoji_names.append(emoji_name)
            return f" {token} "

        replaced = self.EMOJI_ALIAS_PATTERN.sub(replace_alias, demojized)
        return replaced, emoji_tokens, emoji_names

    def normalize_spacing(self, text: str) -> str:
        """Normalize whitespace and punctuation spacing without stripping emojis."""
        spaced = self.PUNCT_SPACING_PATTERN.sub(r" \1 ", text)
        if self.remove_punctuation:
            spaced = re.sub(r"[^\w\s\u0600-\u06FF]", " ", spaced)
        return self.EXTRA_WHITESPACE.sub(" ", spaced).strip()

    def analyze(self, text: str) -> PreprocessResult:
        """Return the cleaned text together with the extracted emoji tokens."""
        cleaned = self.clean(text)
        cleaned = self.normalize_common_english_words(cleaned)
        cleaned, emoji_tokens, emoji_names = self.demojize_and_tokenize_emojis(cleaned)
        normalized = cleaned
        normalized = self.TASHKEEL.sub("", normalized)
        normalized = self.TATWEEL.sub("", normalized)
        normalized = self.ALEF_NORMALIZE.sub("ا", normalized)
        normalized = self.YEH_NORMALIZE.sub("ي", normalized)
        if self.normalize_teh_marbuta:
            normalized = self.TEH_MARBUTA_NORMALIZE.sub("ه", normalized)
        if self.normalize_repeated_chars:
            normalized = self.REPEATED_CHAR.sub(r"\1", normalized)
        normalized = self.normalize_spacing(normalized)
        return PreprocessResult(
            original_text=text if isinstance(text, str) else "",
            cleaned_text=cleaned,
            normalized_text=normalized,
            emoji_tokens=emoji_tokens,
            emoji_names=emoji_names,
        )

    def normalize(self, text: str) -> str:
        """Return only the normalized text for model input."""
        return self.analyze(text).normalized_text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into light-weight sentence segments."""
        normalized = self.normalize(text)
        sentences = re.split(r"[.!?؟\n]+", normalized)
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
    """Factory function for the default preprocessor."""
    return ArabicPreprocessor(remove_noise=True, remove_punctuation=False)


def get_rule_preprocessor() -> ArabicPreprocessor:
    """Factory for the slightly more normalized rule-matching preprocessor."""
    return ArabicPreprocessor(
        remove_noise=True,
        remove_punctuation=False,
        normalize_teh_marbuta=True,
    )


def get_bert_preprocessor(
    model_name: str = "aubmindlab/bert-base-arabertv02",
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
        "الخدمة سيئة 😡",
        "الخدمة ممتازة 😂",
        "good app but expensive!!!",
    ]
    preprocessor = get_preprocessor()
    for sample in examples:
        analysis = preprocessor.analyze(sample)
        print(sample)
        print(analysis.normalized_text)
        print(analysis.emoji_tokens)
        print(analysis.emoji_names)
        print("---")
