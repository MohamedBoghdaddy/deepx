"""
Arabic Text Preprocessing Module
=================================
Light, transformer-safe preprocessing for AraBERT and MARBERT.

Rules:
- normalize Arabic letters lightly
- remove tashkeel
- remove tatweel
- remove URLs and mentions
- keep hashtags as text
- keep emojis and most punctuation
- do not remove stopwords
- do not apply stemming
"""

import re
from typing import Dict, List, Optional

from transformers import AutoTokenizer


class ArabicPreprocessor:
    """Arabic text preprocessor for transformer-based NLP tasks."""

    TASHKEEL = re.compile(r"[\u064B-\u065F\u0670]")
    TATWEEL = re.compile(r"\u0640+")
    ALEF_NORMALIZE = re.compile(r"[\u0622\u0623\u0625\u0671]")
    ALEF_MAKSURA_NORMALIZE = re.compile(r"\u0649")
    EXTRA_WHITESPACE = re.compile(r"\s+")
    REPEATED_CHAR = re.compile(r"(.)\1{2,}")

    URL_PATTERN = re.compile(r"http[s]?://\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
    HASHTAG_PATTERN = re.compile(r"#([^\s#]+)")

    def __init__(
        self,
        remove_noise: bool = True,
        remove_punctuation: bool = False,
        normalize_repeated_chars: bool = True,
        use_bert_tokenizer: bool = False,
        bert_model_name: str = "aubmindlab/bert-base-arabertv02",
        max_length: int = 128,
    ):
        self.remove_noise = remove_noise
        self.remove_punctuation = remove_punctuation
        self.normalize_repeated_chars = normalize_repeated_chars
        self.use_bert_tokenizer = use_bert_tokenizer
        self.max_length = max_length
        self.bert_tokenizer: Optional[AutoTokenizer] = None

        if self.use_bert_tokenizer:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def clean(self, text: str) -> str:
        """Remove URLs and mentions while preserving hashtags as text."""
        if not isinstance(text, str):
            return ""

        if self.remove_noise:
            text = self.URL_PATTERN.sub(" ", text)
            text = self.MENTION_PATTERN.sub(" ", text)
            text = self.HASHTAG_PATTERN.sub(r"\1", text)

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

        return self.EXTRA_WHITESPACE.sub(" ", text).strip()

    def normalize(self, text: str) -> str:
        """Apply light Arabic normalization suitable for transformers."""
        if not isinstance(text, str):
            return ""

        text = self.clean(text)
        text = self.TASHKEEL.sub("", text)
        text = self.TATWEEL.sub("", text)
        text = self.ALEF_NORMALIZE.sub("ا", text)
        text = self.ALEF_MAKSURA_NORMALIZE.sub("ي", text)

        if self.normalize_repeated_chars:
            text = self.REPEATED_CHAR.sub(r"\1\1", text)

        return self.EXTRA_WHITESPACE.sub(" ", text).strip()

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with minimal punctuation handling."""
        text = self.normalize(text)
        sentences = re.split(r"[.!?؟\n]+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def tokenize(self, text: str) -> List[str]:
        """Basic whitespace tokenization after normalization."""
        return self.normalize(text).split()

    def bert_encode(self, text: str) -> Dict:
        """Tokenize text using a transformer tokenizer."""
        if self.bert_tokenizer is None:
            raise ValueError("BERT tokenizer is not initialized. Set use_bert_tokenizer=True.")

        text = self.normalize(text)
        return self.bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def preprocess_for_bert(self, text: str) -> Dict:
        """Preprocessing helper for AraBERT and MARBERT models."""
        return self.bert_encode(text)

    def __call__(self, text: str) -> str:
        return self.normalize(text)


def get_preprocessor() -> ArabicPreprocessor:
    """Factory function for the default transformer-safe preprocessor."""
    return ArabicPreprocessor(remove_noise=True, remove_punctuation=False)


def get_bert_preprocessor(
    model_name: str = "aubmindlab/bert-base-arabertv02",
    max_length: int = 128,
) -> ArabicPreprocessor:
    """Factory function for a tokenizer-backed transformer preprocessor."""
    return ArabicPreprocessor(
        remove_noise=True,
        remove_punctuation=False,
        use_bert_tokenizer=True,
        bert_model_name=model_name,
        max_length=max_length,
    )


if __name__ == "__main__":
    test_text = "ممتاااز جدا!!! 👏👏👏 http://example.com @user #مطعم_جميل"
    preprocessor = get_preprocessor()
    print("Original:")
    print(test_text)
    print("\nNormalized:")
    print(preprocessor.normalize(test_text))
