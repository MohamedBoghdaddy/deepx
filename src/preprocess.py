"""
Arabic Text Preprocessing Module
=================================
Provides Arabic-specific preprocessing for NLP tasks.

Features:
- Text cleaning
- Arabic normalization
- Text splitting
- Basic tokenization
- Optional stopword removal
- Optional light stemming placeholder
- BERT/MARBERT tokenizer support
"""

import re
from typing import List, Dict, Optional

from transformers import AutoTokenizer


class ArabicPreprocessor:
    """Arabic text preprocessor for NLP tasks."""

    TASHKEEL = re.compile(r'[\u064B-\u065F\u0670]')
    TATWEEL = re.compile(r'ـ+')
    ALEF_NORMALIZE = re.compile(r'[إأآا]')
    YAA_NORMALIZE = re.compile(r'[يى]')
    TAA_MARBUTA_NORMALIZE = re.compile(r'ة')
    REPEATED_CHAR = re.compile(r'(.)\1+')
    EXTRA_WHITESPACE = re.compile(r'\s+')

    URL_PATTERN = re.compile(r'http[s]?://\S+|www\.\S+')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#(\w+)')
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s\u0600-\u06FF]')

    ARABIC_STOPWORDS = {
        "من", "في", "على", "الى", "إلى", "عن", "مع",
        "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "هم",
        "كان", "كانت", "يكون", "ان", "أن", "إن",
        "و", "ف", "ثم", "او", "أو", "لا", "ما"
    }

    def __init__(
        self,
        remove_noise: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        normalize_repeated_chars: bool = True,
        use_bert_tokenizer: bool = False,
        bert_model_name: str = "aubmindlab/bert-base-arabertv02",
        max_length: int = 128
    ):
        self.remove_noise = remove_noise
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.normalize_repeated_chars = normalize_repeated_chars
        self.use_bert_tokenizer = use_bert_tokenizer
        self.max_length = max_length

        self.bert_tokenizer: Optional[AutoTokenizer] = None

        if self.use_bert_tokenizer:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def clean(self, text: str) -> str:
        """Remove URLs, mentions, punctuation, and clean hashtags."""
        if not isinstance(text, str):
            return ""

        if self.remove_noise:
            text = self.URL_PATTERN.sub("", text)
            text = self.MENTION_PATTERN.sub("", text)

            # Keep hashtag word instead of deleting it
            text = self.HASHTAG_PATTERN.sub(r"\1", text)

        if self.remove_punctuation:
            text = self.PUNCTUATION_PATTERN.sub(" ", text)

        text = self.EXTRA_WHITESPACE.sub(" ", text).strip()
        return text

    def normalize(self, text: str) -> str:
        """Apply Arabic normalization."""
        if not isinstance(text, str):
            return ""

        text = self.clean(text)

        # Remove tashkeel
        text = self.TASHKEEL.sub("", text)

        # Normalize Alef forms
        text = self.ALEF_NORMALIZE.sub("ا", text)

        # Normalize Yaa / Alef Maqsura
        text = self.YAA_NORMALIZE.sub("ي", text)

        # Normalize Taa Marbuta
        text = self.TAA_MARBUTA_NORMALIZE.sub("ه", text)

        # Remove tatweel
        text = self.TATWEEL.sub("", text)

        # Normalize repeated characters, keep max 2
        if self.normalize_repeated_chars:
            text = self.REPEATED_CHAR.sub(r"\1\1", text)

        text = self.EXTRA_WHITESPACE.sub(" ", text).strip()

        return text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        text = self.normalize(text)

        sentences = re.split(r'[.!؟?،\n]+', text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        return sentences

    def tokenize(self, text: str) -> List[str]:
        """Basic whitespace tokenization."""
        text = self.normalize(text)
        tokens = text.split()

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.ARABIC_STOPWORDS]

        return tokens

    def light_stem(self, token: str) -> str:
        """
        Very simple Arabic light stemming.
        For serious projects, use CAMeL Tools, Farasa, or ISRI stemmer.

        Note:
        With BERT/MARBERT, stemming is usually NOT recommended.
        """
        prefixes = ["ال", "وال", "بال", "كال", "فال", "لل"]
        suffixes = ["ه", "ها", "هم", "كما", "نا", "ون", "ين", "ات"]

        for prefix in prefixes:
            if token.startswith(prefix) and len(token) > len(prefix) + 2:
                token = token[len(prefix):]
                break

        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
                break

        return token

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply light stemming to tokens."""
        return [self.light_stem(token) for token in tokens]

    def bert_encode(self, text: str) -> Dict:
        """
        Tokenize text using BERT/MARBERT tokenizer.

        Important:
        For BERT/MARBERT, do not remove stopwords or apply stemming.
        The model needs the full sentence context.
        """
        if self.bert_tokenizer is None:
            raise ValueError("BERT tokenizer is not initialized. Set use_bert_tokenizer=True.")

        text = self.normalize(text)

        return self.bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def preprocess_for_classical_ml(self, text: str, apply_stemming: bool = False) -> List[str]:
        """
        Preprocessing for classical ML models like:
        Logistic Regression, SVM, Naive Bayes, Random Forest.
        """
        tokens = self.tokenize(text)

        if apply_stemming:
            tokens = self.stem_tokens(tokens)

        return tokens

    def preprocess_for_bert(self, text: str) -> Dict:
        """
        Preprocessing for BERT/MARBERT/AraBERT models.
        """
        return self.bert_encode(text)

    def __call__(self, text: str) -> str:
        """Return normalized text."""
        return self.normalize(text)


def get_preprocessor() -> ArabicPreprocessor:
    """Factory function for normal preprocessing."""
    return ArabicPreprocessor(
        remove_noise=True,
        remove_punctuation=False,
        remove_stopwords=False
    )


def get_bert_preprocessor(
    model_name: str = "aubmindlab/bert-base-arabertv02"
) -> ArabicPreprocessor:
    """Factory function for BERT/MARBERT preprocessing."""
    return ArabicPreprocessor(
        remove_noise=True,
        remove_punctuation=False,
        remove_stopwords=False,
        use_bert_tokenizer=True,
        bert_model_name=model_name,
        max_length=128
    )


if __name__ == "__main__":
    test_text = "ممتاااز جداً!!! 👏👏👏 http://example.com @user #good"

    preprocessor = ArabicPreprocessor(remove_stopwords=True)

    print("Original:")
    print(test_text)

    print("\nNormalized:")
    print(preprocessor.normalize(test_text))

    print("\nSentences:")
    print(preprocessor.split_sentences(test_text))

    print("\nTokens:")
    print(preprocessor.tokenize(test_text))

    print("\nStemmed Tokens:")
    print(preprocessor.stem_tokens(preprocessor.tokenize(test_text)))

    bert_preprocessor = get_bert_preprocessor()

    encoded = bert_preprocessor.preprocess_for_bert(test_text)

    print("\nBERT input_ids:")
    print(encoded["input_ids"])

    print("\nBERT attention_mask:")
    print(encoded["attention_mask"])