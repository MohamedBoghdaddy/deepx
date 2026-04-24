import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocess import (  # noqa: E402
    ArabicPreprocessor,
    contains_significant_franco,
    normalize_franco,
)


class MultilingualPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = ArabicPreprocessor()

    def test_normalize_franco_prefers_longest_phrase(self) -> None:
        normalized = normalize_franco("fol awy bas msh helw")
        self.assertIn("فل أوي", normalized)
        self.assertIn("مش حلو", normalized)

    def test_preprocess_converts_franco_and_preserves_emoji_sentiment(self) -> None:
        result = self.preprocessor.analyze("mesh helw 😡")
        self.assertIn("مش حلو", result.normalized_text)
        self.assertIn("EMO_NEG", result.normalized_text)
        self.assertIn("EMO_NEG", result.emoji_tokens)

    def test_plain_english_is_not_forced_into_arabic(self) -> None:
        result = self.preprocessor.analyze("Very nice shopping but expensive.")
        self.assertIn("nice", result.normalized_text.lower())
        self.assertNotIn("حلو", result.normalized_text)

    def test_significant_franco_ratio_detects_transliterated_text(self) -> None:
        self.assertTrue(contains_significant_franco("mesh helw kwayes momtaz"))
        self.assertFalse(contains_significant_franco("Very nice shopping but expensive"))


if __name__ == "__main__":
    unittest.main()
