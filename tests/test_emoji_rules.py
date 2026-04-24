import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset import ASPECT_SENTIMENT_LABELS, LABEL_TO_IDX  # noqa: E402
from preprocess import ArabicPreprocessor  # noqa: E402
from rules import apply_postprocessing, extract_rule_features  # noqa: E402


def probability_vector(label_name: str, score: float = 0.92) -> np.ndarray:
    vector = np.zeros(len(ASPECT_SENTIMENT_LABELS), dtype=np.float32)
    vector[LABEL_TO_IDX[label_name]] = score
    return vector


class EmojiRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = ArabicPreprocessor()

    def test_positive_emoji_is_preserved_as_sentiment_signal(self) -> None:
        result = self.preprocessor.analyze("الأكل تحفة \U0001F60D")
        self.assertIn("EMO_POS", result.normalized_text)
        decision = apply_postprocessing("الأكل تحفة \U0001F60D", probability_vector("food_positive"))
        self.assertEqual(decision.aspect_sentiments["food"], "positive")

    def test_negative_emoji_is_preserved_as_sentiment_signal(self) -> None:
        result = self.preprocessor.analyze("الخدمة سيئة \U0001F621")
        self.assertIn("EMO_NEG", result.normalized_text)
        decision = apply_postprocessing("الخدمة سيئة \U0001F621", probability_vector("service_negative"))
        self.assertEqual(decision.aspect_sentiments["service"], "negative")

    def test_laughter_emoji_is_flagged_as_sarcasm_candidate(self) -> None:
        features = extract_rule_features("الخدمة ممتازة \U0001F602")
        self.assertTrue(features.sarcasm_candidate)


if __name__ == "__main__":
    unittest.main()
