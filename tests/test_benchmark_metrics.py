import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.metrics import (  # noqa: E402
    compute_aspect_detection_metrics,
    compute_sentiment_metrics,
    tune_aspect_thresholds,
)


class BenchmarkMetricTests(unittest.TestCase):
    def test_aspect_metrics_and_threshold_tuning(self) -> None:
        gold_records = [
            {
                "review_id": 1,
                "review_text": "great food",
                "aspects": ["food"],
                "aspect_sentiments": {"food": "positive"},
            },
            {
                "review_id": 2,
                "review_text": "bad service",
                "aspects": ["service"],
                "aspect_sentiments": {"service": "negative"},
            },
            {
                "review_id": 3,
                "review_text": "nothing special",
                "aspects": ["none"],
                "aspect_sentiments": {"none": "neutral"},
            },
        ]
        pred_records = [
            {
                "review_id": 1,
                "review_text": "great food",
                "aspects": ["food"],
                "aspect_sentiments": {"food": "positive"},
            },
            {
                "review_id": 2,
                "review_text": "bad service",
                "aspects": ["service"],
                "aspect_sentiments": {"service": "neutral"},
            },
            {
                "review_id": 3,
                "review_text": "nothing special",
                "aspects": ["none"],
                "aspect_sentiments": {"none": "neutral"},
            },
        ]
        aspect_probabilities = np.asarray(
            [
                [0.91, 0.04, 0.02, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08],
                [0.05, 0.88, 0.06, 0.03, 0.01, 0.05, 0.04, 0.04, 0.09],
                [0.04, 0.05, 0.02, 0.01, 0.01, 0.03, 0.02, 0.02, 0.84],
            ],
            dtype=np.float32,
        )

        thresholds, score = tune_aspect_thresholds(aspect_probabilities, gold_records, granularity="aspect")
        self.assertIn("food", thresholds)
        self.assertGreaterEqual(score, 0.9)

        metrics = compute_aspect_detection_metrics(
            gold_records=gold_records,
            pred_records=pred_records,
            aspect_probabilities=aspect_probabilities,
            thresholds=thresholds,
        )
        self.assertAlmostEqual(metrics["micro_f1"], 1.0)
        self.assertIn("food", metrics["per_label"])
        self.assertIsNotNone(metrics["pr_auc"]["micro"])

    def test_sentiment_metrics_use_matched_aspect_instances(self) -> None:
        gold_records = [
            {
                "review_id": 1,
                "review_text": "great food",
                "aspects": ["food"],
                "aspect_sentiments": {"food": "positive"},
            },
            {
                "review_id": 2,
                "review_text": "slow delivery",
                "aspects": ["delivery"],
                "aspect_sentiments": {"delivery": "negative"},
            },
        ]
        pred_records = [
            {
                "review_id": 1,
                "review_text": "great food",
                "aspects": ["food"],
                "aspect_sentiments": {"food": "positive"},
            },
            {
                "review_id": 2,
                "review_text": "slow delivery",
                "aspects": ["none"],
                "aspect_sentiments": {"none": "neutral"},
            },
        ]
        sentiment_probabilities = np.zeros((2, 9, 3), dtype=np.float32)
        sentiment_probabilities[0, 0] = np.asarray([0.92, 0.03, 0.05], dtype=np.float32)
        sentiment_probabilities[1, 4] = np.asarray([0.05, 0.90, 0.05], dtype=np.float32)

        metrics, matches = compute_sentiment_metrics(
            gold_records=gold_records,
            pred_records=pred_records,
            sentiment_probabilities=sentiment_probabilities,
        )
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertAlmostEqual(metrics["micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

