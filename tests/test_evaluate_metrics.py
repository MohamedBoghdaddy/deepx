import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate import compute_overall_metrics, compute_per_class_metrics  # noqa: E402


class EvaluateMetricTests(unittest.TestCase):
    def test_per_class_metrics_handle_zero_support(self) -> None:
        probabilities = np.asarray(
            [
                [0.9, 0.1, 0.2],
                [0.8, 0.3, 0.1],
                [0.2, 0.7, 0.4],
            ],
            dtype=np.float32,
        )
        predictions = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        )
        gold = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        )
        label_names = ["label_a", "label_b", "label_c"]

        per_class = compute_per_class_metrics(probabilities, predictions, gold, label_names)

        self.assertEqual(per_class["label_c"]["support"], 0)
        self.assertIsNone(per_class["label_c"]["roc_auc"])
        self.assertIsNone(per_class["label_c"]["average_precision"])

    def test_overall_metrics_include_ranking_sections(self) -> None:
        probabilities = np.asarray(
            [
                [0.9, 0.1, 0.2],
                [0.8, 0.3, 0.1],
                [0.2, 0.7, 0.4],
            ],
            dtype=np.float32,
        )
        predictions = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        )
        gold = np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=int,
        )
        threshold_vector = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
        per_class = compute_per_class_metrics(
            probabilities,
            predictions,
            gold,
            ["label_a", "label_b", "label_c"],
        )

        metrics = compute_overall_metrics(
            probability_matrix=probabilities,
            pred_matrix=predictions,
            gold_matrix=gold,
            threshold_vector=threshold_vector,
            per_class_metrics=per_class,
        )

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("mean_average_precision", metrics)
        self.assertIn("coverage_error", metrics)


if __name__ == "__main__":
    unittest.main()
