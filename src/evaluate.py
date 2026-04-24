"""
Validation evaluation for Arabic ABSA using the same post-processing as submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from dataset import (
    ASPECT_SENTIMENT_LABELS,
    DEFAULT_VALIDATION_PATH,
    OUTPUTS_ROOT,
    create_multi_label_vector,
    infer_column_mapping,
    load_dataframe,
    parse_json_column,
    parse_sentiment_dict,
    resolve_input_path,
)
from predict import DEFAULT_MODEL_PATH, predict_dataframe
from train import compute_metrics


DEFAULT_OUTPUT_PATH = OUTPUTS_ROOT / "validation_metrics.json"


def predictions_to_matrix(predictions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Convert prediction dictionaries into the 27-label multi-hot matrix."""
    vectors = [
        create_multi_label_vector(prediction["aspects"], prediction["aspect_sentiments"])
        for prediction in predictions
    ]
    return np.vstack(vectors).astype(np.float32)


def labels_from_dataframe(dataframe: pd.DataFrame) -> np.ndarray:
    """Extract the gold multi-label matrix from a validation DataFrame."""
    mapping = infer_column_mapping(dataframe, require_labels=True)
    vectors = []
    for _, row in dataframe.iterrows():
        aspects = parse_json_column(row[mapping.aspects]) if mapping.aspects else []
        sentiments = parse_sentiment_dict(row[mapping.aspect_sentiments]) if mapping.aspect_sentiments else {}
        vectors.append(create_multi_label_vector(aspects, sentiments))
    return np.vstack(vectors).astype(np.float32)


def compute_per_label_metrics(pred_matrix: np.ndarray, gold_matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute label-level precision/recall/F1 for reporting."""
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_matrix.astype(int),
        pred_matrix.astype(int),
        average=None,
        zero_division=0,
    )
    return {
        label_name: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label_name in enumerate(ASPECT_SENTIMENT_LABELS)
    }


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    validation_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute validation metrics on post-processed predictions."""
    pred_matrix = predictions_to_matrix(predictions)
    gold_matrix = labels_from_dataframe(validation_df)
    raw_scores = pred_matrix.astype(np.float32)
    overall_metrics = compute_metrics(raw_scores, gold_matrix, threshold=0.5)
    overall_metrics["weighted_f1"] = float(
        f1_score(gold_matrix.astype(int), pred_matrix.astype(int), average="weighted", zero_division=0)
    )
    overall_metrics["num_samples"] = int(len(validation_df))
    overall_metrics["num_labels"] = int(pred_matrix.shape[1])
    overall_metrics["avg_predicted_labels"] = round(float(pred_matrix.sum(axis=1).mean()), 6)
    overall_metrics["avg_true_labels"] = round(float(gold_matrix.sum(axis=1).mean()), 6)
    overall_metrics["per_label"] = compute_per_label_metrics(pred_matrix, gold_matrix)
    return overall_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate Arabic ABSA predictions on validation data.")
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    model_path = resolve_input_path(args.model_path, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH

    validation_df = load_dataframe(validation_path)
    predictions = predict_dataframe(
        dataframe=validation_df,
        model_path=model_path,
        base_model_name=args.base_model_name,
        threshold_path=threshold_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    metrics = evaluate_predictions(predictions, validation_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "num_samples": metrics["num_samples"],
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
