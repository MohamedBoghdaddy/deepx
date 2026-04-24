"""Unified evaluation helpers for benchmark model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from benchmark.error_analysis import (
    build_per_aspect_error_analysis,
    save_json as save_analysis_json,
    save_sentiment_confusion_matrices,
)
from benchmark.metrics import (
    compute_aspect_detection_metrics,
    compute_joint_metrics,
    compute_sentiment_metrics,
    normalize_prediction_records,
    predictions_to_submission,
    records_from_dataframe,
    round_nested,
)
from validator import validate_schema


@dataclass
class ModelPredictionBundle:
    """Container for one model's benchmark inference outputs."""

    model_name: str
    model_family: str
    predictions: List[Dict[str, Any]]
    inference_seconds: float
    review_ids: List[Any]
    aspect_probabilities: Optional[np.ndarray] = None
    sentiment_probabilities: Optional[np.ndarray] = None
    thresholds: Optional[Dict[str, float]] = None
    training_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def save_json(data: Any, output_path: Path) -> None:
    """Write a JSON artifact with UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def order_predictions_like_validation(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Align predictions to the validation dataframe order and backfill missing rows."""
    prediction_lookup = {prediction.get("review_id"): dict(prediction) for prediction in predictions}
    ordered: List[Dict[str, Any]] = []

    for gold_record in gold_records:
        prediction = prediction_lookup.get(gold_record.get("review_id"))
        if prediction is None:
            ordered.append(
                {
                    "review_id": gold_record.get("review_id"),
                    "review_text": gold_record.get("review_text", ""),
                    "aspects": ["none"],
                    "aspect_sentiments": {"none": "neutral"},
                }
            )
        else:
            ordered.append(dict(prediction))
    return normalize_prediction_records(ordered)


def evaluate_prediction_bundle(
    validation_df: pd.DataFrame,
    bundle: ModelPredictionBundle,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate, validate, and save artifacts for a single model run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_records = records_from_dataframe(validation_df)
    ordered_predictions = order_predictions_like_validation(gold_records, bundle.predictions)
    submission_like_predictions = predictions_to_submission(ordered_predictions)

    is_valid, validation_errors = validate_schema(submission_like_predictions)
    schema_report = {
        "is_valid": bool(is_valid),
        "errors": validation_errors,
    }
    save_json(schema_report, output_dir / "schema_validation.json")

    aspect_metrics = compute_aspect_detection_metrics(
        gold_records=gold_records,
        pred_records=ordered_predictions,
        aspect_probabilities=bundle.aspect_probabilities,
        thresholds=bundle.thresholds,
    )
    sentiment_metrics, sentiment_match_records = compute_sentiment_metrics(
        gold_records=gold_records,
        pred_records=ordered_predictions,
        sentiment_probabilities=bundle.sentiment_probabilities,
    )
    joint_metrics = compute_joint_metrics(gold_records, ordered_predictions)

    aspect_error_analysis = build_per_aspect_error_analysis(gold_records, ordered_predictions)
    confusion_matrix_artifacts = save_sentiment_confusion_matrices(
        sentiment_match_records,
        output_dir / "confusion_matrices",
    )

    save_analysis_json(round_nested(aspect_error_analysis), output_dir / "aspect_error_analysis.json")
    save_json(round_nested(submission_like_predictions), output_dir / "validation_submission.json")
    save_json(round_nested(ordered_predictions), output_dir / "validation_predictions.json")
    if bundle.thresholds:
        save_json(round_nested(bundle.thresholds), output_dir / "thresholds.json")

    metrics_payload = {
        "aspect_detection": aspect_metrics,
        "sentiment_classification": sentiment_metrics,
        "joint": joint_metrics,
    }
    save_json(round_nested(metrics_payload), output_dir / "metrics.json")

    result = {
        "model_name": bundle.model_name,
        "model_family": bundle.model_family,
        "metrics": round_nested(metrics_payload),
        "thresholds": round_nested(bundle.thresholds or {}),
        "timing": {
            "inference_seconds": round(bundle.inference_seconds, 6),
            "avg_inference_time_ms": round((bundle.inference_seconds * 1000.0) / max(len(validation_df), 1), 6),
            "training_time_seconds": round(bundle.training_seconds, 6) if bundle.training_seconds is not None else None,
        },
        "metadata": round_nested(bundle.metadata),
        "artifacts": {
            "model_output_dir": str(output_dir),
            "metrics": str(output_dir / "metrics.json"),
            "validation_submission": str(output_dir / "validation_submission.json"),
            "validation_predictions": str(output_dir / "validation_predictions.json"),
            "schema_validation": str(output_dir / "schema_validation.json"),
            "aspect_error_analysis": str(output_dir / "aspect_error_analysis.json"),
            "confusion_matrices": confusion_matrix_artifacts,
        },
    }
    save_json(round_nested(result), output_dir / "result.json")
    return result
