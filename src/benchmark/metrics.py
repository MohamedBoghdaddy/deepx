"""Shared metrics and threshold tuning utilities for benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

from dataset import (
    VALID_ASPECTS,
    VALID_SENTIMENTS,
    coerce_review_id,
    create_multi_label_vector,
    infer_column_mapping,
    parse_json_column,
    parse_sentiment_dict,
    sanitize_aspect_sentiments,
)


ASPECT_TO_INDEX = {aspect: index for index, aspect in enumerate(VALID_ASPECTS)}
SENTIMENT_TO_INDEX = {sentiment: index for index, sentiment in enumerate(VALID_SENTIMENTS)}


@dataclass
class SentimentMatchRecord:
    """One matched aspect instance for sentiment evaluation."""

    review_id: Any
    review_index: int
    aspect: str
    gold_sentiment: str
    pred_sentiment: str
    score_vector: Optional[np.ndarray] = None


def round_float(value: Optional[float]) -> Optional[float]:
    """Round a float while preserving ``None`` values."""
    if value is None:
        return None
    return round(float(value), 6)


def round_nested(value: Any) -> Any:
    """Recursively round floating-point values for artifact serialization."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    return value


def records_from_dataframe(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a labeled dataframe into normalized benchmark records."""
    mapping = infer_column_mapping(dataframe, require_labels=True)
    records: List[Dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        aspects = parse_json_column(row[mapping.aspects]) if mapping.aspects else []
        aspect_sentiments = (
            parse_sentiment_dict(row[mapping.aspect_sentiments])
            if mapping.aspect_sentiments
            else {}
        )
        safe_aspects, safe_sentiments = sanitize_aspect_sentiments(aspects, aspect_sentiments)
        records.append(
            {
                "review_id": coerce_review_id(row[mapping.review_id]),
                "review_text": str(row[mapping.review_text]),
                "aspects": safe_aspects,
                "aspect_sentiments": safe_sentiments,
            }
        )
    return records


def normalize_prediction_records(predictions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize prediction payloads into the benchmark schema."""
    normalized: List[Dict[str, Any]] = []
    for prediction in predictions:
        aspects = list(prediction.get("aspects", []))
        aspect_sentiments = dict(prediction.get("aspect_sentiments", {}))
        safe_aspects, safe_sentiments = sanitize_aspect_sentiments(aspects, aspect_sentiments)
        normalized.append(
            {
                "review_id": coerce_review_id(prediction.get("review_id")),
                "review_text": str(prediction.get("review_text", "")),
                "aspects": safe_aspects,
                "aspect_sentiments": safe_sentiments,
            }
        )
    return normalized


def build_prediction_lookup(
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[Any, Dict[str, Any]]:
    """Key predictions by review id for alignment with validation data."""
    return {prediction.get("review_id"): dict(prediction) for prediction in predictions}


def aspect_matrix_from_records(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build the multi-label aspect matrix from benchmark records."""
    matrix = np.zeros((len(records), len(VALID_ASPECTS)), dtype=np.int64)
    for row_index, record in enumerate(records):
        for aspect in record.get("aspects", []):
            if aspect in ASPECT_TO_INDEX:
                matrix[row_index, ASPECT_TO_INDEX[aspect]] = 1
    return matrix


def joint_matrix_from_records(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build the joint aspect-sentiment matrix from benchmark records."""
    vectors = [
        create_multi_label_vector(record.get("aspects", []), record.get("aspect_sentiments", {}))
        for record in records
    ]
    return np.vstack(vectors).astype(np.int64) if vectors else np.zeros((0, 27), dtype=np.int64)


def decode_prediction(
    aspect_probabilities: Sequence[float],
    sentiment_probabilities: Sequence[Sequence[float]],
    thresholds: Mapping[str, float],
) -> Tuple[List[str], Dict[str, str]]:
    """Decode model scores into the competition submission schema."""
    selected_aspects = [
        aspect
        for aspect, threshold in thresholds.items()
        if float(aspect_probabilities[ASPECT_TO_INDEX[aspect]]) >= float(threshold)
    ]

    aspect_sentiments: Dict[str, str] = {}
    for aspect in selected_aspects:
        if aspect == "none":
            aspect_sentiments[aspect] = "neutral"
            continue
        aspect_index = ASPECT_TO_INDEX[aspect]
        sentiment_index = int(np.argmax(np.asarray(sentiment_probabilities[aspect_index], dtype=np.float32)))
        aspect_sentiments[aspect] = VALID_SENTIMENTS[sentiment_index]

    safe_aspects, safe_sentiments = sanitize_aspect_sentiments(selected_aspects, aspect_sentiments)
    return safe_aspects, safe_sentiments


def aspect_predictions_from_probabilities(
    probability_matrix: np.ndarray,
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """Convert aspect probabilities into a valid prediction matrix."""
    predictions = np.zeros_like(probability_matrix, dtype=np.int64)
    for row_index, row in enumerate(probability_matrix):
        selected_aspects = [
            aspect
            for aspect in VALID_ASPECTS
            if float(row[ASPECT_TO_INDEX[aspect]]) >= float(thresholds[aspect])
        ]
        safe_aspects, _ = sanitize_aspect_sentiments(selected_aspects, {aspect: "neutral" for aspect in selected_aspects})
        for aspect in safe_aspects:
            predictions[row_index, ASPECT_TO_INDEX[aspect]] = 1
    return predictions


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Return PR-AUC when the metric is defined."""
    if y_true.ndim != 1:
        raise ValueError("safe_average_precision expects 1D vectors.")
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    label_names: Sequence[str],
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics for the aspect detection task."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label: Dict[str, Dict[str, Any]] = {}
    pr_auc_values: List[float] = []
    for index, label_name in enumerate(label_names):
        label_pr_auc = None
        if y_score is not None:
            label_pr_auc = safe_average_precision(y_true[:, index], y_score[:, index])
            if label_pr_auc is not None:
                pr_auc_values.append(label_pr_auc)

        per_label[label_name] = {
            "precision": round_float(float(precision[index])),
            "recall": round_float(float(recall[index])),
            "f1": round_float(float(f1[index])),
            "support": int(support[index]),
            "threshold": round_float(float(thresholds[label_name])) if thresholds and label_name in thresholds else None,
            "true_positives": int(np.sum((y_true[:, index] == 1) & (y_pred[:, index] == 1))),
            "false_positives": int(np.sum((y_true[:, index] == 0) & (y_pred[:, index] == 1))),
            "false_negatives": int(np.sum((y_true[:, index] == 1) & (y_pred[:, index] == 0))),
            "predicted_positives": int(y_pred[:, index].sum()),
            "pr_auc": round_float(label_pr_auc),
        }

    micro_pr_auc = None
    macro_pr_auc = None
    if y_score is not None and int(y_true.sum()) > 0:
        micro_pr_auc = float(average_precision_score(y_true, y_score, average="micro"))
        macro_pr_auc = float(np.mean(pr_auc_values)) if pr_auc_values else None

    return {
        "micro_precision": round_float(float(precision_score(y_true, y_pred, average="micro", zero_division=0))),
        "macro_precision": round_float(float(precision_score(y_true, y_pred, average="macro", zero_division=0))),
        "weighted_precision": round_float(float(precision_score(y_true, y_pred, average="weighted", zero_division=0))),
        "micro_recall": round_float(float(recall_score(y_true, y_pred, average="micro", zero_division=0))),
        "macro_recall": round_float(float(recall_score(y_true, y_pred, average="macro", zero_division=0))),
        "weighted_recall": round_float(float(recall_score(y_true, y_pred, average="weighted", zero_division=0))),
        "micro_f1": round_float(float(f1_score(y_true, y_pred, average="micro", zero_division=0))),
        "macro_f1": round_float(float(f1_score(y_true, y_pred, average="macro", zero_division=0))),
        "weighted_f1": round_float(float(f1_score(y_true, y_pred, average="weighted", zero_division=0))),
        "precision": {
            "micro": round_float(float(precision_score(y_true, y_pred, average="micro", zero_division=0))),
            "macro": round_float(float(precision_score(y_true, y_pred, average="macro", zero_division=0))),
            "weighted": round_float(float(precision_score(y_true, y_pred, average="weighted", zero_division=0))),
        },
        "recall": {
            "micro": round_float(float(recall_score(y_true, y_pred, average="micro", zero_division=0))),
            "macro": round_float(float(recall_score(y_true, y_pred, average="macro", zero_division=0))),
            "weighted": round_float(float(recall_score(y_true, y_pred, average="weighted", zero_division=0))),
        },
        "f1": {
            "micro": round_float(float(f1_score(y_true, y_pred, average="micro", zero_division=0))),
            "macro": round_float(float(f1_score(y_true, y_pred, average="macro", zero_division=0))),
            "weighted": round_float(float(f1_score(y_true, y_pred, average="weighted", zero_division=0))),
        },
        "subset_accuracy": round_float(float(accuracy_score(y_true, y_pred))),
        "pr_auc": {
            "micro": round_float(micro_pr_auc),
            "macro": round_float(macro_pr_auc),
            "per_label": {label: metrics["pr_auc"] for label, metrics in per_label.items()},
        },
        "per_label": per_label,
    }


def build_sentiment_match_records(
    gold_records: Sequence[Mapping[str, Any]],
    pred_records: Sequence[Mapping[str, Any]],
    sentiment_probabilities: Optional[np.ndarray] = None,
) -> List[SentimentMatchRecord]:
    """Match gold and predicted aspects to evaluate sentiment classification."""
    prediction_lookup = build_prediction_lookup(pred_records)
    matches: List[SentimentMatchRecord] = []

    for row_index, gold_record in enumerate(gold_records):
        review_id = gold_record.get("review_id")
        prediction = prediction_lookup.get(review_id, {})
        gold_sentiments = dict(gold_record.get("aspect_sentiments", {}))
        pred_sentiments = dict(prediction.get("aspect_sentiments", {}))
        predicted_aspects = set(prediction.get("aspects", []))

        for aspect in gold_record.get("aspects", []):
            if aspect not in predicted_aspects:
                continue

            score_vector = None
            if sentiment_probabilities is not None and row_index < len(sentiment_probabilities):
                score_vector = np.asarray(sentiment_probabilities[row_index, ASPECT_TO_INDEX[aspect]], dtype=np.float32)

            matches.append(
                SentimentMatchRecord(
                    review_id=review_id,
                    review_index=row_index,
                    aspect=aspect,
                    gold_sentiment=gold_sentiments[aspect],
                    pred_sentiment=pred_sentiments.get(aspect, "neutral"),
                    score_vector=score_vector,
                )
            )

    return matches


def compute_sentiment_metrics(
    gold_records: Sequence[Mapping[str, Any]],
    pred_records: Sequence[Mapping[str, Any]],
    sentiment_probabilities: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], List[SentimentMatchRecord]]:
    """Compute sentiment metrics on aspect matches."""
    match_records = build_sentiment_match_records(gold_records, pred_records, sentiment_probabilities)
    total_gold_instances = int(sum(len(record.get("aspects", [])) for record in gold_records))
    matched_instances = len(match_records)

    if not match_records:
        empty_metrics = {
            "micro_precision": 0.0,
            "macro_precision": 0.0,
            "weighted_precision": 0.0,
            "micro_recall": 0.0,
            "macro_recall": 0.0,
            "weighted_recall": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "precision": {"micro": 0.0, "macro": 0.0, "weighted": 0.0},
            "recall": {"micro": 0.0, "macro": 0.0, "weighted": 0.0},
            "f1": {"micro": 0.0, "macro": 0.0, "weighted": 0.0},
            "pr_auc": {"micro": None, "macro": None, "per_label": {label: None for label in VALID_SENTIMENTS}},
            "per_label": {
                label: {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                    "pr_auc": None,
                }
                for label in VALID_SENTIMENTS
            },
            "matched_instances": 0,
            "total_gold_instances": total_gold_instances,
            "coverage": 0.0,
        }
        return empty_metrics, match_records

    y_true = np.asarray([SENTIMENT_TO_INDEX[record.gold_sentiment] for record in match_records], dtype=np.int64)
    y_pred = np.asarray([SENTIMENT_TO_INDEX[record.pred_sentiment] for record in match_records], dtype=np.int64)
    score_matrix = None
    if all(record.score_vector is not None for record in match_records):
        score_matrix = np.vstack([np.asarray(record.score_vector, dtype=np.float32) for record in match_records])

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(VALID_SENTIMENTS))),
        average=None,
        zero_division=0,
    )

    pr_auc_per_label: Dict[str, Optional[float]] = {label: None for label in VALID_SENTIMENTS}
    micro_pr_auc = None
    macro_pr_auc = None
    if score_matrix is not None:
        y_true_binarized = label_binarize(y_true, classes=list(range(len(VALID_SENTIMENTS))))
        pr_auc_values: List[float] = []
        for index, sentiment in enumerate(VALID_SENTIMENTS):
            label_pr_auc = safe_average_precision(y_true_binarized[:, index], score_matrix[:, index])
            pr_auc_per_label[sentiment] = round_float(label_pr_auc)
            if label_pr_auc is not None:
                pr_auc_values.append(label_pr_auc)
        if int(y_true_binarized.sum()) > 0:
            micro_pr_auc = float(average_precision_score(y_true_binarized, score_matrix, average="micro"))
        macro_pr_auc = float(np.mean(pr_auc_values)) if pr_auc_values else None

    per_label: Dict[str, Dict[str, Any]] = {}
    for index, sentiment in enumerate(VALID_SENTIMENTS):
        per_label[sentiment] = {
            "precision": round_float(float(precision[index])),
            "recall": round_float(float(recall[index])),
            "f1": round_float(float(f1[index])),
            "support": int(support[index]),
            "pr_auc": pr_auc_per_label[sentiment],
        }

    metrics = {
        "micro_precision": round_float(float(precision_score(y_true, y_pred, average="micro", zero_division=0))),
        "macro_precision": round_float(float(precision_score(y_true, y_pred, average="macro", zero_division=0))),
        "weighted_precision": round_float(float(precision_score(y_true, y_pred, average="weighted", zero_division=0))),
        "micro_recall": round_float(float(recall_score(y_true, y_pred, average="micro", zero_division=0))),
        "macro_recall": round_float(float(recall_score(y_true, y_pred, average="macro", zero_division=0))),
        "weighted_recall": round_float(float(recall_score(y_true, y_pred, average="weighted", zero_division=0))),
        "micro_f1": round_float(float(f1_score(y_true, y_pred, average="micro", zero_division=0))),
        "macro_f1": round_float(float(f1_score(y_true, y_pred, average="macro", zero_division=0))),
        "weighted_f1": round_float(float(f1_score(y_true, y_pred, average="weighted", zero_division=0))),
        "precision": {
            "micro": round_float(float(precision_score(y_true, y_pred, average="micro", zero_division=0))),
            "macro": round_float(float(precision_score(y_true, y_pred, average="macro", zero_division=0))),
            "weighted": round_float(float(precision_score(y_true, y_pred, average="weighted", zero_division=0))),
        },
        "recall": {
            "micro": round_float(float(recall_score(y_true, y_pred, average="micro", zero_division=0))),
            "macro": round_float(float(recall_score(y_true, y_pred, average="macro", zero_division=0))),
            "weighted": round_float(float(recall_score(y_true, y_pred, average="weighted", zero_division=0))),
        },
        "f1": {
            "micro": round_float(float(f1_score(y_true, y_pred, average="micro", zero_division=0))),
            "macro": round_float(float(f1_score(y_true, y_pred, average="macro", zero_division=0))),
            "weighted": round_float(float(f1_score(y_true, y_pred, average="weighted", zero_division=0))),
        },
        "pr_auc": {
            "micro": round_float(micro_pr_auc),
            "macro": round_float(macro_pr_auc),
            "per_label": pr_auc_per_label,
        },
        "per_label": per_label,
        "matched_instances": matched_instances,
        "total_gold_instances": total_gold_instances,
        "coverage": round_float(matched_instances / max(total_gold_instances, 1)),
    }
    return metrics, match_records


def compute_joint_metrics(
    gold_records: Sequence[Mapping[str, Any]],
    pred_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute end-to-end joint aspect-sentiment metrics from final predictions."""
    gold_matrix = joint_matrix_from_records(gold_records)
    pred_matrix = joint_matrix_from_records(pred_records)
    return {
        "micro_f1": round_float(float(f1_score(gold_matrix, pred_matrix, average="micro", zero_division=0))),
        "macro_f1": round_float(float(f1_score(gold_matrix, pred_matrix, average="macro", zero_division=0))),
        "weighted_f1": round_float(float(f1_score(gold_matrix, pred_matrix, average="weighted", zero_division=0))),
        "exact_match_accuracy": round_float(float(accuracy_score(gold_matrix, pred_matrix))),
    }


def compute_aspect_detection_metrics(
    gold_records: Sequence[Mapping[str, Any]],
    pred_records: Sequence[Mapping[str, Any]],
    aspect_probabilities: Optional[np.ndarray] = None,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the requested aspect detection metrics."""
    gold_matrix = aspect_matrix_from_records(gold_records)
    pred_matrix = aspect_matrix_from_records(pred_records)
    return compute_multilabel_metrics(
        y_true=gold_matrix,
        y_pred=pred_matrix,
        y_score=aspect_probabilities,
        label_names=VALID_ASPECTS,
        thresholds=thresholds,
    )


def tune_aspect_thresholds(
    aspect_probabilities: np.ndarray,
    gold_records: Sequence[Mapping[str, Any]],
    search_space: Optional[Sequence[float]] = None,
    granularity: str = "aspect",
    num_passes: int = 2,
) -> Tuple[Dict[str, float], float]:
    """Tune global or per-aspect thresholds for multi-label aspect detection."""
    if granularity not in {"global", "aspect"}:
        raise ValueError("granularity must be 'global' or 'aspect'.")

    gold_matrix = aspect_matrix_from_records(gold_records)
    candidate_thresholds = [round(float(value), 2) for value in (search_space or np.arange(0.2, 0.81, 0.05))]
    best_thresholds = {aspect: 0.5 for aspect in VALID_ASPECTS}

    def score_thresholds(threshold_map: Mapping[str, float]) -> float:
        pred_matrix = aspect_predictions_from_probabilities(aspect_probabilities, threshold_map)
        return float(f1_score(gold_matrix, pred_matrix, average="micro", zero_division=0))

    best_score = score_thresholds(best_thresholds)

    if granularity == "global":
        for threshold in candidate_thresholds:
            candidate = {aspect: float(threshold) for aspect in VALID_ASPECTS}
            score = score_thresholds(candidate)
            if score > best_score:
                best_thresholds = candidate
                best_score = score
        return best_thresholds, best_score

    for _ in range(int(num_passes)):
        improved = False
        for aspect in VALID_ASPECTS:
            current_best_value = best_thresholds[aspect]
            current_best_score = best_score
            for threshold in candidate_thresholds:
                candidate = dict(best_thresholds)
                candidate[aspect] = float(threshold)
                score = score_thresholds(candidate)
                if score > current_best_score:
                    current_best_value = float(threshold)
                    current_best_score = score
            if current_best_score > best_score:
                best_thresholds[aspect] = current_best_value
                best_score = current_best_score
                improved = True
        if not improved:
            break

    return best_thresholds, best_score


def predictions_to_submission(predictions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return the exact sample-submission-compatible schema."""
    return [
        {
            "review_id": prediction.get("review_id"),
            "aspects": list(prediction.get("aspects", [])),
            "aspect_sentiments": dict(prediction.get("aspect_sentiments", {})),
        }
        for prediction in predictions
    ]


def flatten_benchmark_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a nested benchmark result into one CSV-friendly row."""
    aspect_metrics = result["metrics"]["aspect_detection"]
    sentiment_metrics = result["metrics"]["sentiment_classification"]
    joint_metrics = result["metrics"]["joint"]
    thresholds = result.get("thresholds") or {}

    return {
        "model_name": result["model_name"],
        "model_family": result["model_family"],
        "aspect_micro_f1": aspect_metrics["micro_f1"],
        "aspect_macro_f1": aspect_metrics["macro_f1"],
        "aspect_weighted_f1": aspect_metrics["weighted_f1"],
        "aspect_micro_precision": aspect_metrics["micro_precision"],
        "aspect_micro_recall": aspect_metrics["micro_recall"],
        "aspect_pr_auc_micro": aspect_metrics["pr_auc"]["micro"],
        "aspect_pr_auc_macro": aspect_metrics["pr_auc"]["macro"],
        "sentiment_micro_f1": sentiment_metrics["micro_f1"],
        "sentiment_macro_f1": sentiment_metrics["macro_f1"],
        "sentiment_weighted_f1": sentiment_metrics["weighted_f1"],
        "sentiment_micro_precision": sentiment_metrics["micro_precision"],
        "sentiment_micro_recall": sentiment_metrics["micro_recall"],
        "sentiment_pr_auc_micro": sentiment_metrics["pr_auc"]["micro"],
        "sentiment_pr_auc_macro": sentiment_metrics["pr_auc"]["macro"],
        "sentiment_coverage": sentiment_metrics["coverage"],
        "joint_micro_f1": joint_metrics["micro_f1"],
        "joint_exact_match_accuracy": joint_metrics["exact_match_accuracy"],
        "avg_inference_time_ms": result["timing"]["avg_inference_time_ms"],
        "training_time_seconds": result["timing"].get("training_time_seconds"),
        "thresholds": json.dumps(thresholds, ensure_ascii=False, sort_keys=True) if thresholds else "",
    }


def build_model_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a compact summary used by the terminal comparison table."""
    aspect_metrics = result["metrics"]["aspect_detection"]
    sentiment_metrics = result["metrics"]["sentiment_classification"]
    return {
        "Model": result["model_name"],
        "Aspect Micro F1": aspect_metrics["micro_f1"],
        "Aspect Macro F1": aspect_metrics["macro_f1"],
        "Sentiment Micro F1": sentiment_metrics["micro_f1"],
        "Sentiment Coverage": sentiment_metrics["coverage"],
        "Joint Micro F1": result["metrics"]["joint"]["micro_f1"],
        "Inference ms/review": result["timing"]["avg_inference_time_ms"],
    }
