"""
Evaluation Module for Arabic ABSA
==================================
Handles model evaluation with richer multi-label metrics, reporting, and error analysis.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    coverage_error,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import ABDataset, ASPECT_SENTIMENT_LABELS
from preprocess import ArabicPreprocessor
from train import ABSAModel, DEFAULT_MODELS, compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_VALIDATION_PATH = DATASET_ROOT / "DeepX_validation.xlsx"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "model.pt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "validation_metrics.json"

LOW_CONFIDENCE_MARGIN = 0.10
HIGH_CONFIDENCE_MARGIN = 0.30
LONG_SENTENCE_WORD_THRESHOLD = 30

POSSIBLE_DIALECT_MARKERS = (
    "مو",
    "مش",
    "مره",
    "كتير",
    "وايد",
    "لسه",
    "لسا",
    "ليه",
    "ليش",
    "شلون",
    "مافي",
    "هال",
)
POSSIBLE_SARCASM_MARKERS = (
    "هههه",
    "ههه",
    "lol",
    "😂",
    "🤣",
    "😒",
    "🙃",
    "أكيد",
    "طبعاً",
    "يا سلام",
)
AMBIGUITY_MARKERS = (
    "لكن",
    "بس",
    "مع ان",
    "مع إن",
    "رغم",
    "بالرغم",
)


def resolve_model_name(model_name: Optional[str]) -> Optional[str]:
    """Resolve a model alias to a Hugging Face model id."""
    if model_name is None:
        return None
    return DEFAULT_MODELS.get(model_name, model_name)


def resolve_tokenizer_source(checkpoint: Dict, model_path: str, fallback_model_name: Optional[str]) -> str:
    """Resolve a local tokenizer directory first, then fall back to the model id."""
    tokenizer_dir_name = checkpoint.get("tokenizer_dir_name")
    if tokenizer_dir_name:
        tokenizer_dir = Path(model_path).resolve().parent / tokenizer_dir_name
        if tokenizer_dir.exists():
            return str(tokenizer_dir)

    return (
        checkpoint.get("tokenizer_name")
        or checkpoint.get("model_name")
        or resolve_model_name(fallback_model_name)
    )


def resolve_label_names(checkpoint: Dict) -> List[str]:
    """Resolve label names from checkpoint metadata when available."""
    checkpoint_label_names = checkpoint.get("label_names")
    if isinstance(checkpoint_label_names, list) and checkpoint_label_names:
        return [str(label) for label in checkpoint_label_names]
    return list(ASPECT_SENTIMENT_LABELS)


def to_label_to_idx(label_names: Sequence[str]) -> Dict[str, int]:
    """Create a label-to-index mapping from an ordered label list."""
    return {label: index for index, label in enumerate(label_names)}


def label_binarize(labels: np.ndarray) -> np.ndarray:
    """Convert floating label vectors into binary arrays."""
    return (labels >= 0.5).astype(int)


def threshold_predictions(predictions: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the configured probability threshold to get binary predictions."""
    return (predictions >= threshold).astype(int)


def safe_json_value(value: Any) -> Any:
    """Convert numpy values to JSON-safe Python primitives."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [safe_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(safe_json_value(data), handle, indent=2, ensure_ascii=False)


def save_text(text: str, output_path: Path) -> None:
    """Save UTF-8 text output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_validation_metrics(metrics: Dict, output_path: str) -> None:
    """Save validation metrics to JSON file."""
    output_file = Path(output_path)
    save_json(metrics, output_file)
    print(f"Metrics saved to: {output_file}")


def load_model(
    model_path: str,
    model_name: Optional[str],
    num_labels: int,
    device: torch.device,
) -> Tuple[ABSAModel, Dict]:
    """Load a trained model from checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    resolved_model_name = resolve_model_name(model_name) or checkpoint.get("model_name")
    if not resolved_model_name:
        raise ValueError(
            "Base model name could not be resolved. Pass --base_model_name or "
            "retrain with the updated trainer."
        )

    transformer_config = checkpoint.get("transformer_config")
    if transformer_config:
        model = ABSAModel(
            resolved_model_name,
            num_labels,
            load_pretrained=False,
            config_dict=transformer_config,
        )
    else:
        model = ABSAModel(resolved_model_name, num_labels, load_pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def collect_predictions_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str = "Evaluating",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the model and collect sigmoid probabilities plus labels if present."""
    model.eval()
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=description, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            all_predictions.append(torch.sigmoid(logits).cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return predictions, labels


def safe_average(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> Optional[float]:
    """Return a safe average or None when no values are available."""
    if not values:
        return None
    values_array = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.mean(values_array))
    weights_array = np.asarray(weights, dtype=np.float64)
    if float(weights_array.sum()) <= 0:
        return float(np.mean(values_array))
    return float(np.average(values_array, weights=weights_array))


def compute_per_class_ranking_metrics(
    labels_binary: np.ndarray,
    predictions: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute per-class ROC-AUC and average precision without crashing on sparse labels."""
    ranking_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    for index, label_name in enumerate(label_names):
        y_true = labels_binary[:, index]
        y_score = predictions[:, index]
        support = int(y_true.sum())
        roc_auc: Optional[float] = None
        average_precision: Optional[float] = None

        if 0 < support < len(y_true):
            roc_auc = float(roc_auc_score(y_true, y_score))
        if support > 0:
            average_precision = float(average_precision_score(y_true, y_score))

        ranking_metrics[label_name] = {
            "roc_auc": roc_auc,
            "average_precision": average_precision,
        }

    return ranking_metrics


def compute_ranking_summary(
    labels_binary: np.ndarray,
    predictions: np.ndarray,
    per_class_ranking_metrics: Dict[str, Dict[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    """Compute robust summary ranking metrics for multi-label evaluation."""
    total_positives = int(labels_binary.sum())
    total_decisions = int(labels_binary.size)
    supports = labels_binary.sum(axis=0).astype(float)

    roc_values = [
        metric["roc_auc"]
        for metric in per_class_ranking_metrics.values()
        if metric["roc_auc"] is not None
    ]
    roc_weights = [
        float(supports[index])
        for index, metric in enumerate(per_class_ranking_metrics.values())
        if metric["roc_auc"] is not None
    ]

    ap_values = [
        metric["average_precision"]
        for metric in per_class_ranking_metrics.values()
        if metric["average_precision"] is not None
    ]
    ap_weights = [
        float(supports[index])
        for index, metric in enumerate(per_class_ranking_metrics.values())
        if metric["average_precision"] is not None
    ]

    roc_auc_micro: Optional[float] = None
    if 0 < total_positives < total_decisions:
        roc_auc_micro = float(roc_auc_score(labels_binary.ravel(), predictions.ravel()))

    average_precision_micro: Optional[float] = None
    if total_positives > 0:
        average_precision_micro = float(
            average_precision_score(labels_binary.ravel(), predictions.ravel())
        )

    average_precision_macro = safe_average(
        [float(value) for value in ap_values if value is not None]
    )
    average_precision_weighted = safe_average(
        [float(value) for value in ap_values if value is not None],
        ap_weights,
    )

    coverage: Optional[float] = None
    if total_positives > 0:
        try:
            coverage = float(coverage_error(labels_binary, predictions))
        except ValueError:
            coverage = None

    return {
        "roc_auc_micro": roc_auc_micro,
        "roc_auc_macro": safe_average([float(value) for value in roc_values if value is not None]),
        "roc_auc_weighted": safe_average(
            [float(value) for value in roc_values if value is not None],
            roc_weights,
        ),
        "pr_auc_micro": average_precision_micro,
        "pr_auc_macro": average_precision_macro,
        "pr_auc_weighted": average_precision_weighted,
        "average_precision_micro": average_precision_micro,
        "average_precision_macro": average_precision_macro,
        "average_precision_weighted": average_precision_weighted,
        "mean_average_precision": average_precision_macro,
        "coverage_error": coverage,
    }


def compute_overall_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    label_names: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Optional[float]]]]:
    """Compute overall multi-label metrics plus ranking-based metrics."""
    labels_binary = label_binarize(labels)
    pred_binary = threshold_predictions(predictions, threshold)
    base_metrics = compute_metrics(predictions, labels, threshold)

    weighted_precision = precision_score(
        labels_binary,
        pred_binary,
        average="weighted",
        zero_division=0,
    )
    weighted_recall = recall_score(
        labels_binary,
        pred_binary,
        average="weighted",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        labels_binary,
        pred_binary,
        average="weighted",
        zero_division=0,
    )

    per_class_ranking_metrics = compute_per_class_ranking_metrics(
        labels_binary,
        predictions,
        label_names,
    )
    ranking_summary = compute_ranking_summary(
        labels_binary,
        predictions,
        per_class_ranking_metrics,
    )

    metrics: Dict[str, Any] = {
        **base_metrics,
        **ranking_summary,
        "accuracy": float(accuracy_score(labels_binary, pred_binary)),
        "precision": float(base_metrics.get("micro_precision", 0.0)),
        "recall": float(base_metrics.get("micro_recall", 0.0)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "num_samples": int(labels_binary.shape[0]),
        "num_labels": int(labels_binary.shape[1]),
        "label_cardinality": float(labels_binary.sum(axis=1).mean()),
        "predicted_label_cardinality": float(pred_binary.sum(axis=1).mean()),
    }
    metrics["subset_accuracy"] = metrics["accuracy"]
    return metrics, per_class_ranking_metrics


def compute_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
    per_class_ranking_metrics: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-class precision, recall, F1, support, and ranking metrics."""
    labels_binary = label_binarize(labels)
    pred_binary = threshold_predictions(predictions, threshold)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_binary,
        pred_binary,
        average=None,
        zero_division=0,
    )

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for index, label_name in enumerate(label_names):
        label_preds = pred_binary[:, index]
        label_true = labels_binary[:, index]
        tp = int(((label_preds == 1) & (label_true == 1)).sum())
        fp = int(((label_preds == 1) & (label_true == 0)).sum())
        fn = int(((label_preds == 0) & (label_true == 1)).sum())
        ranking_metrics = (per_class_ranking_metrics or {}).get(label_name, {})

        per_class_metrics[label_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "predicted_positives": int(label_preds.sum()),
            "predicted_count": int(label_preds.sum()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "roc_auc": ranking_metrics.get("roc_auc"),
            "average_precision": ranking_metrics.get("average_precision"),
        }

    return per_class_metrics


def compute_per_aspect_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate label metrics into aspect-level performance metrics."""
    labels_binary = label_binarize(labels)
    pred_binary = threshold_predictions(predictions, threshold)

    aspect_to_indices: Dict[str, List[int]] = {}
    for index, label_name in enumerate(label_names):
        if "_" not in label_name:
            continue
        aspect, _sentiment = label_name.rsplit("_", 1)
        aspect_to_indices.setdefault(aspect, []).append(index)

    per_aspect_metrics: Dict[str, Dict[str, Any]] = {}
    for aspect, indices in aspect_to_indices.items():
        aspect_preds = pred_binary[:, indices]
        aspect_true = labels_binary[:, indices]
        tp = int(((aspect_preds == 1) & (aspect_true == 1)).sum())
        fp = int(((aspect_preds == 1) & (aspect_true == 0)).sum())
        fn = int(((aspect_preds == 0) & (aspect_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_aspect_metrics[aspect] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "predicted_count": int(aspect_preds.sum()),
            "true_count": int(aspect_true.sum()),
            "true_positives": tp,
        }

    return per_aspect_metrics


def labels_from_indices(indices: Set[int], label_names: Sequence[str]) -> List[str]:
    """Convert label indices into ordered label names."""
    return [label_names[index] for index in sorted(indices)]


def probabilities_for_sample(probabilities: np.ndarray, label_names: Sequence[str]) -> Dict[str, float]:
    """Create a rounded probability map for a single sample."""
    return {
        label_name: round(float(probabilities[index]), 6)
        for index, label_name in enumerate(label_names)
    }


def get_sample_row(dataframe: pd.DataFrame, index: int) -> Dict[str, Any]:
    """Return a pandas row as a plain dictionary with stable defaults."""
    row = dataframe.iloc[index].to_dict()
    return {str(key): value for key, value in row.items()}


def build_error_record(
    row: Dict[str, Any],
    sample_index: int,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
    true_indices: Set[int],
    predicted_indices: Set[int],
    false_positive_indices: Set[int],
    false_negative_indices: Set[int],
    error_type: str,
) -> Dict[str, Any]:
    """Create a serializable error-analysis record."""
    wrong_indices = false_positive_indices | false_negative_indices
    margins = [abs(float(probabilities[index]) - threshold) for index in wrong_indices]
    review_id = row.get("review_id", sample_index)
    if pd.notna(review_id):
        try:
            review_id = int(review_id)
        except (TypeError, ValueError):
            review_id = str(review_id)
    else:
        review_id = sample_index

    return {
        "review_id": review_id,
        "sample_index": int(sample_index),
        "original_text": str(row.get("review_text", "")) if pd.notna(row.get("review_text")) else "",
        "true_labels": labels_from_indices(true_indices, label_names),
        "predicted_labels": labels_from_indices(predicted_indices, label_names),
        "matched_labels": labels_from_indices(true_indices & predicted_indices, label_names),
        "false_positive_labels": labels_from_indices(false_positive_indices, label_names),
        "false_negative_labels": labels_from_indices(false_negative_indices, label_names),
        "probabilities_per_label": probabilities_for_sample(probabilities, label_names),
        "threshold": float(threshold),
        "error_type": error_type,
        "max_probability": float(np.max(probabilities)),
        "min_wrong_margin_to_threshold": float(min(margins)) if margins else None,
        "max_wrong_margin_to_threshold": float(max(margins)) if margins else None,
        "word_count": len(str(row.get("review_text", "")).split()),
    }


def is_possible_dialect(text: str) -> bool:
    """Heuristic flag for dialectal wording."""
    normalized_text = text.lower()
    return any(marker in normalized_text for marker in POSSIBLE_DIALECT_MARKERS)


def is_possible_sarcasm_or_emphasis(text: str) -> bool:
    """Heuristic flag for sarcasm, laughter, or strong emphasis cues."""
    normalized_text = text.lower()
    if any(marker in normalized_text for marker in POSSIBLE_SARCASM_MARKERS):
        return True
    return bool(re.search(r"[!؟]{2,}", normalized_text))


def is_ambiguous_sentiment(text: str, labels: Sequence[str]) -> bool:
    """Heuristic flag for mixed sentiment or contrastive phrasing."""
    normalized_text = text.lower()
    sentiments = {label.rsplit("_", 1)[-1] for label in labels if "_" in label}
    has_mixed_polarity = {"positive", "negative"}.issubset(sentiments)
    has_contrastive_marker = any(marker in normalized_text for marker in AMBIGUITY_MARKERS)
    return has_mixed_polarity or has_contrastive_marker


def summarize_failure_patterns(
    validation_df: pd.DataFrame,
    failed_indices: Sequence[int],
    labels_binary: np.ndarray,
    pred_binary: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, Any]:
    """Build lightweight heuristic summaries for common ABSA failure patterns."""
    long_sentence_indices: List[int] = []
    dialect_indices: List[int] = []
    sarcasm_indices: List[int] = []
    ambiguous_indices: List[int] = []

    for index in failed_indices:
        row = get_sample_row(validation_df, index)
        text = str(row.get("review_text", "")) if pd.notna(row.get("review_text")) else ""
        combined_labels = labels_from_indices(
            set(np.where(labels_binary[index] == 1)[0]).union(set(np.where(pred_binary[index] == 1)[0])),
            label_names,
        )

        if len(text.split()) >= LONG_SENTENCE_WORD_THRESHOLD:
            long_sentence_indices.append(index)
        if is_possible_dialect(text):
            dialect_indices.append(index)
        if is_possible_sarcasm_or_emphasis(text):
            sarcasm_indices.append(index)
        if is_ambiguous_sentiment(text, combined_labels):
            ambiguous_indices.append(index)

    return {
        "long_sentences": {
            "count": len(long_sentence_indices),
            "example_review_ids": [
                int(validation_df.iloc[index]["review_id"])
                for index in long_sentence_indices[:5]
                if pd.notna(validation_df.iloc[index]["review_id"])
            ],
        },
        "possible_dialect_variation": {
            "count": len(dialect_indices),
            "example_review_ids": [
                int(validation_df.iloc[index]["review_id"])
                for index in dialect_indices[:5]
                if pd.notna(validation_df.iloc[index]["review_id"])
            ],
        },
        "possible_sarcasm_or_emphasis": {
            "count": len(sarcasm_indices),
            "example_review_ids": [
                int(validation_df.iloc[index]["review_id"])
                for index in sarcasm_indices[:5]
                if pd.notna(validation_df.iloc[index]["review_id"])
            ],
        },
        "ambiguous_or_mixed_sentiment": {
            "count": len(ambiguous_indices),
            "example_review_ids": [
                int(validation_df.iloc[index]["review_id"])
                for index in ambiguous_indices[:5]
                if pd.notna(validation_df.iloc[index]["review_id"])
            ],
        },
    }


def build_error_analysis(
    validation_df: pd.DataFrame,
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> Dict[str, Any]:
    """Create a structured error analysis artifact for failed validation examples."""
    labels_binary = label_binarize(labels)
    pred_binary = threshold_predictions(predictions, threshold)

    false_positive_examples: List[Dict[str, Any]] = []
    false_negative_examples: List[Dict[str, Any]] = []
    completely_wrong_examples: List[Dict[str, Any]] = []
    low_confidence_examples: List[Dict[str, Any]] = []
    high_confidence_wrong_examples: List[Dict[str, Any]] = []
    failed_indices: List[int] = []

    for sample_index in range(len(validation_df)):
        row = get_sample_row(validation_df, sample_index)
        true_indices = set(np.where(labels_binary[sample_index] == 1)[0])
        predicted_indices = set(np.where(pred_binary[sample_index] == 1)[0])
        false_positive_indices = predicted_indices - true_indices
        false_negative_indices = true_indices - predicted_indices

        if not false_positive_indices and not false_negative_indices:
            continue

        failed_indices.append(sample_index)
        sample_probabilities = predictions[sample_index]
        wrong_indices = false_positive_indices | false_negative_indices
        wrong_margins = [abs(float(sample_probabilities[index]) - threshold) for index in wrong_indices]

        if false_positive_indices:
            false_positive_examples.append(
                build_error_record(
                    row,
                    sample_index,
                    sample_probabilities,
                    label_names,
                    threshold,
                    true_indices,
                    predicted_indices,
                    false_positive_indices,
                    false_negative_indices,
                    "false_positive",
                )
            )

        if false_negative_indices:
            false_negative_examples.append(
                build_error_record(
                    row,
                    sample_index,
                    sample_probabilities,
                    label_names,
                    threshold,
                    true_indices,
                    predicted_indices,
                    false_positive_indices,
                    false_negative_indices,
                    "false_negative",
                )
            )

        if true_indices.isdisjoint(predicted_indices):
            completely_wrong_examples.append(
                build_error_record(
                    row,
                    sample_index,
                    sample_probabilities,
                    label_names,
                    threshold,
                    true_indices,
                    predicted_indices,
                    false_positive_indices,
                    false_negative_indices,
                    "completely_wrong",
                )
            )

        if wrong_margins and min(wrong_margins) <= LOW_CONFIDENCE_MARGIN:
            low_confidence_examples.append(
                build_error_record(
                    row,
                    sample_index,
                    sample_probabilities,
                    label_names,
                    threshold,
                    true_indices,
                    predicted_indices,
                    false_positive_indices,
                    false_negative_indices,
                    "low_confidence",
                )
            )

        if wrong_margins and max(wrong_margins) >= HIGH_CONFIDENCE_MARGIN:
            high_confidence_wrong_examples.append(
                build_error_record(
                    row,
                    sample_index,
                    sample_probabilities,
                    label_names,
                    threshold,
                    true_indices,
                    predicted_indices,
                    false_positive_indices,
                    false_negative_indices,
                    "high_confidence_wrong",
                )
            )

    false_positive_examples.sort(
        key=lambda item: (
            -len(item["false_positive_labels"]),
            -(item["max_wrong_margin_to_threshold"] or 0.0),
        )
    )
    false_negative_examples.sort(
        key=lambda item: (
            -len(item["false_negative_labels"]),
            -(item["max_wrong_margin_to_threshold"] or 0.0),
        )
    )
    completely_wrong_examples.sort(
        key=lambda item: (-(item["max_wrong_margin_to_threshold"] or 0.0), -item["word_count"])
    )
    low_confidence_examples.sort(
        key=lambda item: item["min_wrong_margin_to_threshold"] or 0.0
    )
    high_confidence_wrong_examples.sort(
        key=lambda item: -(item["max_wrong_margin_to_threshold"] or 0.0)
    )

    return {
        "threshold": float(threshold),
        "summary": {
            "total_samples": int(len(validation_df)),
            "failed_samples": int(len(failed_indices)),
            "false_positive_samples": int(len(false_positive_examples)),
            "false_negative_samples": int(len(false_negative_examples)),
            "completely_wrong_samples": int(len(completely_wrong_examples)),
            "low_confidence_samples": int(len(low_confidence_examples)),
            "high_confidence_wrong_samples": int(len(high_confidence_wrong_examples)),
        },
        "heuristics": summarize_failure_patterns(
            validation_df,
            failed_indices,
            labels_binary,
            pred_binary,
            label_names,
        ),
        "examples": {
            "false_positives": false_positive_examples,
            "false_negatives": false_negative_examples,
            "completely_wrong_samples": completely_wrong_examples,
            "low_confidence_predictions": low_confidence_examples,
            "high_confidence_wrong_predictions": high_confidence_wrong_examples,
        },
    }


def top_and_bottom_classes(
    per_class_metrics: Dict[str, Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]], List[str]]:
    """Return best and worst supported classes plus classes with zero support."""
    supported = [
        (label_name, metrics)
        for label_name, metrics in per_class_metrics.items()
        if metrics["support"] > 0
    ]
    zero_support = [
        label_name
        for label_name, metrics in per_class_metrics.items()
        if metrics["support"] == 0
    ]
    supported.sort(key=lambda item: (-item[1]["f1"], -item[1]["support"], item[0]))
    best = supported[:top_k]
    worst = sorted(
        supported,
        key=lambda item: (item[1]["f1"], item[1]["support"], item[0]),
    )[:top_k]
    return best, worst, zero_support


def format_metric(value: Optional[float]) -> str:
    """Format a metric value for terminal and markdown output."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def render_class_table(rows: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Render a small markdown table for class-level metrics."""
    lines = [
        "| Class | F1 | Precision | Recall | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label_name, metrics in rows:
        lines.append(
            f"| `{label_name}` | {metrics['f1']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['support']} |"
        )
    return "\n".join(lines)


def generate_evaluation_report(
    metrics: Dict[str, Any],
    per_class_metrics: Dict[str, Dict[str, Any]],
    error_analysis: Dict[str, Any],
) -> str:
    """Generate a human-readable markdown evaluation report."""
    best_classes, worst_classes, zero_support_classes = top_and_bottom_classes(per_class_metrics)
    heuristics = error_analysis["heuristics"]
    error_summary = error_analysis["summary"]

    lines = [
        "# Arabic ABSA Evaluation Report",
        "",
        "## Run Summary",
        f"- Model: `{metrics.get('model_name', 'unknown')}`",
        f"- Threshold: `{metrics.get('threshold', 0.5):.2f}`",
        f"- Validation samples: `{metrics.get('num_samples', 0)}`",
        f"- Labels: `{metrics.get('num_labels', 0)}`",
        "",
        "## Overall Metrics",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy (exact-match / subset accuracy) | {format_metric(metrics.get('accuracy'))} |",
        f"| Precision (micro) | {format_metric(metrics.get('micro_precision'))} |",
        f"| Recall (micro) | {format_metric(metrics.get('micro_recall'))} |",
        f"| Micro F1 | {format_metric(metrics.get('micro_f1'))} |",
        f"| Macro F1 | {format_metric(metrics.get('macro_f1'))} |",
        f"| Weighted F1 | {format_metric(metrics.get('weighted_f1'))} |",
        f"| ROC-AUC (micro) | {format_metric(metrics.get('roc_auc_micro'))} |",
        f"| ROC-AUC (macro) | {format_metric(metrics.get('roc_auc_macro'))} |",
        f"| PR-AUC / Average Precision (micro) | {format_metric(metrics.get('average_precision_micro'))} |",
        f"| PR-AUC / Average Precision (macro) | {format_metric(metrics.get('average_precision_macro'))} |",
        f"| Mean Average Precision (mAP) | {format_metric(metrics.get('mean_average_precision'))} |",
        f"| Coverage Error | {format_metric(metrics.get('coverage_error'))} |",
        "",
        "Accuracy alone is misleading for imbalanced ABSA because it is exact-match accuracy here: a sample is counted as correct only when every aspect-sentiment label is right. That makes it harsh on multi-label reviews and insensitive to whether the model is improving on rare classes.",
        "",
        "Micro F1 pools all label decisions together, so it mostly reflects performance on frequent aspect-sentiment pairs. Macro F1 gives each class equal weight and exposes weakness on minority labels. Weighted F1 sits between them by weighting each class by support, which is useful when you want a more deployment-oriented summary without fully hiding imbalance.",
        "",
        "PR-AUC / Average Precision is usually more informative than ROC-AUC for imbalanced ABSA because it focuses on the precision-recall tradeoff over the positive class. ROC-AUC can still look healthy when negatives dominate, while PR-AUC drops quickly when the model surfaces too many false alarms on rare aspects.",
        "",
        "## Best Performing Classes",
        render_class_table(best_classes) if best_classes else "_No supported classes found._",
        "",
        "## Worst Performing Classes",
        render_class_table(worst_classes) if worst_classes else "_No supported classes found._",
        "",
        f"Classes with zero positive support in this split: {', '.join(f'`{label}`' for label in zero_support_classes) if zero_support_classes else 'None'}",
        "",
        "## Error Analysis",
        f"- Failed samples: `{error_summary['failed_samples']}` / `{error_summary['total_samples']}`",
        f"- False-positive samples: `{error_summary['false_positive_samples']}`",
        f"- False-negative samples: `{error_summary['false_negative_samples']}`",
        f"- Completely wrong samples: `{error_summary['completely_wrong_samples']}`",
        f"- Low-confidence failed predictions: `{error_summary['low_confidence_samples']}`",
        f"- High-confidence wrong predictions: `{error_summary['high_confidence_wrong_samples']}`",
        "",
        "## Common Failure Patterns",
        f"- Possible sarcasm or strong-emphasis cues appeared in `{heuristics['possible_sarcasm_or_emphasis']['count']}` failed samples. These often flip literal polarity and make aspect extraction look deceptively easy while sentiment remains wrong.",
        f"- Possible dialect markers appeared in `{heuristics['possible_dialect_variation']['count']}` failed samples. Dialectal phrasing can drift away from pretraining distribution and reduce confidence calibration.",
        f"- Long failed reviews (at least `{LONG_SENTENCE_WORD_THRESHOLD}` words) appeared in `{heuristics['long_sentences']['count']}` samples. Longer texts increase aspect overlap and make label interactions harder.",
        f"- Ambiguous or mixed sentiment signals appeared in `{heuristics['ambiguous_or_mixed_sentiment']['count']}` failed samples. These are typical ABSA pain points when praise and criticism coexist in the same review.",
        "",
        "Detailed sample-level failures are saved in `error_analysis.json` for manual review.",
    ]
    return "\n".join(lines) + "\n"


def print_terminal_summary(
    metrics: Dict[str, Any],
    per_class_metrics: Dict[str, Dict[str, Any]],
    output_paths: Dict[str, Path],
) -> None:
    """Print a concise summary after evaluation finishes."""
    best_classes, worst_classes, _ = top_and_bottom_classes(per_class_metrics, top_k=3)
    best_summary = ", ".join(
        f"{label_name} ({values['f1']:.3f})"
        for label_name, values in best_classes
    ) or "N/A"
    worst_summary = ", ".join(
        f"{label_name} ({values['f1']:.3f})"
        for label_name, values in worst_classes
    ) or "N/A"

    print("\nEvaluation summary")
    print(f"Model: {metrics.get('model_name', 'unknown')}")
    print(f"Threshold: {metrics.get('threshold', 0.5):.2f}")
    print(f"Accuracy (subset): {format_metric(metrics.get('accuracy'))}")
    print(
        "Micro/ Macro/ Weighted F1: "
        f"{format_metric(metrics.get('micro_f1'))} / "
        f"{format_metric(metrics.get('macro_f1'))} / "
        f"{format_metric(metrics.get('weighted_f1'))}"
    )
    print(
        "PR-AUC (micro) / ROC-AUC (micro): "
        f"{format_metric(metrics.get('average_precision_micro'))} / "
        f"{format_metric(metrics.get('roc_auc_micro'))}"
    )
    print(f"Best classes: {best_summary}")
    print(f"Worst classes: {worst_summary}")
    print("Saved artifacts:")
    print(f"  - {output_paths['validation_metrics']}")
    print(f"  - {output_paths['per_class_metrics']}")
    print(f"  - {output_paths['error_analysis']}")
    print(f"  - {output_paths['evaluation_report']}")


def evaluate_detailed(
    model: nn.Module,
    dataloader: DataLoader,
    val_df: Optional[pd.DataFrame] = None,
    label_names: Optional[Sequence[str]] = None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Perform detailed evaluation with richer metrics, reporting, and error analysis."""
    predictions, labels = collect_predictions_and_labels(
        model,
        dataloader,
        device,
        description="Evaluating",
    )
    resolved_label_names = list(label_names or ASPECT_SENTIMENT_LABELS)

    metrics: Dict[str, Any] = {}
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    error_analysis: Dict[str, Any] = {}

    if labels is not None:
        metrics, per_class_ranking_metrics = compute_overall_metrics(
            predictions,
            labels,
            threshold,
            resolved_label_names,
        )
        per_class_metrics = compute_per_class_metrics(
            predictions,
            labels,
            resolved_label_names,
            threshold,
            per_class_ranking_metrics,
        )
        metrics["per_label"] = per_class_metrics
        metrics["per_aspect"] = compute_per_aspect_metrics(
            predictions,
            labels,
            resolved_label_names,
            threshold,
        )
        if val_df is not None:
            error_analysis = build_error_analysis(
                val_df.reset_index(drop=True),
                predictions,
                labels,
                resolved_label_names,
                threshold,
            )

    return {
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "error_analysis": error_analysis,
    }


def tune_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict]:
    """Tune the classification threshold."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {}
    for threshold in thresholds:
        metrics, _ = evaluate_simple(model, dataloader, device, threshold=threshold)
        results[f"{threshold:.2f}"] = {
            "micro_f1": metrics["micro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
        }

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold, metrics in results.items():
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_threshold = float(threshold)

    return best_threshold, results


def evaluate_simple(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> Tuple[Dict, np.ndarray]:
    """Simple evaluation without detailed analysis."""
    predictions, labels = collect_predictions_and_labels(
        model,
        dataloader,
        device,
        description="Evaluating",
    )
    metrics: Dict[str, Any] = {}
    if labels is not None:
        metrics = compute_metrics(predictions, labels, threshold)

    return metrics, predictions


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Arabic ABSA model.")
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_df = pd.read_excel(args.validation_path)
    model, checkpoint = load_model(
        str(args.model_path),
        args.base_model_name,
        len(ASPECT_SENTIMENT_LABELS),
        device,
    )
    label_names = resolve_label_names(checkpoint)

    checkpoint_config = checkpoint.get("config", {})
    max_length = args.max_length or checkpoint_config.get("max_length", 256)
    batch_size = args.batch_size or checkpoint_config.get("batch_size", 16)
    threshold = (
        args.threshold
        if args.threshold is not None
        else checkpoint.get("best_threshold", checkpoint.get("threshold", 0.5))
    )
    tokenizer_name = resolve_tokenizer_source(checkpoint, str(args.model_path), args.base_model_name)

    if not tokenizer_name:
        raise ValueError("Tokenizer name could not be resolved for evaluation.")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = ABDataset(
        val_df,
        tokenizer,
        max_length=max_length,
        preprocessor=ArabicPreprocessor(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    evaluation = evaluate_detailed(
        model,
        dataloader,
        val_df=val_df,
        label_names=label_names,
        device=device,
        threshold=threshold,
    )
    metrics = evaluation["metrics"]
    per_class_metrics = evaluation["per_class_metrics"]
    error_analysis = evaluation["error_analysis"]

    metrics["precision"] = metrics.get("micro_precision", 0.0)
    metrics["recall"] = metrics.get("micro_recall", 0.0)
    metrics["best_threshold"] = threshold
    metrics["threshold"] = threshold
    metrics["model_name"] = checkpoint.get("model_name", tokenizer_name)
    metrics["label_names"] = label_names

    output_dir = args.output_path.parent
    per_class_metrics_path = output_dir / "per_class_metrics.json"
    error_analysis_path = output_dir / "error_analysis.json"
    evaluation_report_path = output_dir / "evaluation_report.md"

    report_markdown = generate_evaluation_report(metrics, per_class_metrics, error_analysis)

    save_validation_metrics(metrics, str(args.output_path))
    save_json(per_class_metrics, per_class_metrics_path)
    save_json(error_analysis, error_analysis_path)
    save_text(report_markdown, evaluation_report_path)

    print_terminal_summary(
        metrics,
        per_class_metrics,
        {
            "validation_metrics": args.output_path,
            "per_class_metrics": per_class_metrics_path,
            "error_analysis": error_analysis_path,
            "evaluation_report": evaluation_report_path,
        },
    )


if __name__ == "__main__":
    main()
