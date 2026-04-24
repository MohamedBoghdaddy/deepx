"""
Validation evaluation for multilingual Arabic ABSA with competition-ready reports.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
from transformers import AutoTokenizer

from dataset import (
    ABDataset,
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
from predict import (
    DEFAULT_MODEL_PATH,
    collect_probability_records,
    load_thresholds_for_checkpoint,
    load_trained_model,
    resolve_tokenizer_source,
)
from preprocess import ArabicPreprocessor
from rules import extract_rule_features, resolve_label_threshold


DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "validation_metrics.json"
DEFAULT_PER_CLASS_PATH = DEFAULT_OUTPUT_DIR / "per_class_metrics.json"
DEFAULT_ERROR_ANALYSIS_PATH = DEFAULT_OUTPUT_DIR / "error_analysis.json"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "evaluation_report.md"

ERROR_TYPE_ORDER = [
    "false_positive",
    "false_negative",
    "completely_wrong",
    "low_confidence",
    "high_conf_wrong",
]


def save_json(data: Any, output_path: Path) -> None:
    """Write UTF-8 JSON with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_label_names(model_path: Path, checkpoint: Mapping[str, Any]) -> List[str]:
    """Load label names from label_mapping.json, then checkpoint metadata, then defaults."""
    mapping_path = model_path.resolve().parent / "label_mapping.json"
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        label_names = data.get("label_names")
        if isinstance(label_names, list) and label_names:
            return [str(label) for label in label_names]

    checkpoint_label_names = checkpoint.get("label_names")
    if isinstance(checkpoint_label_names, list) and checkpoint_label_names:
        return [str(label) for label in checkpoint_label_names]

    checkpoint_label_to_idx = checkpoint.get("label_to_idx")
    if isinstance(checkpoint_label_to_idx, dict) and checkpoint_label_to_idx:
        return [
            label
            for label, _ in sorted(
                checkpoint_label_to_idx.items(),
                key=lambda item: int(item[1]),
            )
        ]

    return list(ASPECT_SENTIMENT_LABELS)


def align_probability_matrix(
    probability_matrix: np.ndarray,
    label_names: Sequence[str],
) -> np.ndarray:
    """Reorder model probabilities into the canonical dataset label order."""
    if list(label_names) == list(ASPECT_SENTIMENT_LABELS):
        return probability_matrix

    label_to_index = {label: index for index, label in enumerate(label_names)}
    if set(label_to_index) != set(ASPECT_SENTIMENT_LABELS):
        raise ValueError(
            "Checkpoint label names do not match the canonical ABSA label set."
        )

    ordered_indices = [label_to_index[label] for label in ASPECT_SENTIMENT_LABELS]
    return probability_matrix[:, ordered_indices]


def labels_from_dataframe(dataframe: pd.DataFrame) -> np.ndarray:
    """Extract the gold multi-label matrix from a validation DataFrame."""
    mapping = infer_column_mapping(dataframe, require_labels=True)
    vectors = []
    for _, row in dataframe.iterrows():
        aspects = parse_json_column(row[mapping.aspects]) if mapping.aspects else []
        sentiments = parse_sentiment_dict(row[mapping.aspect_sentiments]) if mapping.aspect_sentiments else {}
        vectors.append(create_multi_label_vector(aspects, sentiments))
    return np.vstack(vectors).astype(np.float32)


def collect_validation_probabilities(
    validation_df: pd.DataFrame,
    model_path: Path,
    base_model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the model over validation data and return per-sample probability records."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(model_path, base_model_name, resolved_device)
    tokenizer_source = resolve_tokenizer_source(checkpoint, model_path, base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    checkpoint_config = checkpoint.get("config", {})
    effective_batch_size = int(batch_size or checkpoint_config.get("batch_size", 8))
    effective_max_length = int(max_length or checkpoint_config.get("max_length", 256))
    dataset = ABDataset(
        validation_df,
        tokenizer,
        max_length=effective_max_length,
        preprocessor=ArabicPreprocessor(),
        is_test=False,
    )
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    records = collect_probability_records(model, dataloader, resolved_device)
    return records, checkpoint


def build_threshold_vector(
    label_names: Sequence[str],
    threshold_config: Mapping[str, Any],
) -> np.ndarray:
    """Resolve one threshold per label in the requested order."""
    return np.asarray(
        [resolve_label_threshold(threshold_config, label_name) for label_name in label_names],
        dtype=np.float32,
    )


def binarize_predictions(
    probability_matrix: np.ndarray,
    threshold_vector: np.ndarray,
) -> np.ndarray:
    """Convert sigmoid probabilities into a multi-label prediction matrix."""
    return (probability_matrix >= threshold_vector.reshape(1, -1)).astype(int)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute ROC-AUC when both classes are present."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute average precision when the class has positives."""
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def compute_per_class_metrics(
    probability_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    gold_matrix: np.ndarray,
    label_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-class classification and ranking metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_matrix,
        pred_matrix,
        average=None,
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for index, label_name in enumerate(label_names):
        true_column = gold_matrix[:, index]
        pred_column = pred_matrix[:, index]
        probability_column = probability_matrix[:, index]
        true_positives = int(np.sum((true_column == 1) & (pred_column == 1)))
        false_positives = int(np.sum((true_column == 0) & (pred_column == 1)))
        false_negatives = int(np.sum((true_column == 1) & (pred_column == 0)))
        predicted_positives = int(pred_column.sum())

        per_class[label_name] = {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
            "pred_positives": predicted_positives,
            "predicted_positives": predicted_positives,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "roc_auc": safe_roc_auc(true_column, probability_column),
            "average_precision": safe_average_precision(true_column, probability_column),
        }
    return per_class


def compute_ranking_metrics(
    probability_matrix: np.ndarray,
    gold_matrix: np.ndarray,
    per_class_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute ROC-AUC, PR-AUC, mAP, and coverage error with safe fallbacks."""
    roc_auc_per_class = {
        label_name: metrics.get("roc_auc")
        for label_name, metrics in per_class_metrics.items()
    }
    average_precision_per_class = {
        label_name: metrics.get("average_precision")
        for label_name, metrics in per_class_metrics.items()
    }

    valid_roc_auc = [value for value in roc_auc_per_class.values() if value is not None]
    valid_average_precision = [
        value for value in average_precision_per_class.values()
        if value is not None
    ]

    micro_roc_auc = None
    if len(np.unique(gold_matrix.ravel())) >= 2:
        micro_roc_auc = float(roc_auc_score(gold_matrix.ravel(), probability_matrix.ravel()))

    micro_average_precision = None
    if int(gold_matrix.sum()) > 0:
        micro_average_precision = float(
            average_precision_score(gold_matrix, probability_matrix, average="micro")
        )

    ranking_coverage_error = None
    if np.all(gold_matrix.sum(axis=1) > 0):
        ranking_coverage_error = float(coverage_error(gold_matrix, probability_matrix))

    return {
        "roc_auc": {
            "micro": micro_roc_auc,
            "macro": float(np.mean(valid_roc_auc)) if valid_roc_auc else None,
            "per_class": roc_auc_per_class,
        },
        "pr_auc": {
            "micro": micro_average_precision,
            "macro": float(np.mean(valid_average_precision)) if valid_average_precision else None,
            "per_class": average_precision_per_class,
        },
        "mean_average_precision": float(np.mean(valid_average_precision)) if valid_average_precision else None,
        "coverage_error": ranking_coverage_error,
    }


def compute_overall_metrics(
    probability_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    gold_matrix: np.ndarray,
    threshold_vector: np.ndarray,
    per_class_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute all overall competition metrics."""
    ranking_metrics = compute_ranking_metrics(probability_matrix, gold_matrix, per_class_metrics)
    subset_accuracy = float(accuracy_score(gold_matrix, pred_matrix))
    metrics = {
        "accuracy": subset_accuracy,
        "exact_match_accuracy": subset_accuracy,
        "subset_accuracy": subset_accuracy,
        "micro_precision": float(precision_score(gold_matrix, pred_matrix, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(gold_matrix, pred_matrix, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(gold_matrix, pred_matrix, average="weighted", zero_division=0)),
        "micro_recall": float(recall_score(gold_matrix, pred_matrix, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(gold_matrix, pred_matrix, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(gold_matrix, pred_matrix, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(gold_matrix, pred_matrix, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold_matrix, pred_matrix, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold_matrix, pred_matrix, average="weighted", zero_division=0)),
        "precision": {
            "micro": float(precision_score(gold_matrix, pred_matrix, average="micro", zero_division=0)),
            "macro": float(precision_score(gold_matrix, pred_matrix, average="macro", zero_division=0)),
            "weighted": float(precision_score(gold_matrix, pred_matrix, average="weighted", zero_division=0)),
        },
        "recall": {
            "micro": float(recall_score(gold_matrix, pred_matrix, average="micro", zero_division=0)),
            "macro": float(recall_score(gold_matrix, pred_matrix, average="macro", zero_division=0)),
            "weighted": float(recall_score(gold_matrix, pred_matrix, average="weighted", zero_division=0)),
        },
        "f1": {
            "micro": float(f1_score(gold_matrix, pred_matrix, average="micro", zero_division=0)),
            "macro": float(f1_score(gold_matrix, pred_matrix, average="macro", zero_division=0)),
            "weighted": float(f1_score(gold_matrix, pred_matrix, average="weighted", zero_division=0)),
        },
        "roc_auc": ranking_metrics["roc_auc"],
        "pr_auc": ranking_metrics["pr_auc"],
        "mean_average_precision": ranking_metrics["mean_average_precision"],
        "coverage_error": ranking_metrics["coverage_error"],
        "num_samples": int(gold_matrix.shape[0]),
        "num_labels": int(gold_matrix.shape[1]),
        "avg_predicted_labels": round(float(pred_matrix.sum(axis=1).mean()), 6),
        "avg_true_labels": round(float(gold_matrix.sum(axis=1).mean()), 6),
        "thresholds": {
            "global_threshold": round(float(np.median(threshold_vector)), 6),
            "min_threshold": round(float(np.min(threshold_vector)), 6),
            "max_threshold": round(float(np.max(threshold_vector)), 6),
        },
        "per_label": per_class_metrics,
    }
    return metrics


def active_labels(label_vector: Sequence[int], label_names: Sequence[str]) -> List[str]:
    """Return the active label names from a binary vector."""
    return [
        label_names[index]
        for index, is_active in enumerate(label_vector)
        if int(is_active) == 1
    ]


def build_probability_dict(
    probability_vector: Sequence[float],
    label_names: Sequence[str],
) -> Dict[str, float]:
    """Return a rounded probability dictionary keyed by class name."""
    return {
        label_name: round(float(probability_vector[index]), 6)
        for index, label_name in enumerate(label_names)
    }


def sample_error_margin(
    probability_vector: np.ndarray,
    pred_vector: np.ndarray,
    gold_vector: np.ndarray,
    threshold_vector: np.ndarray,
) -> float:
    """Measure how close the model was to the threshold on wrong labels."""
    error_indices = np.where(pred_vector != gold_vector)[0]
    if error_indices.size == 0:
        return 1.0
    margins = np.abs(probability_vector[error_indices] - threshold_vector[error_indices])
    return float(np.mean(margins))


def classify_error_types(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    probability_vector: np.ndarray,
    threshold_vector: np.ndarray,
    label_names: Sequence[str],
) -> List[str]:
    """Assign one or more error categories to a failed prediction."""
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    shared = true_set & pred_set
    missing = true_set - pred_set
    extra = pred_set - true_set
    probability_dict = build_probability_dict(probability_vector, label_names)

    categories: List[str] = []
    if extra and not missing:
        categories.append("false_positive")
    if missing and not extra:
        categories.append("false_negative")
    if (missing or extra) and not shared:
        categories.append("completely_wrong")

    margin = sample_error_margin(
        probability_vector=probability_vector,
        pred_vector=np.asarray([1 if label in pred_set else 0 for label in label_names], dtype=int),
        gold_vector=np.asarray([1 if label in true_set else 0 for label in label_names], dtype=int),
        threshold_vector=threshold_vector,
    )
    if margin <= 0.08:
        categories.append("low_confidence")

    confident_wrong_positive = any(
        probability_dict[label] >= max(0.85, float(threshold_vector[label_names.index(label)]) + 0.15)
        for label in extra
    )
    confident_wrong_negative = any(
        probability_dict[label] <= min(0.15, float(threshold_vector[label_names.index(label)]) - 0.15)
        for label in missing
    )
    if confident_wrong_positive or confident_wrong_negative:
        categories.append("high_conf_wrong")

    if not categories:
        categories.append("completely_wrong")
    return categories


def build_error_analysis(
    validation_df: pd.DataFrame,
    probability_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    gold_matrix: np.ndarray,
    threshold_vector: np.ndarray,
    label_names: Sequence[str],
    max_examples_per_type: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """Build a curated error-analysis set with at least several samples per error type."""
    mapping = infer_column_mapping(validation_df, require_labels=True)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    summary_counts: Counter = Counter()

    for row_index, (_, row) in enumerate(validation_df.iterrows()):
        pred_vector = pred_matrix[row_index]
        gold_vector = gold_matrix[row_index]
        if np.array_equal(pred_vector, gold_vector):
            continue

        true_labels = active_labels(gold_vector, label_names)
        pred_labels = active_labels(pred_vector, label_names)
        categories = classify_error_types(
            true_labels=true_labels,
            pred_labels=pred_labels,
            probability_vector=probability_matrix[row_index],
            threshold_vector=threshold_vector,
            label_names=label_names,
        )

        base_entry = {
            "text": str(row[mapping.review_text]),
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "probs": build_probability_dict(probability_matrix[row_index], label_names),
        }

        for category in categories:
            buckets[category].append({**base_entry, "error_type": category})
            summary_counts[category] += 1

    selected_examples: List[Dict[str, Any]] = []
    selected_counts: Counter = Counter()
    for error_type in ERROR_TYPE_ORDER:
        for example in buckets.get(error_type, [])[:max_examples_per_type]:
            selected_examples.append(example)
            selected_counts[error_type] += 1

    return selected_examples, dict(summary_counts), dict(selected_counts)


def infer_failure_patterns(error_examples: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Infer common failure patterns from the curated error examples."""
    counts: Counter = Counter()
    for example in error_examples:
        text = str(example.get("text", ""))
        features = extract_rule_features(text)
        word_count = len(features.tokens)

        if features.sarcasm_candidate:
            counts["sarcasm"] += 1
        if features.sentiment_conflict:
            counts["ambiguous_sentiment"] += 1
        if word_count >= 30:
            counts["long_sentences"] += 1
        if re.search(r"[23789]", text) or re.search(r"\b(?:msh|mesh|7|3|2|5|8)\w*", text.lower()):
            counts["dialect_variation"] += 1
        if re.search(r"[A-Za-z]", text) and re.search(r"[\u0600-\u06FF]", text):
            counts["mixed_language"] += 1
    return dict(counts)


def select_best_and_worst_classes(
    per_class_metrics: Mapping[str, Mapping[str, Any]],
    top_k: int = 5,
) -> Tuple[List[Tuple[str, Mapping[str, Any]]], List[Tuple[str, Mapping[str, Any]]], List[str]]:
    """Return best/worst supported classes plus zero-support labels."""
    supported = [
        (label_name, metrics)
        for label_name, metrics in per_class_metrics.items()
        if int(metrics.get("support", 0)) > 0
    ]
    zero_support = [
        label_name
        for label_name, metrics in per_class_metrics.items()
        if int(metrics.get("support", 0)) == 0
    ]

    ranked = sorted(
        supported,
        key=lambda item: (
            float(item[1].get("f1", 0.0)),
            float(item[1].get("precision", 0.0)),
            float(item[1].get("recall", 0.0)),
        ),
        reverse=True,
    )
    worst_ranked = sorted(
        supported,
        key=lambda item: (
            float(item[1].get("f1", 0.0)),
            float(item[1].get("support", 0)),
        ),
    )
    return ranked[:top_k], worst_ranked[:top_k], zero_support


def format_class_rows(rows: Iterable[Tuple[str, Mapping[str, Any]]]) -> str:
    """Render Markdown rows for best/worst class tables."""
    formatted_rows = []
    for label_name, metrics in rows:
        formatted_rows.append(
            "| `{label}` | {f1:.4f} | {precision:.4f} | {recall:.4f} | {support} |".format(
                label=label_name,
                f1=float(metrics.get("f1", 0.0)),
                precision=float(metrics.get("precision", 0.0)),
                recall=float(metrics.get("recall", 0.0)),
                support=int(metrics.get("support", 0)),
            )
        )
    return "\n".join(formatted_rows) if formatted_rows else "| None | 0.0000 | 0.0000 | 0.0000 | 0 |"


def render_evaluation_report(
    metrics: Mapping[str, Any],
    per_class_metrics: Mapping[str, Mapping[str, Any]],
    error_examples: Sequence[Mapping[str, Any]],
    error_summary: Mapping[str, int],
    selected_error_counts: Mapping[str, int],
    failure_patterns: Mapping[str, int],
    model_name: str,
    output_dir: Path,
) -> str:
    """Generate a human-readable Markdown report."""
    best_classes, worst_classes, zero_support = select_best_and_worst_classes(per_class_metrics)
    report_lines = [
        "# Arabic ABSA Evaluation Report",
        "",
        "## Run Summary",
        f"- Model: `{model_name}`",
        f"- Threshold range: `{metrics['thresholds']['min_threshold']:.2f}` to `{metrics['thresholds']['max_threshold']:.2f}`",
        f"- Validation samples: `{metrics['num_samples']}`",
        f"- Labels: `{metrics['num_labels']}`",
        "",
        "## Overall Metrics",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy (exact-match / subset accuracy) | {metrics['accuracy']:.4f} |",
        f"| Precision (micro) | {metrics['micro_precision']:.4f} |",
        f"| Precision (macro) | {metrics['macro_precision']:.4f} |",
        f"| Precision (weighted) | {metrics['weighted_precision']:.4f} |",
        f"| Recall (micro) | {metrics['micro_recall']:.4f} |",
        f"| Recall (macro) | {metrics['macro_recall']:.4f} |",
        f"| Recall (weighted) | {metrics['weighted_recall']:.4f} |",
        f"| F1 (micro) | {metrics['micro_f1']:.4f} |",
        f"| F1 (macro) | {metrics['macro_f1']:.4f} |",
        f"| F1 (weighted) | {metrics['weighted_f1']:.4f} |",
        f"| ROC-AUC (micro) | {(metrics['roc_auc']['micro'] or 0.0):.4f} |",
        f"| ROC-AUC (macro) | {(metrics['roc_auc']['macro'] or 0.0):.4f} |",
        f"| PR-AUC / Average Precision (micro) | {(metrics['pr_auc']['micro'] or 0.0):.4f} |",
        f"| PR-AUC / Average Precision (macro) | {(metrics['pr_auc']['macro'] or 0.0):.4f} |",
        f"| Mean Average Precision (mAP) | {(metrics['mean_average_precision'] or 0.0):.4f} |",
        f"| Coverage Error | {(metrics['coverage_error'] or 0.0):.4f} |",
        "",
        (
            "Accuracy alone is misleading for imbalanced ABSA because exact-match accuracy only counts a "
            "review as correct when every aspect-sentiment label is correct. Multi-label reviews are "
            "therefore punished harshly, even when most aspects are right."
        ),
        "",
        (
            "Micro F1 pools all label decisions together and is driven by frequent classes. Macro F1 gives "
            "each class equal weight and exposes weakness on rare labels. Weighted F1 sits between them by "
            "respecting class support without letting majority classes dominate completely."
        ),
        "",
        (
            "PR-AUC is more informative than ROC-AUC for imbalanced ABSA because it focuses directly on the "
            "precision-recall tradeoff for the positive class. ROC-AUC can still look strong when negatives "
            "dominate, even if the model produces too many false alarms on rare aspects."
        ),
        "",
        "## Best Performing Classes",
        "| Class | F1 | Precision | Recall | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
        format_class_rows(best_classes),
        "",
        "## Worst Performing Classes",
        "| Class | F1 | Precision | Recall | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
        format_class_rows(worst_classes),
        "",
    ]

    if zero_support:
        report_lines.extend(
            [
                "Classes with zero positive support in this split: "
                + ", ".join(f"`{label}`" for label in zero_support),
                "",
            ]
        )

    report_lines.extend(
        [
            "## Error Analysis",
            f"- Curated failed examples saved: `{len(error_examples)}`",
            f"- False-positive candidates: `{error_summary.get('false_positive', 0)}`",
            f"- False-negative candidates: `{error_summary.get('false_negative', 0)}`",
            f"- Completely wrong candidates: `{error_summary.get('completely_wrong', 0)}`",
            f"- Low-confidence candidates: `{error_summary.get('low_confidence', 0)}`",
            f"- High-confidence wrong candidates: `{error_summary.get('high_conf_wrong', 0)}`",
            "",
            "## Common Failure Patterns",
        ]
    )

    if failure_patterns.get("sarcasm"):
        report_lines.append(
            f"- Possible sarcasm or irony cues appeared in `{failure_patterns['sarcasm']}` curated failures."
        )
    if failure_patterns.get("dialect_variation"):
        report_lines.append(
            f"- Dialect or Franco-Arabic variation appeared in `{failure_patterns['dialect_variation']}` curated failures."
        )
    if failure_patterns.get("mixed_language"):
        report_lines.append(
            f"- Mixed-script or mixed-language text appeared in `{failure_patterns['mixed_language']}` curated failures."
        )
    if failure_patterns.get("long_sentences"):
        report_lines.append(
            f"- Long reviews appeared in `{failure_patterns['long_sentences']}` curated failures."
        )
    if failure_patterns.get("ambiguous_sentiment"):
        report_lines.append(
            f"- Ambiguous or mixed sentiment cues appeared in `{failure_patterns['ambiguous_sentiment']}` curated failures."
        )
    if not failure_patterns:
        report_lines.append("- No strong automatic failure pattern stood out in the curated sample.")

    report_lines.extend(
        [
            "",
            "Curated examples per saved error type:",
            ", ".join(
                f"`{error_type}`={selected_error_counts.get(error_type, 0)}"
                for error_type in ERROR_TYPE_ORDER
            ),
            "",
            f"Detailed sample-level failures are saved in `{(output_dir / 'error_analysis.json').name}` for manual review.",
        ]
    )

    return "\n".join(report_lines).strip() + "\n"


def round_nested_numbers(value: Any) -> Any:
    """Round floats recursively for cleaner JSON output."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: round_nested_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested_numbers(item) for item in value]
    return value


def print_summary(metrics: Mapping[str, Any], output_dir: Path) -> None:
    """Print a concise terminal summary once evaluation completes."""
    summary = {
        "accuracy": round(float(metrics["accuracy"]), 6),
        "micro_f1": round(float(metrics["micro_f1"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "weighted_f1": round(float(metrics["weighted_f1"]), 6),
        "roc_auc_micro": round(float(metrics["roc_auc"]["micro"] or 0.0), 6),
        "pr_auc_micro": round(float(metrics["pr_auc"]["micro"] or 0.0), 6),
        "mean_average_precision": round(float(metrics["mean_average_precision"] or 0.0), 6),
        "coverage_error": round(float(metrics["coverage_error"] or 0.0), 6),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Evaluate multilingual Arabic ABSA on validation data.")
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    validation_input = args.test_file or args.validation_path
    validation_path = resolve_input_path(validation_input, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    model_path = resolve_input_path(args.model_path, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None
    output_dir = resolve_input_path(args.output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH

    if output_path.name != "validation_metrics.json":
        metrics_path = output_path
        output_dir = output_path.parent
    else:
        metrics_path = output_dir / "validation_metrics.json"

    per_class_path = output_dir / DEFAULT_PER_CLASS_PATH.name
    error_analysis_path = output_dir / DEFAULT_ERROR_ANALYSIS_PATH.name
    report_path = output_dir / DEFAULT_REPORT_PATH.name

    validation_df = load_dataframe(validation_path)
    records, checkpoint = collect_validation_probabilities(
        validation_df=validation_df,
        model_path=model_path,
        base_model_name=args.base_model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if not records:
        raise ValueError("No validation records were produced during evaluation.")

    checkpoint_label_names = load_label_names(model_path, checkpoint)
    probability_matrix = np.vstack(
        [np.asarray(record["label_probabilities"], dtype=np.float32) for record in records]
    )
    probability_matrix = align_probability_matrix(probability_matrix, checkpoint_label_names)
    gold_matrix = labels_from_dataframe(validation_df).astype(int)

    threshold_config = load_thresholds_for_checkpoint(checkpoint, threshold_path)
    threshold_vector = build_threshold_vector(ASPECT_SENTIMENT_LABELS, threshold_config)
    pred_matrix = binarize_predictions(probability_matrix, threshold_vector)

    per_class_metrics = compute_per_class_metrics(
        probability_matrix=probability_matrix,
        pred_matrix=pred_matrix,
        gold_matrix=gold_matrix,
        label_names=ASPECT_SENTIMENT_LABELS,
    )
    metrics = compute_overall_metrics(
        probability_matrix=probability_matrix,
        pred_matrix=pred_matrix,
        gold_matrix=gold_matrix,
        threshold_vector=threshold_vector,
        per_class_metrics=per_class_metrics,
    )

    error_examples, error_summary, selected_error_counts = build_error_analysis(
        validation_df=validation_df,
        probability_matrix=probability_matrix,
        pred_matrix=pred_matrix,
        gold_matrix=gold_matrix,
        threshold_vector=threshold_vector,
        label_names=ASPECT_SENTIMENT_LABELS,
    )
    failure_patterns = infer_failure_patterns(error_examples)

    metrics["error_analysis_summary"] = error_summary
    metrics["output_files"] = {
        "validation_metrics": str(metrics_path),
        "per_class_metrics": str(per_class_path),
        "error_analysis": str(error_analysis_path),
        "evaluation_report": str(report_path),
    }

    report_text = render_evaluation_report(
        metrics=round_nested_numbers(metrics),
        per_class_metrics=round_nested_numbers(per_class_metrics),
        error_examples=error_examples,
        error_summary=error_summary,
        selected_error_counts=selected_error_counts,
        failure_patterns=failure_patterns,
        model_name=str(checkpoint.get("model_name") or args.base_model_name or "unknown"),
        output_dir=output_dir,
    )

    save_json(round_nested_numbers(metrics), metrics_path)
    save_json(round_nested_numbers(per_class_metrics), per_class_path)
    save_json(round_nested_numbers(error_examples), error_analysis_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print_summary(metrics, output_dir)


if __name__ == "__main__":
    main()
