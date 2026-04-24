"""Error analysis and confusion matrix generation for the benchmark pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dataset import VALID_ASPECTS, VALID_SENTIMENTS
from benchmark.metrics import SentimentMatchRecord

try:  # pragma: no cover - optional plotting dependency in some environments
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def save_json(data: Any, output_path: Path) -> None:
    """Persist a JSON artifact with UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def render_confusion_matrix_png(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> Optional[str]:
    """Render a confusion matrix heatmap when matplotlib is available."""
    if plt is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=list(labels),
        yticklabels=list(labels),
        ylabel="Gold",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                int(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def build_per_aspect_error_analysis(
    gold_records: Sequence[Mapping[str, Any]],
    pred_records: Sequence[Mapping[str, Any]],
    max_examples_per_bucket: int = 10,
) -> Dict[str, Any]:
    """Generate aspect-level false-positive, false-negative, and wrong-sentiment analysis."""
    prediction_lookup = {record.get("review_id"): dict(record) for record in pred_records}
    analysis: Dict[str, Any] = {}

    for aspect in VALID_ASPECTS:
        aspect_summary: MutableMapping[str, Any] = {
            "support": 0,
            "predicted_count": 0,
            "true_positive_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "wrong_sentiment_count": 0,
            "false_positives": [],
            "false_negatives": [],
            "wrong_sentiments": [],
        }

        for gold_record in gold_records:
            review_id = gold_record.get("review_id")
            prediction = prediction_lookup.get(review_id, {})
            gold_aspects = set(gold_record.get("aspects", []))
            pred_aspects = set(prediction.get("aspects", []))
            gold_sentiments = dict(gold_record.get("aspect_sentiments", {}))
            pred_sentiments = dict(prediction.get("aspect_sentiments", {}))

            if aspect in gold_aspects:
                aspect_summary["support"] += 1
            if aspect in pred_aspects:
                aspect_summary["predicted_count"] += 1

            example_payload = {
                "review_id": review_id,
                "review_text": gold_record.get("review_text", ""),
                "gold_aspects": sorted(gold_aspects),
                "pred_aspects": sorted(pred_aspects),
                "gold_sentiments": gold_sentiments,
                "pred_sentiments": pred_sentiments,
            }

            if aspect in gold_aspects and aspect in pred_aspects:
                aspect_summary["true_positive_count"] += 1
                if gold_sentiments.get(aspect) != pred_sentiments.get(aspect):
                    aspect_summary["wrong_sentiment_count"] += 1
                    if len(aspect_summary["wrong_sentiments"]) < max_examples_per_bucket:
                        aspect_summary["wrong_sentiments"].append(example_payload)
            elif aspect not in gold_aspects and aspect in pred_aspects:
                aspect_summary["false_positive_count"] += 1
                if len(aspect_summary["false_positives"]) < max_examples_per_bucket:
                    aspect_summary["false_positives"].append(example_payload)
            elif aspect in gold_aspects and aspect not in pred_aspects:
                aspect_summary["false_negative_count"] += 1
                if len(aspect_summary["false_negatives"]) < max_examples_per_bucket:
                    aspect_summary["false_negatives"].append(example_payload)

        analysis[aspect] = dict(aspect_summary)

    return analysis


def save_sentiment_confusion_matrices(
    match_records: Sequence[SentimentMatchRecord],
    output_dir: Path,
) -> Dict[str, Any]:
    """Save overall and per-aspect sentiment confusion matrices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_csv_path = output_dir / "sentiment_confusion_matrix.csv"
    overall_json_path = output_dir / "sentiment_confusion_matrix.json"
    overall_png_path = output_dir / "sentiment_confusion_matrix.png"
    by_aspect_dir = output_dir / "sentiment_confusion_by_aspect"
    by_aspect_dir.mkdir(parents=True, exist_ok=True)

    if match_records:
        y_true = [record.gold_sentiment for record in match_records]
        y_pred = [record.pred_sentiment for record in match_records]
        overall_matrix = confusion_matrix(y_true, y_pred, labels=VALID_SENTIMENTS)
    else:
        overall_matrix = np.zeros((len(VALID_SENTIMENTS), len(VALID_SENTIMENTS)), dtype=np.int64)

    overall_frame = pd.DataFrame(
        overall_matrix,
        index=[f"gold_{label}" for label in VALID_SENTIMENTS],
        columns=[f"pred_{label}" for label in VALID_SENTIMENTS],
    )
    overall_frame.to_csv(overall_csv_path, encoding="utf-8")
    save_json(
        {
            "labels": VALID_SENTIMENTS,
            "matrix": overall_matrix.tolist(),
        },
        overall_json_path,
    )

    by_aspect_paths: Dict[str, Dict[str, str]] = {}
    by_aspect_payload: Dict[str, Any] = {}
    for aspect in VALID_ASPECTS:
        aspect_matches = [record for record in match_records if record.aspect == aspect]
        if aspect_matches:
            aspect_true = [record.gold_sentiment for record in aspect_matches]
            aspect_pred = [record.pred_sentiment for record in aspect_matches]
            aspect_matrix = confusion_matrix(aspect_true, aspect_pred, labels=VALID_SENTIMENTS)
        else:
            aspect_matrix = np.zeros((len(VALID_SENTIMENTS), len(VALID_SENTIMENTS)), dtype=np.int64)

        aspect_csv_path = by_aspect_dir / f"{aspect}_confusion_matrix.csv"
        aspect_json_path = by_aspect_dir / f"{aspect}_confusion_matrix.json"
        aspect_png_path = by_aspect_dir / f"{aspect}_confusion_matrix.png"

        pd.DataFrame(
            aspect_matrix,
            index=[f"gold_{label}" for label in VALID_SENTIMENTS],
            columns=[f"pred_{label}" for label in VALID_SENTIMENTS],
        ).to_csv(aspect_csv_path, encoding="utf-8")

        payload = {
            "labels": VALID_SENTIMENTS,
            "matrix": aspect_matrix.tolist(),
            "matched_instances": len(aspect_matches),
        }
        save_json(payload, aspect_json_path)
        by_aspect_payload[aspect] = payload
        by_aspect_paths[aspect] = {
            "csv": str(aspect_csv_path),
            "json": str(aspect_json_path),
        }

        png_path = render_confusion_matrix_png(
            aspect_matrix,
            VALID_SENTIMENTS,
            f"Sentiment Confusion Matrix - {aspect}",
            aspect_png_path,
        )
        if png_path:
            by_aspect_paths[aspect]["png"] = png_path

    overall_png = render_confusion_matrix_png(
        overall_matrix,
        VALID_SENTIMENTS,
        "Sentiment Confusion Matrix - Overall",
        overall_png_path,
    )

    return {
        "overall": {
            "csv": str(overall_csv_path),
            "json": str(overall_json_path),
            "png": overall_png,
            "matrix": overall_matrix.tolist(),
        },
        "by_aspect": by_aspect_paths,
        "by_aspect_payload": by_aspect_payload,
    }
