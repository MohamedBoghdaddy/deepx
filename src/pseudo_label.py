"""
Pseudo-label generation for semi-supervised Arabic ABSA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from dataset import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_UNLABELED_PATH,
    DEFAULT_VALIDATION_PATH,
    OUTPUTS_ROOT,
    infer_column_mapping,
    load_dataframe,
    resolve_input_path,
)
from predict import DEFAULT_MODEL_PATH, predict_dataframe
from train import merge_pseudo_labels, train_model


DEFAULT_PSEUDO_LABEL_OUTPUT = OUTPUTS_ROOT / "pseudo_labeled.json"
DEFAULT_STATS_OUTPUT = OUTPUTS_ROOT / "pseudo_label_stats.json"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Generate pseudo labels for Arabic ABSA.")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--min_confidence", type=float, default=0.9)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_PSEUDO_LABEL_OUTPUT)
    parser.add_argument("--stats_path", type=Path, default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--run_retrain", action="store_true")
    parser.add_argument("--retrain_output_dir", type=Path, default=OUTPUTS_ROOT / "pseudo_retrained")
    parser.add_argument("--epochs", type=int, default=2)
    return parser


def should_keep_prediction(prediction: Dict[str, Any], min_confidence: float) -> tuple[bool, str]:
    """Filter pseudo labels conservatively."""
    confidence = float(prediction.get("confidence", 0.0))
    explanation = prediction.get("explanation", {})
    if confidence < min_confidence:
        return False, "low_confidence"
    if explanation.get("sarcasm_candidate"):
        return False, "sarcasm_candidate"
    if explanation.get("sentiment_conflict"):
        return False, "sentiment_conflict"
    if prediction.get("aspects") == ["none"] and confidence < max(min_confidence, 0.97):
        return False, "low_confidence_none"
    return True, "kept"


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    train_path = resolve_input_path(args.train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH
    unlabeled_path = resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    model_path = resolve_input_path(args.model_path, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None
    output_path = resolve_input_path(args.output_path, DEFAULT_PSEUDO_LABEL_OUTPUT) or DEFAULT_PSEUDO_LABEL_OUTPUT
    stats_path = resolve_input_path(args.stats_path, DEFAULT_STATS_OUTPUT) or DEFAULT_STATS_OUTPUT
    retrain_output_dir = resolve_input_path(args.retrain_output_dir, OUTPUTS_ROOT / "pseudo_retrained") or (OUTPUTS_ROOT / "pseudo_retrained")

    unlabeled_df = load_dataframe(unlabeled_path)
    predictions = predict_dataframe(
        dataframe=unlabeled_df,
        model_path=model_path,
        base_model_name=args.base_model_name,
        threshold_path=threshold_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_explanations=True,
    )

    mapping = infer_column_mapping(unlabeled_df, require_labels=False)
    original_rows = {
        row[mapping.review_id]: row.to_dict()
        for _, row in unlabeled_df.iterrows()
    }

    kept_rows: List[Dict[str, Any]] = []
    rejection_stats: Dict[str, int] = {}
    for prediction in predictions:
        keep, reason = should_keep_prediction(prediction, args.min_confidence)
        rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
        if not keep:
            continue

        row = dict(original_rows.get(prediction["review_id"], {}))
        row["review_id"] = prediction["review_id"]
        row["review_text"] = prediction.get("review_text", row.get(mapping.review_text, ""))
        row["aspects"] = prediction["aspects"]
        row["aspect_sentiments"] = prediction["aspect_sentiments"]
        row["confidence"] = round(float(prediction["confidence"]), 6)
        row["source"] = "pseudo_label"
        row["model_name"] = args.base_model_name or "checkpoint"
        kept_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(kept_rows, handle, ensure_ascii=False, indent=2, default=str)

    average_confidence = round(
        float(sum(row["confidence"] for row in kept_rows) / max(len(kept_rows), 1)),
        6,
    )
    stats = {
        "total_unlabeled_reviews": int(len(unlabeled_df)),
        "kept_pseudo_labels": int(len(kept_rows)),
        "average_confidence": average_confidence,
        "filter_stats": rejection_stats,
        "output_path": str(output_path),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    if args.run_retrain:
        train_df = load_dataframe(train_path)
        pseudo_df = pd.DataFrame(kept_rows)
        merged_train_df = merge_pseudo_labels(train_df, pseudo_df)
        train_model(
            train_df=merged_train_df,
            val_df=load_dataframe(validation_path),
            model_name=args.base_model_name or "UBC-NLP/MARBERTv2",
            config={"num_epochs": args.epochs},
            output_dir=retrain_output_dir,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
