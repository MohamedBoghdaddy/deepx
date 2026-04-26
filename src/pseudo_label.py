"""
High-confidence pseudo-label generation for unlabeled Arabic ABSA reviews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from dataset import DEFAULT_UNLABELED_PATH, OUTPUTS_ROOT, resolve_input_path
from unlabeled_utils import (
    DEFAULT_CLEAN_UNLABELED_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_PSEUDO_LABELED_PATH,
    build_prediction_analysis_table,
    load_and_clean_unlabeled_data,
    save_csv,
)


DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def build_pseudo_label_dataframe(
    prediction_df: pd.DataFrame,
    confidence_threshold: float,
) -> pd.DataFrame:
    """Keep only high-confidence model predictions for optional downstream training."""
    if prediction_df.empty:
        return prediction_df.copy()

    kept_df = prediction_df.loc[
        (prediction_df["confidence_score"] >= float(confidence_threshold))
        & prediction_df["prediction_json_valid"].astype(bool)
    ].copy()

    if kept_df.empty:
        return kept_df

    kept_df["pseudo_label_source"] = "model_high_confidence"
    kept_df["aspects"] = kept_df["predicted_aspects"]
    kept_df["aspect_sentiments"] = kept_df["predicted_sentiments"]

    return kept_df[
        [
            "review_id",
            "review_text",
            "predicted_aspects",
            "predicted_sentiments",
            "confidence_score",
            "star_rating",
            "business_category",
            "platform",
            "weak_sentiment",
            "weak_signal_alignment",
            "pseudo_label_source",
            "aspects",
            "aspect_sentiments",
        ]
    ].reset_index(drop=True)


def run_pseudo_labeling(
    model_dir: Path = DEFAULT_MODEL_DIR,
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    output_path: Path = DEFAULT_PSEUDO_LABELED_PATH,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    clean_output_path: Optional[Path] = DEFAULT_CLEAN_UNLABELED_PATH,
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Clean unlabeled data, run inference, and export pseudo labels."""
    cleaned_df, cleaning_summary = load_and_clean_unlabeled_data(
        unlabeled_path=unlabeled_path,
        output_path=clean_output_path,
    )
    prediction_df = build_prediction_analysis_table(
        cleaned_df=cleaned_df,
        model_dir=model_dir,
        model_path=model_path,
        threshold_path=threshold_path,
        batch_size=batch_size,
        max_length=max_length,
    )
    pseudo_df = build_pseudo_label_dataframe(
        prediction_df=prediction_df,
        confidence_threshold=confidence_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(
        pseudo_df,
        output_path,
        json_columns=("predicted_aspects", "predicted_sentiments", "aspects", "aspect_sentiments"),
    )

    average_confidence = (
        round(float(pseudo_df["confidence_score"].mean()), 6)
        if not pseudo_df.empty
        else 0.0
    )
    summary = {
        "input_path": str(resolve_input_path(unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH),
        "clean_output_path": str(clean_output_path) if clean_output_path is not None else None,
        "output_path": str(output_path),
        "confidence_threshold": float(confidence_threshold),
        "total_clean_reviews": int(len(cleaned_df)),
        "pseudo_labeled_samples": int(len(pseudo_df)),
        "average_confidence": average_confidence,
        "cleaning_summary": cleaning_summary,
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate high-confidence pseudo labels from the unlabeled DeepX reviews."
    )
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_PSEUDO_LABELED_PATH)
    parser.add_argument("--clean_output_path", type=Path, default=DEFAULT_CLEAN_UNLABELED_PATH)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    output_path = resolve_input_path(args.output_path, DEFAULT_PSEUDO_LABELED_PATH) or DEFAULT_PSEUDO_LABELED_PATH
    clean_output_path = (
        resolve_input_path(args.clean_output_path, DEFAULT_CLEAN_UNLABELED_PATH)
        if args.clean_output_path is not None
        else None
    )
    summary = run_pseudo_labeling(
        model_dir=resolve_input_path(args.model_dir, DEFAULT_MODEL_DIR) or DEFAULT_MODEL_DIR,
        model_path=resolve_input_path(args.model_path, args.model_path) if args.model_path else None,
        unlabeled_path=resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH,
        output_path=output_path,
        confidence_threshold=float(args.confidence_threshold),
        clean_output_path=clean_output_path,
        threshold_path=resolve_input_path(args.threshold_path, args.threshold_path) if args.threshold_path else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
