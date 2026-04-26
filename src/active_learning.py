"""
Select high-value unlabeled reviews for manual annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dataset import DEFAULT_UNLABELED_PATH, resolve_input_path
from unlabeled_utils import (
    DEFAULT_ACTIVE_LEARNING_PATH,
    DEFAULT_CLEAN_UNLABELED_PATH,
    DEFAULT_MODEL_DIR,
    build_prediction_analysis_table,
    load_and_clean_unlabeled_data,
    save_csv,
)


def evaluate_active_learning_priority(row: pd.Series) -> Tuple[float, List[str], str]:
    """Score one unlabeled prediction for manual-review value."""
    confidence = float(row.get("confidence_score", 0.0))
    reasons: List[str] = []
    score = 0.0

    if confidence < 0.45:
        reasons.append("low_confidence")
        score += 5.0
    elif confidence < 0.60:
        reasons.append("medium_confidence")
        score += 3.5
    elif confidence < 0.75:
        reasons.append("borderline_confidence")
        score += 2.0

    if bool(row.get("mixed_sentiment")):
        reasons.append("mixed_sentiment")
        score += 3.0

    if int(row.get("num_predicted_aspects", 0)) >= 2:
        reasons.append("multiple_aspects")
        score += 2.5

    if bool(row.get("short_meaningful")):
        reasons.append("short_meaningful")
        score += 2.0

    if bool(row.get("is_multilingual")):
        reasons.append("multilingual")
        score += 2.0

    if bool(row.get("is_franco_arabic")):
        reasons.append("franco_arabic")
        score += 2.5

    if str(row.get("weak_signal_alignment", "")) == "contradictory":
        reasons.append("star_rating_text_contradiction")
        score += 3.0

    if bool(row.get("general_ambiance_app_confusion")):
        reasons.append("general_ambiance_app_confusion")
        score += 2.5

    confused_pair = row.get("confused_aspect_pair")
    if isinstance(confused_pair, dict) and float(confused_pair.get("margin", 1.0)) < 0.08:
        reasons.append("high_aspect_uncertainty")
        score += 2.0

    predicted_aspects = row.get("predicted_aspects") or []
    if predicted_aspects == ["none"] and confidence < 0.70:
        reasons.append("unclear_none_prediction")
        score += 2.0

    if bool(row.get("contains_emoji")) and confidence < 0.70:
        reasons.append("emoji_sentiment_edge_case")
        score += 1.0

    if score >= 7.0:
        priority = "high"
    elif score >= 4.0:
        priority = "medium"
    else:
        priority = "low"

    return score, reasons, priority


def build_active_learning_dataframe(
    prediction_df: pd.DataFrame,
    num_samples: int,
) -> pd.DataFrame:
    """Rank and select unlabeled samples for manual review."""
    if prediction_df.empty:
        return prediction_df.copy()

    scored_rows: List[Dict[str, Any]] = []
    for _, row in prediction_df.iterrows():
        priority_score, reasons, priority = evaluate_active_learning_priority(row)
        predicted_labels = row.get("predicted_labels") or []
        scored_rows.append(
            {
                "review_id": row.get("review_id"),
                "review_text": row.get("review_text"),
                "star_rating": row.get("star_rating"),
                "business_category": row.get("business_category"),
                "platform": row.get("platform"),
                "predicted_labels": " | ".join(predicted_labels),
                "confidence_score": round(float(row.get("confidence_score", 0.0)), 6),
                "uncertainty_reason": "; ".join(reasons) if reasons else "low_information",
                "suggested_manual_review_priority": priority,
                "priority_score": round(float(priority_score), 6),
                "weak_sentiment": row.get("weak_sentiment"),
                "weak_signal_alignment": row.get("weak_signal_alignment"),
                "is_multilingual": bool(row.get("is_multilingual")),
                "is_franco_arabic": bool(row.get("is_franco_arabic")),
                "contains_emoji": bool(row.get("contains_emoji")),
            }
        )

    ranked_df = pd.DataFrame(scored_rows)
    ranked_df = ranked_df.sort_values(
        by=["priority_score", "confidence_score", "is_franco_arabic", "is_multilingual"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    return ranked_df.head(int(num_samples)).reset_index(drop=True)


def run_active_learning_selection(
    model_dir: Path = DEFAULT_MODEL_DIR,
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    output_path: Path = DEFAULT_ACTIVE_LEARNING_PATH,
    num_samples: int = 200,
    clean_output_path: Optional[Path] = DEFAULT_CLEAN_UNLABELED_PATH,
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Run inference on unlabeled data and export active-learning candidates."""
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
    active_df = build_active_learning_dataframe(prediction_df, num_samples=num_samples)
    save_csv(active_df, output_path)

    summary = {
        "input_path": str(resolve_input_path(unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH),
        "output_path": str(output_path),
        "clean_output_path": str(clean_output_path) if clean_output_path is not None else None,
        "requested_samples": int(num_samples),
        "selected_samples": int(len(active_df)),
        "cleaning_summary": cleaning_summary,
        "priority_distribution": (
            active_df["suggested_manual_review_priority"].value_counts().to_dict()
            if not active_df.empty
            else {}
        ),
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Select high-value unlabeled samples for future manual ABSA labeling."
    )
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_ACTIVE_LEARNING_PATH)
    parser.add_argument("--clean_output_path", type=Path, default=DEFAULT_CLEAN_UNLABELED_PATH)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    summary = run_active_learning_selection(
        model_dir=resolve_input_path(args.model_dir, DEFAULT_MODEL_DIR) or DEFAULT_MODEL_DIR,
        model_path=resolve_input_path(args.model_path, args.model_path) if args.model_path else None,
        unlabeled_path=resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH,
        output_path=resolve_input_path(args.output_path, DEFAULT_ACTIVE_LEARNING_PATH) or DEFAULT_ACTIVE_LEARNING_PATH,
        num_samples=args.num_samples,
        clean_output_path=resolve_input_path(args.clean_output_path, DEFAULT_CLEAN_UNLABELED_PATH) or DEFAULT_CLEAN_UNLABELED_PATH,
        threshold_path=resolve_input_path(args.threshold_path, args.threshold_path) if args.threshold_path else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
