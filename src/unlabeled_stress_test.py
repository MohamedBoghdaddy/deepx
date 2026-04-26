"""
Stress-test a trained ABSA model on unlabeled DeepX reviews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from dataset import DEFAULT_UNLABELED_PATH, VALID_ASPECTS, VALID_SENTIMENTS, resolve_input_path
from unlabeled_utils import (
    DEFAULT_CLEAN_UNLABELED_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_STRESS_REPORT_PATH,
    build_prediction_analysis_table,
    examples_from_dataframe,
    load_and_clean_unlabeled_data,
    save_json,
    top_confused_aspects,
)


LOW_CONFIDENCE_THRESHOLD = 0.60


def validate_prediction_row(row: pd.Series) -> List[str]:
    """Validate one prediction against the competition schema."""
    issues: List[str] = []
    predicted_aspects = row.get("predicted_aspects")
    predicted_sentiments = row.get("predicted_sentiments")

    if not isinstance(predicted_aspects, list) or not predicted_aspects:
        issues.append("missing_or_empty_aspects")
        return issues

    if not isinstance(predicted_sentiments, dict):
        issues.append("missing_aspect_sentiment_map")
        return issues

    invalid_aspects = [aspect for aspect in predicted_aspects if aspect not in VALID_ASPECTS]
    if invalid_aspects:
        issues.append("invalid_aspects")

    for aspect in predicted_aspects:
        if aspect not in predicted_sentiments:
            issues.append("missing_sentiment_for_aspect")
            break
        if predicted_sentiments[aspect] not in VALID_SENTIMENTS:
            issues.append("invalid_sentiment_value")
            break

    if "none" in predicted_aspects and predicted_aspects != ["none"]:
        issues.append("none_with_other_aspects")

    if predicted_aspects == ["none"] and predicted_sentiments.get("none") != "neutral":
        issues.append("none_not_neutral")

    if not bool(row.get("prediction_json_valid", False)):
        issues.append("json_serialization_failure")

    return sorted(set(issues))


def build_stress_report(
    prediction_df: pd.DataFrame,
    output_path: Path,
    cleaning_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a robustness report over unlabeled predictions."""
    if prediction_df.empty:
        return {
            "total_tested_samples": 0,
            "invalid_outputs_count": 0,
            "output_path": str(output_path),
            "cleaning_summary": cleaning_summary,
            "examples_of_failure_cases": [],
            "recommendations": ["No unlabeled samples were available after cleaning."],
        }

    validation_issues = prediction_df.apply(validate_prediction_row, axis=1)
    invalid_mask = validation_issues.apply(bool)
    low_conf_mask = prediction_df["confidence_score"] < LOW_CONFIDENCE_THRESHOLD
    short_mask = prediction_df["token_count"] <= 3
    multilingual_mask = prediction_df["is_multilingual"].astype(bool)
    franco_mask = prediction_df["is_franco_arabic"].astype(bool)
    emoji_mask = prediction_df["contains_emoji"].astype(bool)
    contradiction_mask = prediction_df["weak_signal_alignment"] == "contradictory"
    general_confusion_mask = prediction_df["general_ambiance_app_confusion"].astype(bool)
    none_mask = prediction_df["predicted_aspects"].apply(lambda value: value == ["none"])

    failure_cases: List[Dict[str, Any]] = []
    for title, mask in (
        ("invalid_output", invalid_mask),
        ("low_confidence", low_conf_mask),
        ("multilingual_low_confidence", multilingual_mask & low_conf_mask),
        ("franco_low_confidence", franco_mask & low_conf_mask),
        ("short_review_edge_cases", short_mask),
        ("emoji_edge_cases", emoji_mask & low_conf_mask),
        ("star_rating_text_contradictions", contradiction_mask),
        ("general_ambiance_app_confusion", general_confusion_mask),
    ):
        for example in examples_from_dataframe(prediction_df, mask, limit=3):
            failure_cases.append({"failure_type": title, **example})

    recommendations: List[str] = []
    if int(invalid_mask.sum()) > 0:
        recommendations.append("Tighten post-processing schema checks before submission generation.")
    if float(low_conf_mask.mean()) > 0.20:
        recommendations.append("Review threshold tuning and calibration because low-confidence predictions are common.")
    if int((multilingual_mask & low_conf_mask).sum()) > 25:
        recommendations.append("Add more multilingual and code-switched augmentation or pseudo-label review.")
    if int((franco_mask & low_conf_mask).sum()) > 10:
        recommendations.append("Expand the Franco-Arabic lexicon and manually label a small Franco subset.")
    if int(general_confusion_mask.sum()) > 20:
        recommendations.append("Strengthen rules or training examples separating general, ambiance, and app_experience.")
    if int((short_mask & low_conf_mask).sum()) > 15:
        recommendations.append("Add more short-review examples and lightweight rules for terse sentiment cues.")
    if int((emoji_mask & low_conf_mask).sum()) > 10:
        recommendations.append("Augment emoji-heavy reviews and verify emoji polarity handling.")
    if int(contradiction_mask.sum()) > 20:
        recommendations.append("Inspect star-rating contradictions to find mixed price-vs-general or service-vs-overall cases.")
    if not recommendations:
        recommendations.append("Robustness looks reasonable; keep monitoring low-confidence multilingual edge cases.")

    return {
        "total_tested_samples": int(len(prediction_df)),
        "invalid_outputs_count": int(invalid_mask.sum()),
        "multilingual_samples_count": int(multilingual_mask.sum()),
        "franco_samples_count": int(franco_mask.sum()),
        "short_reviews_count": int(short_mask.sum()),
        "emoji_reviews_count": int(emoji_mask.sum()),
        "low_confidence_count": int(low_conf_mask.sum()),
        "none_predictions_count": int(none_mask.sum()),
        "star_text_contradictions_count": int(contradiction_mask.sum()),
        "common_confused_aspects": top_confused_aspects(prediction_df),
        "examples_of_failure_cases": failure_cases[:24],
        "recommendations": recommendations,
        "cleaning_summary": cleaning_summary,
        "output_path": str(output_path),
    }


def run_unlabeled_stress_test(
    model_dir: Path = DEFAULT_MODEL_DIR,
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    output_path: Path = DEFAULT_STRESS_REPORT_PATH,
    clean_output_path: Optional[Path] = DEFAULT_CLEAN_UNLABELED_PATH,
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the model on cleaned unlabeled data and generate a robustness report."""
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
    report = build_stress_report(
        prediction_df=prediction_df,
        output_path=output_path,
        cleaning_summary=cleaning_summary,
    )
    save_json(report, output_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a robustness stress test on the unlabeled DeepX reviews."
    )
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_STRESS_REPORT_PATH)
    parser.add_argument("--clean_output_path", type=Path, default=DEFAULT_CLEAN_UNLABELED_PATH)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    report = run_unlabeled_stress_test(
        model_dir=resolve_input_path(args.model_dir, DEFAULT_MODEL_DIR) or DEFAULT_MODEL_DIR,
        model_path=resolve_input_path(args.model_path, args.model_path) if args.model_path else None,
        unlabeled_path=resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH,
        output_path=resolve_input_path(args.output_path, DEFAULT_STRESS_REPORT_PATH) or DEFAULT_STRESS_REPORT_PATH,
        clean_output_path=resolve_input_path(args.clean_output_path, DEFAULT_CLEAN_UNLABELED_PATH) or DEFAULT_CLEAN_UNLABELED_PATH,
        threshold_path=resolve_input_path(args.threshold_path, args.threshold_path) if args.threshold_path else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
