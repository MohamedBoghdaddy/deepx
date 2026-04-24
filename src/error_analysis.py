"""
Structured validation error analysis for Arabic ABSA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from dataset import (
    DEFAULT_VALIDATION_PATH,
    OUTPUTS_ROOT,
    infer_column_mapping,
    load_dataframe,
    parse_json_column,
    parse_sentiment_dict,
    resolve_input_path,
)
from ensemble_predict import average_probability_records, collect_model_probability_records, discover_model_paths
from predict import DEFAULT_MODEL_PATH, load_thresholds_for_checkpoint, postprocess_probability_records, predict_dataframe


DEFAULT_OUTPUT_PATH = OUTPUTS_ROOT / "error_analysis.csv"


def build_gold_lookup(validation_df: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
    """Create a lookup of gold annotations by review id."""
    mapping = infer_column_mapping(validation_df, require_labels=True)
    gold_lookup: Dict[Any, Dict[str, Any]] = {}
    for _, row in validation_df.iterrows():
        review_id = row[mapping.review_id]
        gold_lookup[review_id] = {
            "review_text": str(row[mapping.review_text]),
            "aspects": parse_json_column(row[mapping.aspects]) if mapping.aspects else [],
            "aspect_sentiments": parse_sentiment_dict(row[mapping.aspect_sentiments]) if mapping.aspect_sentiments else {},
        }
    return gold_lookup


def build_predictions(
    validation_df: pd.DataFrame,
    model_paths: List[Path],
    base_model_name: str | None,
    threshold_path: Path | None,
    batch_size: int | None,
    max_length: int | None,
) -> List[Dict[str, Any]]:
    """Generate validation predictions from one model or an ensemble."""
    if len(model_paths) <= 1:
        return predict_dataframe(
            dataframe=validation_df,
            model_path=model_paths[0],
            base_model_name=base_model_name,
            threshold_path=threshold_path,
            batch_size=batch_size,
            max_length=max_length,
            include_explanations=True,
        )

    all_records = []
    last_checkpoint: Dict[str, Any] = {}
    for model_path in model_paths:
        probability_records, checkpoint = collect_model_probability_records(
            validation_df,
            model_path,
            base_model_name,
            batch_size,
            max_length,
        )
        all_records.append(probability_records)
        last_checkpoint = checkpoint

    averaged_records = average_probability_records(all_records)
    threshold_config = load_thresholds_for_checkpoint(last_checkpoint, threshold_path)
    return postprocess_probability_records(
        averaged_records,
        threshold_config=threshold_config,
        include_explanations=True,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate validation error analysis for Arabic ABSA.")
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_paths", nargs="*", type=Path, default=None)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH
    validation_df = load_dataframe(validation_path)

    model_paths = [resolve_input_path(path) for path in args.model_paths] if args.model_paths else discover_model_paths(OUTPUTS_ROOT)
    model_paths = [path for path in model_paths if path and path.exists()]
    if not model_paths:
        fallback = resolve_input_path(DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
        if fallback.exists():
            model_paths = [fallback]
        else:
            raise FileNotFoundError("No model checkpoint found for error analysis.")

    gold_lookup = build_gold_lookup(validation_df)
    predictions = build_predictions(
        validation_df,
        model_paths,
        args.base_model_name,
        threshold_path,
        args.batch_size,
        args.max_length,
    )

    rows: List[Dict[str, Any]] = []
    missed_aspects = Counter()
    wrong_sentiments = Counter()

    for prediction in predictions:
        review_id = prediction["review_id"]
        gold = gold_lookup[review_id]
        true_aspects = set(gold["aspects"])
        pred_aspects = set(prediction["aspects"])
        missing_aspects = sorted(true_aspects - pred_aspects)
        extra_aspects = sorted(pred_aspects - true_aspects)

        sentiment_errors = {}
        for aspect in sorted(true_aspects & pred_aspects):
            true_sentiment = gold["aspect_sentiments"].get(aspect, "neutral")
            pred_sentiment = prediction["aspect_sentiments"].get(aspect, "neutral")
            if true_sentiment != pred_sentiment:
                sentiment_errors[aspect] = {
                    "true": true_sentiment,
                    "predicted": pred_sentiment,
                }
                wrong_sentiments[f"{aspect}: {true_sentiment}->{pred_sentiment}"] += 1

        for aspect in missing_aspects:
            missed_aspects[aspect] += 1

        if not missing_aspects and not extra_aspects and not sentiment_errors:
            continue

        rows.append(
            {
                "review_id": review_id,
                "text": gold["review_text"],
                "true_aspects": json.dumps(gold["aspects"], ensure_ascii=False),
                "predicted_aspects": json.dumps(prediction["aspects"], ensure_ascii=False),
                "missing_aspects": json.dumps(missing_aspects, ensure_ascii=False),
                "extra_aspects": json.dumps(extra_aspects, ensure_ascii=False),
                "true_aspect_sentiments": json.dumps(gold["aspect_sentiments"], ensure_ascii=False),
                "predicted_aspect_sentiments": json.dumps(prediction["aspect_sentiments"], ensure_ascii=False),
                "sentiment_errors": json.dumps(sentiment_errors, ensure_ascii=False),
                "confidence": float(prediction.get("confidence", 0.0)),
            }
        )

    error_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Most missed aspects:")
    for aspect, count in missed_aspects.most_common(5):
        print(f"  {aspect}: {count}")

    print("Most wrong sentiments:")
    for sentiment_case, count in wrong_sentiments.most_common(5):
        print(f"  {sentiment_case}: {count}")

    print(
        json.dumps(
            {
                "num_error_rows": len(error_df),
                "output_path": str(output_path),
                "ensemble_size": len(model_paths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
