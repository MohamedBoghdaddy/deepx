"""
Run the full unlabeled-data enhancement pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from active_learning import build_active_learning_dataframe
from dataset import DEFAULT_UNLABELED_PATH, OUTPUTS_ROOT, load_dataframe, resolve_input_path
from mine_keywords import mine_aspect_keyword_report, print_keyword_summary
from pseudo_label import build_pseudo_label_dataframe
from unlabeled_stress_test import build_stress_report
from unlabeled_utils import (
    DEFAULT_MODEL_DIR,
    build_prediction_analysis_table,
    clean_unlabeled_dataframe,
    prepare_domain_adaptation_corpus,
    save_csv,
    save_json,
)


DEFAULT_PIPELINE_OUTPUT_DIR = OUTPUTS_ROOT / "unlabeled_pipeline"


def run_unlabeled_pipeline(
    model_dir: Path = DEFAULT_MODEL_DIR,
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    confidence_threshold: float = 0.75,
    num_active_samples: int = 200,
    output_dir: Path = DEFAULT_PIPELINE_OUTPUT_DIR,
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the unlabeled-data enhancement workflow end to end."""
    resolved_unlabeled_path = resolve_input_path(unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    raw_unlabeled_df = load_dataframe(resolved_unlabeled_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_output_path = output_dir / "clean_unlabeled.csv"
    pseudo_output_path = output_dir / "pseudo_labeled.csv"
    keyword_output_path = output_dir / "aspect_keyword_report.json"
    active_output_path = output_dir / "active_learning_samples.csv"
    corpus_output_path = output_dir / "domain_adaptation_corpus.txt"
    stress_output_path = output_dir / "unlabeled_stress_report.json"
    summary_output_path = output_dir / "pipeline_summary.json"

    cleaned_df, cleaning_summary = clean_unlabeled_dataframe(raw_unlabeled_df, output_path=clean_output_path)
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
    save_csv(
        pseudo_df,
        pseudo_output_path,
        json_columns=("predicted_aspects", "predicted_sentiments", "aspects", "aspect_sentiments"),
    )

    keyword_report = mine_aspect_keyword_report(
        normalized_reviews=cleaned_df.get("normalized_review_text", []).tolist(),
        original_reviews=cleaned_df.get("review_text", []).tolist(),
    )
    keyword_report["cleaning_summary"] = cleaning_summary
    keyword_report["output_path"] = str(keyword_output_path)
    save_json(keyword_report, keyword_output_path)
    print_keyword_summary(keyword_report)

    active_df = build_active_learning_dataframe(
        prediction_df=prediction_df,
        num_samples=num_active_samples,
    )
    save_csv(active_df, active_output_path)

    prepare_domain_adaptation_corpus(cleaned_df, corpus_output_path)

    stress_report = build_stress_report(
        prediction_df=prediction_df,
        output_path=stress_output_path,
        cleaning_summary=cleaning_summary,
    )
    save_json(stress_report, stress_output_path)

    summary = {
        "input_path": str(resolved_unlabeled_path),
        "model_dir": str(resolve_input_path(model_dir, DEFAULT_MODEL_DIR) or DEFAULT_MODEL_DIR),
        "confidence_threshold": float(confidence_threshold),
        "num_active_samples_requested": int(num_active_samples),
        "clean_rows": int(len(cleaned_df)),
        "pseudo_labeled_samples": int(len(pseudo_df)),
        "active_learning_samples": int(len(active_df)),
        "outputs": {
            "clean_unlabeled": str(clean_output_path),
            "pseudo_labeled": str(pseudo_output_path),
            "aspect_keyword_report": str(keyword_output_path),
            "active_learning_samples": str(active_output_path),
            "domain_adaptation_corpus": str(corpus_output_path),
            "unlabeled_stress_report": str(stress_output_path),
            "pipeline_summary": str(summary_output_path),
        },
        "cleaning_summary": cleaning_summary,
        "stress_test_summary": {
            "invalid_outputs_count": stress_report.get("invalid_outputs_count", 0),
            "low_confidence_count": stress_report.get("low_confidence_count", 0),
            "common_confused_aspects": stress_report.get("common_confused_aspects", []),
        },
    }
    save_json(summary, summary_output_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the full unlabeled-data enhancement pipeline for DeepX ABSA."
    )
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_PIPELINE_OUTPUT_DIR)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--confidence_threshold", type=float, default=0.75)
    parser.add_argument("--num_active_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    summary = run_unlabeled_pipeline(
        model_dir=resolve_input_path(args.model_dir, DEFAULT_MODEL_DIR) or DEFAULT_MODEL_DIR,
        model_path=resolve_input_path(args.model_path, args.model_path) if args.model_path else None,
        unlabeled_path=resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH,
        output_dir=resolve_input_path(args.output_dir, DEFAULT_PIPELINE_OUTPUT_DIR) or DEFAULT_PIPELINE_OUTPUT_DIR,
        threshold_path=resolve_input_path(args.threshold_path, args.threshold_path) if args.threshold_path else None,
        confidence_threshold=float(args.confidence_threshold),
        num_active_samples=int(args.num_active_samples),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
