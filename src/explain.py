"""
Explainability export for Arabic ABSA predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dataset import DEFAULT_VALIDATION_PATH, OUTPUTS_ROOT, load_dataframe, resolve_input_path
from predict import DEFAULT_MODEL_PATH, predict_dataframe


DEFAULT_OUTPUT_PATH = OUTPUTS_ROOT / "demo_predictions.json"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Generate explainable prediction demos for Arabic ABSA.")
    parser.add_argument("--input_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    input_path = resolve_input_path(args.input_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    model_path = resolve_input_path(args.model_path, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH

    dataframe = load_dataframe(input_path)
    predictions = predict_dataframe(
        dataframe=dataframe.head(args.limit).copy(),
        model_path=model_path,
        base_model_name=args.base_model_name,
        threshold_path=threshold_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_explanations=True,
    )

    demo_payload: List[Dict[str, Any]] = []
    for prediction in predictions:
        explanation = prediction.get("explanation", {})
        demo_payload.append(
            {
                "review_id": prediction["review_id"],
                "review_text": prediction.get("review_text", ""),
                "predicted_aspects": prediction["aspects"],
                "aspect_sentiments": prediction["aspect_sentiments"],
                "confidence": prediction.get("confidence", 0.0),
                "aspect_probabilities": explanation.get("aspect_probabilities", {}),
                "sentiment_probabilities": explanation.get("sentiment_probabilities", {}),
                "matched_keywords": explanation.get("matched_keywords", {}),
                "matched_sentiment_terms": explanation.get("matched_sentiment_terms", {}),
                "emoji_tokens": explanation.get("emoji_tokens", []),
                "emoji_names": explanation.get("emoji_names", []),
                "rule_decisions": explanation.get("rule_decisions", []),
                "sarcasm_candidate": explanation.get("sarcasm_candidate", False),
                "sentiment_conflict": explanation.get("sentiment_conflict", False),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(demo_payload, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "num_demo_predictions": len(demo_payload),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
