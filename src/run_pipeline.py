"""
End-to-end pipeline: reuse training if available, otherwise train, then run testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dataset import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_UNLABELED_PATH,
    DEFAULT_VALIDATION_PATH,
    load_dataframe,
    resolve_input_path,
)
from predict import DEFAULT_OUTPUT_PATH, generate_submission, predict_dataframe
from train import (
    DEFAULT_CONFIG,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PSEUDO_LABEL_PATH,
    ensure_trained_model,
    str2bool,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the end-to-end pipeline."""
    parser = argparse.ArgumentParser(
        description="Train-if-needed and then run test-time prediction for Arabic ABSA."
    )
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--test_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODELS["xlmr"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
    )
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"])
    parser.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=DEFAULT_CONFIG["early_stopping_patience"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--use_pseudo_labels", type=str2bool, default=False)
    parser.add_argument("--pseudo_label_path", type=Path, default=DEFAULT_PSEUDO_LABEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--predict_batch_size", type=int, default=None)
    parser.add_argument("--predict_max_length", type=int, default=None)
    parser.add_argument("--force_retrain", type=str2bool, default=False)
    parser.add_argument("--allow_checkpoint_fallback", type=str2bool, default=True)
    return parser


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Ensure a checkpoint exists, then run test-time prediction."""
    train_path = resolve_input_path(args.train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH
    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    test_path = resolve_input_path(args.test_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    output_dir = resolve_input_path(args.output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH
    pseudo_label_path = resolve_input_path(args.pseudo_label_path, DEFAULT_PSEUDO_LABEL_PATH)
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None

    training_result = ensure_trained_model(
        train_path=train_path,
        validation_path=validation_path,
        model_name=args.model_name,
        config={
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "dropout": args.dropout,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
        },
        output_dir=output_dir,
        use_pseudo_labels=args.use_pseudo_labels,
        pseudo_label_path=pseudo_label_path,
        force_retrain=args.force_retrain,
        allow_checkpoint_fallback=args.allow_checkpoint_fallback,
    )

    checkpoint_path = Path(training_result["checkpoint_path"])
    test_df = load_dataframe(test_path)
    predictions = predict_dataframe(
        dataframe=test_df,
        model_path=checkpoint_path,
        threshold_path=threshold_path,
        batch_size=args.predict_batch_size,
        max_length=args.predict_max_length,
    )
    submission = generate_submission(predictions, output_path)

    return {
        **training_result,
        "test_path": str(test_path),
        "output_path": str(output_path),
        "num_predictions": len(submission),
    }


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
