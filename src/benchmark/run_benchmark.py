"""Run the full Arabic ABSA benchmark across all requested models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from benchmark.bilstm_baseline import BiLSTMConfig, run_bilstm_benchmark
from benchmark.evaluate_model import evaluate_prediction_bundle
from benchmark.qwen_zero_shot import QwenZeroShotConfig, run_qwen_zero_shot_benchmark
from benchmark.report_generator import (
    build_csv_rows,
    print_terminal_table,
    render_benchmark_report,
    save_aggregate_json,
    save_report,
)
from benchmark.transformer_runner import TransformerTrainingConfig, run_transformer_benchmark
from dataset import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_UNLABELED_PATH,
    DEFAULT_VALIDATION_PATH,
    OUTPUTS_ROOT,
    load_dataframe,
    resolve_input_path,
)


DEFAULT_MODELS = ["arabert", "marbert", "qwen", "bilstm"]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the benchmark entrypoint."""
    parser = argparse.ArgumentParser(description="Run the full Arabic ABSA benchmark pipeline.")
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--output_dir", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--device", default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_granularity", choices=["global", "aspect"], default="aspect")
    parser.add_argument("--threshold_tuning_passes", type=int, default=2)

    parser.add_argument("--bilstm_batch_size", type=int, default=32)
    parser.add_argument("--bilstm_max_length", type=int, default=128)
    parser.add_argument("--bilstm_epochs", type=int, default=8)
    parser.add_argument("--bilstm_learning_rate", type=float, default=1e-3)
    parser.add_argument("--bilstm_hidden_size", type=int, default=256)
    parser.add_argument("--bilstm_dropout", type=float, default=0.3)
    parser.add_argument("--bilstm_embedding_path", default=None)

    parser.add_argument("--qwen_provider", choices=["auto", "ollama", "transformers"], default="auto")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=200)
    parser.add_argument("--qwen_temperature", type=float, default=0.0)
    return parser


def resolve_device(device_arg: str | None) -> torch.device:
    """Resolve the torch device for the benchmark run."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    """CLI entrypoint for the benchmark package."""
    args = build_arg_parser().parse_args()
    train_path = resolve_input_path(args.train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH
    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    _ = resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    output_dir = resolve_input_path(args.output_dir, OUTPUTS_ROOT) or OUTPUTS_ROOT
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataframe(train_path)
    validation_df = load_dataframe(validation_path)
    device = resolve_device(args.device)

    results: List[dict] = []
    for model_name in args.models:
        model_output_dir = benchmark_dir / model_name
        if model_name in {"arabert", "marbert"}:
            bundle = run_transformer_benchmark(
                model_key=model_name,
                train_df=train_df,
                validation_df=validation_df,
                output_dir=model_output_dir,
                config=TransformerTrainingConfig(
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    num_epochs=args.epochs,
                    dropout=args.dropout,
                    early_stopping_patience=args.early_stopping_patience,
                    seed=args.seed,
                    threshold_granularity=args.threshold_granularity,
                    threshold_tuning_passes=args.threshold_tuning_passes,
                ),
                device=device,
            )
        elif model_name == "bilstm":
            bundle = run_bilstm_benchmark(
                train_df=train_df,
                validation_df=validation_df,
                output_dir=model_output_dir,
                config=BiLSTMConfig(
                    batch_size=args.bilstm_batch_size,
                    max_length=args.bilstm_max_length,
                    num_epochs=args.bilstm_epochs,
                    learning_rate=args.bilstm_learning_rate,
                    hidden_size=args.bilstm_hidden_size,
                    dropout=args.bilstm_dropout,
                    seed=args.seed,
                    threshold_granularity=args.threshold_granularity,
                    threshold_tuning_passes=args.threshold_tuning_passes,
                    embedding_path=args.bilstm_embedding_path,
                ),
                device=device,
            )
        else:
            bundle = run_qwen_zero_shot_benchmark(
                validation_df=validation_df,
                output_dir=model_output_dir,
                config=QwenZeroShotConfig(
                    provider=args.qwen_provider,
                    max_new_tokens=args.qwen_max_new_tokens,
                    temperature=args.qwen_temperature,
                ),
            )

        results.append(
            evaluate_prediction_bundle(
                validation_df=validation_df,
                bundle=bundle,
                output_dir=model_output_dir,
            )
        )

    csv_rows = build_csv_rows(results)
    benchmark_csv_path = output_dir / "benchmark_results.csv"
    benchmark_json_path = output_dir / "benchmark_results.json"
    report_path = output_dir / "BENCHMARK_REPORT.md"

    pd.DataFrame(csv_rows).to_csv(benchmark_csv_path, index=False, encoding="utf-8")
    save_aggregate_json(results, benchmark_json_path)
    report_text = render_benchmark_report(results, output_dir)
    save_report(report_text, report_path)
    print_terminal_table(results)


if __name__ == "__main__":
    main()
