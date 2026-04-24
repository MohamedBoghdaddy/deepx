"""
Ensemble prediction for Arabic ABSA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ABDataset, DEFAULT_UNLABELED_PATH, OUTPUTS_ROOT, load_dataframe, resolve_input_path
from predict import (
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    collect_probability_records,
    generate_submission,
    load_thresholds_for_checkpoint,
    load_trained_model,
    postprocess_probability_records,
    resolve_tokenizer_source,
)
from preprocess import ArabicPreprocessor


def discover_model_paths(root: Path) -> List[Path]:
    """Discover available model checkpoints recursively under outputs/."""
    discovered = sorted(path.resolve() for path in root.rglob("model.pt"))
    unique_paths: List[Path] = []
    seen = set()
    for path in discovered:
        if str(path) not in seen:
            unique_paths.append(path)
            seen.add(str(path))
    return unique_paths


def collect_model_probability_records(
    dataframe: pd.DataFrame,
    model_path: Path,
    base_model_name: str | None,
    batch_size: int | None,
    max_length: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one model and collect probability records for the entire dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(model_path, base_model_name, device)
    tokenizer_source = resolve_tokenizer_source(checkpoint, model_path, base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    preprocessor = ArabicPreprocessor()

    checkpoint_config = checkpoint.get("config", {})
    effective_batch_size = int(batch_size or checkpoint_config.get("batch_size", 8))
    effective_max_length = int(max_length or checkpoint_config.get("max_length", 256))
    dataset = ABDataset(
        dataframe,
        tokenizer,
        max_length=effective_max_length,
        preprocessor=preprocessor,
        is_test=False if "aspects" in dataframe.columns and "aspect_sentiments" in dataframe.columns else True,
    )
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    return collect_probability_records(model, dataloader, device), checkpoint


def average_probability_records(model_records: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Average aligned probability records from multiple models."""
    if not model_records:
        return []
    base_records = []
    num_models = len(model_records)

    for index, first_record in enumerate(model_records[0]):
        probability_sum = np.zeros_like(first_record["label_probabilities"], dtype=np.float32)
        review_id = first_record["review_id"]
        review_text = first_record["review_text"]
        for records in model_records:
            record = records[index]
            if record["review_id"] != review_id:
                raise ValueError("Model probability records are misaligned by review_id.")
            probability_sum += np.asarray(record["label_probabilities"], dtype=np.float32)
        averaged_record = dict(first_record)
        averaged_record["label_probabilities"] = probability_sum / float(num_models)
        averaged_record["ensemble_size"] = num_models
        base_records.append(averaged_record)
    return base_records


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run ensemble prediction for Arabic ABSA.")
    parser.add_argument("--test_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--model_paths", nargs="*", type=Path, default=None)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    test_path = resolve_input_path(args.test_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None

    dataframe = load_dataframe(test_path)
    model_paths = [resolve_input_path(path) for path in args.model_paths] if args.model_paths else discover_model_paths(OUTPUTS_ROOT)
    model_paths = [path for path in model_paths if path and path.exists()]
    if not model_paths:
        fallback_model = resolve_input_path(DEFAULT_MODEL_PATH, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
        if fallback_model.exists():
            model_paths = [fallback_model]
        else:
            raise FileNotFoundError("No model checkpoints were found for ensemble prediction.")

    model_probability_runs: List[List[Dict[str, Any]]] = []
    last_checkpoint: Dict[str, Any] = {}
    for model_path in model_paths:
        probability_records, checkpoint = collect_model_probability_records(
            dataframe,
            model_path,
            args.base_model_name,
            args.batch_size,
            args.max_length,
        )
        model_probability_runs.append(probability_records)
        last_checkpoint = checkpoint

    averaged_records = average_probability_records(model_probability_runs)
    threshold_config = load_thresholds_for_checkpoint(last_checkpoint, threshold_path)
    predictions = postprocess_probability_records(averaged_records, threshold_config=threshold_config)
    submission = generate_submission(predictions, output_path)
    print(
        json.dumps(
            {
                "ensemble_size": len(model_paths),
                "model_paths": [str(path) for path in model_paths],
                "num_predictions": len(submission),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
