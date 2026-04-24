"""
Evaluation Module for Arabic ABSA
==================================
Handles model evaluation with various metrics and threshold tuning.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import ABDataset, ASPECT_SENTIMENT_LABELS, LABEL_TO_IDX
from preprocess import ArabicPreprocessor
from train import ABSAModel, DEFAULT_MODELS, compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_VALIDATION_PATH = DATASET_ROOT / "DeepX_validation.xlsx"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "model.pt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "validation_metrics.json"


def resolve_model_name(model_name: Optional[str]) -> Optional[str]:
    """Resolve a model alias to a Hugging Face model id."""
    if model_name is None:
        return None
    return DEFAULT_MODELS.get(model_name, model_name)


def load_model(
    model_path: str,
    model_name: Optional[str],
    num_labels: int,
    device: torch.device,
) -> Tuple[ABSAModel, Dict]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    resolved_model_name = resolve_model_name(model_name) or checkpoint.get("model_name")
    if not resolved_model_name:
        raise ValueError(
            "Base model name could not be resolved. Pass --base_model_name or "
            "retrain with the updated trainer."
        )

    model = ABSAModel(resolved_model_name, num_labels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_detailed(
    model: nn.Module,
    dataloader: DataLoader,
    val_df: Optional[pd.DataFrame] = None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> Dict:
    """Perform detailed evaluation with per-aspect metrics."""
    del val_df
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            all_predictions.append(torch.sigmoid(logits).cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    all_predictions_array = np.concatenate(all_predictions, axis=0)
    all_labels_array = np.concatenate(all_labels, axis=0) if all_labels else None

    metrics: Dict = {}
    if all_labels_array is not None:
        metrics = compute_metrics(all_predictions_array, all_labels_array, threshold)

    aspect_metrics = {}
    aspects = [
        "food",
        "service",
        "price",
        "cleanliness",
        "delivery",
        "ambiance",
        "app_experience",
        "general",
        "none",
    ]

    for aspect in aspects:
        aspect_indices = [
            LABEL_TO_IDX[f"{aspect}_{sentiment}"]
            for sentiment in ["positive", "negative", "neutral"]
            if f"{aspect}_{sentiment}" in LABEL_TO_IDX
        ]
        if not aspect_indices or all_labels_array is None:
            continue

        aspect_preds = all_predictions_array[:, aspect_indices]
        aspect_labels = all_labels_array[:, aspect_indices]
        aspect_pred_binary = (aspect_preds >= threshold).astype(int)
        aspect_label_binary = (aspect_labels >= 0.5).astype(int)

        tp = ((aspect_pred_binary == 1) & (aspect_label_binary == 1)).sum()
        fp = ((aspect_pred_binary == 1) & (aspect_label_binary == 0)).sum()
        fn = ((aspect_pred_binary == 0) & (aspect_label_binary == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        aspect_metrics[aspect] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": int(aspect_pred_binary.sum()),
            "true_count": int(aspect_label_binary.sum()),
        }

    metrics["per_aspect"] = aspect_metrics
    return metrics


def tune_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict]:
    """Tune the classification threshold."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {}
    for threshold in thresholds:
        metrics, _ = evaluate_simple(model, dataloader, device, threshold=threshold)
        results[f"{threshold:.2f}"] = {
            "micro_f1": metrics["micro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
        }

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold, metrics in results.items():
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_threshold = float(threshold)

    return best_threshold, results


def evaluate_simple(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> Tuple[Dict, np.ndarray]:
    """Simple evaluation without detailed analysis."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            all_predictions.append(torch.sigmoid(logits).cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    all_predictions_array = np.concatenate(all_predictions, axis=0)
    metrics = {}
    if all_labels:
        all_labels_array = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics(all_predictions_array, all_labels_array, threshold)

    return metrics, all_predictions_array


def save_validation_metrics(metrics: Dict, output_path: str):
    """Save validation metrics to JSON file."""

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    metrics_serializable = convert_to_serializable(metrics)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics_serializable, handle, indent=2, ensure_ascii=False)

    print(f"Metrics saved to: {output_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Arabic ABSA model.")
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser


def main():
    """CLI entry point."""
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_df = pd.read_excel(args.validation_path)
    model, checkpoint = load_model(
        str(args.model_path),
        args.base_model_name,
        len(ASPECT_SENTIMENT_LABELS),
        device,
    )

    checkpoint_config = checkpoint.get("config", {})
    max_length = args.max_length or checkpoint_config.get("max_length", 256)
    batch_size = args.batch_size or checkpoint_config.get("batch_size", 16)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    tokenizer_name = resolve_model_name(args.base_model_name) or checkpoint.get("model_name")

    if not tokenizer_name:
        raise ValueError("Tokenizer name could not be resolved for evaluation.")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = ABDataset(
        val_df,
        tokenizer,
        max_length=max_length,
        preprocessor=ArabicPreprocessor(),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    metrics = evaluate_detailed(model, dataloader, val_df, device=device, threshold=threshold)
    metrics["threshold"] = threshold
    metrics["model_name"] = tokenizer_name
    save_validation_metrics(metrics, str(args.output_path))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
