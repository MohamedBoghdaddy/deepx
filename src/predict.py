"""
Prediction Module for Arabic ABSA
==================================
Handles prediction on unlabeled data and submission generation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import (
    ABDataset,
    ASPECT_SENTIMENT_LABELS,
    VALID_ASPECTS,
    VALID_SENTIMENTS,
    decode_multi_label_vector,
)
from preprocess import ArabicPreprocessor
from train import ABSAModel, DEFAULT_MODELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_TEST_PATH = DATASET_ROOT / "DeepX_unlabeled.xlsx"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "model.pt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "submission.json"


def resolve_model_name(model_name: Optional[str]) -> Optional[str]:
    """Resolve a model alias to a Hugging Face model id."""
    if model_name is None:
        return None
    return DEFAULT_MODELS.get(model_name, model_name)


def resolve_tokenizer_source(checkpoint: Dict, model_path: str, fallback_model_name: Optional[str]) -> str:
    """Resolve a local tokenizer directory first, then fall back to the model id."""
    tokenizer_dir_name = checkpoint.get("tokenizer_dir_name")
    if tokenizer_dir_name:
        tokenizer_dir = Path(model_path).resolve().parent / tokenizer_dir_name
        if tokenizer_dir.exists():
            return str(tokenizer_dir)

    return (
        checkpoint.get("tokenizer_name")
        or checkpoint.get("model_name")
        or resolve_model_name(fallback_model_name)
    )


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> List[Dict]:
    """Generate predictions for unlabeled data."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            predictions = torch.sigmoid(logits).cpu().numpy()

            for index, prediction in enumerate(predictions):
                review_id = int(batch["review_id"][index].item())
                aspects, sentiments = decode_multi_label_vector(prediction, threshold)
                results.append(
                    {
                        "review_id": review_id,
                        "aspects": aspects,
                        "aspect_sentiments": sentiments,
                    }
                )

    return results


def load_trained_model(
    model_path: str,
    base_model_name: Optional[str],
    device: torch.device = torch.device("cpu"),
) -> Tuple[ABSAModel, Dict]:
    """Load a trained model from checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    resolved_base_model_name = resolve_model_name(base_model_name) or checkpoint.get("model_name")
    if not resolved_base_model_name:
        raise ValueError(
            "Base model name could not be resolved. Pass --base_model_name or "
            "retrain with the updated trainer."
        )

    model = ABSAModel(
        resolved_base_model_name,
        len(ASPECT_SENTIMENT_LABELS),
        load_pretrained=False,
        config_dict=checkpoint.get("transformer_config"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    metadata = {
        "config": checkpoint.get("config", {}),
        "threshold": checkpoint.get("best_threshold", checkpoint.get("threshold", 0.5)),
        "model_name": resolved_base_model_name,
        "tokenizer_source": resolve_tokenizer_source(checkpoint, model_path, base_model_name),
    }
    return model, metadata


def generate_submission(
    predictions: List[Dict],
    output_path: str,
    sample_submission_path: Optional[str] = None,
):
    """Generate submission JSON file."""
    del sample_submission_path
    validated_predictions = []

    for prediction in predictions:
        review_id = prediction.get("review_id")
        aspects = prediction.get("aspects", [])
        sentiments = prediction.get("aspect_sentiments", {})

        valid_aspects = []
        valid_sentiments = {}
        for aspect in aspects:
            if aspect not in VALID_ASPECTS:
                continue

            valid_aspects.append(aspect)
            sentiment = sentiments.get(aspect, "neutral")
            if sentiment not in VALID_SENTIMENTS:
                sentiment = "neutral"
            valid_sentiments[aspect] = sentiment

        if not valid_aspects:
            valid_aspects = ["none"]
            valid_sentiments = {"none": "neutral"}

        validated_predictions.append(
            {
                "review_id": review_id,
                "aspects": valid_aspects,
                "aspect_sentiments": valid_sentiments,
            }
        )

    validated_predictions.sort(key=lambda item: item["review_id"])
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(validated_predictions, handle, indent=2, ensure_ascii=False)

    print(f"Submission saved to: {output_file}")
    print(f"Total predictions: {len(validated_predictions)}")
    return validated_predictions


def run_prediction(
    test_df: pd.DataFrame,
    model_path: str,
    base_model_name: Optional[str],
    output_path: str,
    threshold: Optional[float] = None,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Run the full prediction pipeline."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {model_path}")
    model, metadata = load_trained_model(model_path, base_model_name, device)

    resolved_base_model_name = metadata["model_name"]
    checkpoint_config = metadata.get("config", {})
    resolved_threshold = metadata["threshold"] if threshold is None else threshold
    resolved_max_length = checkpoint_config.get("max_length", 256) if max_length is None else max_length
    resolved_batch_size = checkpoint_config.get("batch_size", 16) if batch_size is None else batch_size

    tokenizer_source = metadata["tokenizer_source"]
    print(f"Loading tokenizer: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    preprocessor = ArabicPreprocessor()

    print("Creating test dataset...")
    test_dataset = ABDataset(
        test_df,
        tokenizer,
        max_length=resolved_max_length,
        preprocessor=preprocessor,
        is_test=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Generating predictions...")
    predictions = predict(model, test_loader, device, resolved_threshold)

    print("Generating submission file...")
    generate_submission(predictions, output_path)
    return predictions


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for prediction."""
    parser = argparse.ArgumentParser(description="Run prediction for the Arabic ABSA model.")
    parser.add_argument("--test_path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser


def main():
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    test_df = pd.read_excel(args.test_path)
    predictions = run_prediction(
        test_df=test_df,
        model_path=str(args.model_path),
        base_model_name=args.base_model_name,
        output_path=str(args.output_path),
        threshold=args.threshold,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"Generated {len(predictions)} predictions.")


if __name__ == "__main__":
    main()
