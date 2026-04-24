"""
Single-model inference and submission generation for Arabic ABSA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import (
    ABDataset,
    ASPECT_SENTIMENT_LABELS,
    DEFAULT_SAMPLE_SUBMISSION_PATH,
    DEFAULT_UNLABELED_PATH,
    OUTPUTS_ROOT,
    load_dataframe,
    resolve_input_path,
)
from preprocess import ArabicPreprocessor
from rules import (
    DEFAULT_THRESHOLD_PATH,
    PredictionDecision,
    apply_postprocessing,
    load_threshold_config,
)
from train import ABSAModel, DEFAULT_MODELS, resolve_model_name


DEFAULT_MODEL_PATH = OUTPUTS_ROOT / "model.pt"
DEFAULT_OUTPUT_PATH = OUTPUTS_ROOT.parent / "submission.json"


def resolve_tokenizer_source(
    checkpoint: Mapping[str, Any],
    model_path: Path,
    fallback_model_name: Optional[str],
) -> str:
    """Prefer a saved tokenizer folder, then fall back to the checkpoint model name."""
    tokenizer_dir_name = checkpoint.get("tokenizer_dir_name")
    if tokenizer_dir_name:
        tokenizer_dir = model_path.resolve().parent / str(tokenizer_dir_name)
        if tokenizer_dir.exists():
            return str(tokenizer_dir)
    return str(
        checkpoint.get("tokenizer_name")
        or checkpoint.get("model_name")
        or resolve_model_name(fallback_model_name)
    )


def load_trained_model(
    model_path: Path,
    base_model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[ABSAModel, Dict[str, Any]]:
    """Load a checkpoint and instantiate the matching model for inference."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(model_path, map_location=resolved_device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility path
        checkpoint = torch.load(model_path, map_location=resolved_device)

    resolved_model_name = (
        resolve_model_name(base_model_name)
        if base_model_name
        else checkpoint.get("model_name")
    )
    if not resolved_model_name:
        raise ValueError(
            "Could not resolve the base transformer name. Pass --base_model_name explicitly."
        )

    model = ABSAModel(
        resolved_model_name,
        num_labels=len(ASPECT_SENTIMENT_LABELS),
        dropout=float(checkpoint.get("config", {}).get("dropout", 0.1)),
        load_pretrained=False,
        config_dict=checkpoint.get("transformer_config"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()
    return model, checkpoint


def normalize_batch_review_ids(batch_review_ids: Any) -> List[Any]:
    """Convert DataLoader-collated review ids into a plain Python list."""
    if isinstance(batch_review_ids, torch.Tensor):
        return batch_review_ids.detach().cpu().tolist()
    if isinstance(batch_review_ids, list):
        normalized = []
        for value in batch_review_ids:
            if isinstance(value, torch.Tensor):
                normalized.append(value.item())
            else:
                normalized.append(value)
        return normalized
    return [batch_review_ids]


def collect_probability_records(
    model: ABSAModel,
    dataloader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Run forward inference and return one record per review with label probabilities."""
    probability_records: List[Dict[str, Any]] = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            review_ids = normalize_batch_review_ids(batch["review_id"])
            texts = list(batch["text"])
            normalized_texts = list(batch["normalized_text"])

            for index, probability_vector in enumerate(probabilities):
                probability_records.append(
                    {
                        "review_id": review_ids[index],
                        "review_text": texts[index],
                        "normalized_text": normalized_texts[index],
                        "label_probabilities": np.asarray(probability_vector, dtype=np.float32),
                    }
                )
    return probability_records


def build_prediction_entry(
    record: Mapping[str, Any],
    decision: PredictionDecision,
    include_explanations: bool = False,
) -> Dict[str, Any]:
    """Convert a post-processed decision into a serializable prediction record."""
    prediction = {
        "review_id": record["review_id"],
        "review_text": record.get("review_text", ""),
        "aspects": decision.aspects,
        "aspect_sentiments": decision.aspect_sentiments,
        "confidence": decision.prediction_confidence,
    }
    if include_explanations:
        prediction["explanation"] = decision.to_dict()
    return prediction


def postprocess_probability_records(
    probability_records: Sequence[Mapping[str, Any]],
    threshold_config: Optional[Mapping[str, Any]] = None,
    include_explanations: bool = False,
) -> List[Dict[str, Any]]:
    """Apply the shared rules/threshold layer to raw probability records."""
    thresholds = dict(threshold_config or load_threshold_config())
    predictions: List[Dict[str, Any]] = []
    for record in probability_records:
        decision = apply_postprocessing(
            text=str(record.get("review_text", "")),
            label_probabilities=record["label_probabilities"],
            threshold_config=thresholds,
        )
        predictions.append(
            build_prediction_entry(record, decision, include_explanations=include_explanations)
        )
    return predictions


def load_thresholds_for_checkpoint(
    checkpoint: Mapping[str, Any],
    threshold_path: Optional[Path],
) -> Dict[str, Any]:
    """Load tuned thresholds, falling back to the checkpoint's best global threshold."""
    threshold_config = load_threshold_config(threshold_path)
    has_threshold_file = False
    if threshold_path is not None:
        has_threshold_file = Path(threshold_path).exists()
    else:
        has_threshold_file = DEFAULT_THRESHOLD_PATH.exists()

    if not has_threshold_file and "best_threshold" in checkpoint:
        threshold_config["global_threshold"] = float(checkpoint["best_threshold"])
    return threshold_config


def predict_dataframe(
    dataframe: pd.DataFrame,
    model_path: Path,
    base_model_name: Optional[str] = None,
    threshold_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    include_explanations: bool = False,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Run the full prediction pipeline for a DataFrame."""
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(model_path, base_model_name, resolved_device)
    tokenizer_source = resolve_tokenizer_source(checkpoint, model_path, base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    checkpoint_config = checkpoint.get("config", {})
    effective_batch_size = int(batch_size or checkpoint_config.get("batch_size", 8))
    effective_max_length = int(max_length or checkpoint_config.get("max_length", 256))
    preprocessor = ArabicPreprocessor()
    dataset = ABDataset(
        dataframe,
        tokenizer,
        max_length=effective_max_length,
        preprocessor=preprocessor,
        is_test=False if "aspects" in dataframe.columns and "aspect_sentiments" in dataframe.columns else True,
    )
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    probability_records = collect_probability_records(model, dataloader, resolved_device)
    threshold_config = load_thresholds_for_checkpoint(checkpoint, threshold_path)
    return postprocess_probability_records(
        probability_records,
        threshold_config=threshold_config,
        include_explanations=include_explanations,
    )


def generate_submission(
    predictions: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Write the exact competition submission schema."""
    submission = [
        {
            "review_id": prediction["review_id"],
            "aspects": list(prediction["aspects"]),
            "aspect_sentiments": dict(prediction["aspect_sentiments"]),
        }
        for prediction in predictions
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(submission, handle, ensure_ascii=False, indent=2)
    return submission


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the prediction CLI parser."""
    parser = argparse.ArgumentParser(description="Run single-model prediction for Arabic ABSA.")
    parser.add_argument("--test_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--threshold_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    test_path = resolve_input_path(args.test_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    model_path = resolve_input_path(args.model_path, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
    threshold_path = resolve_input_path(args.threshold_path) if args.threshold_path else None
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH

    dataframe = load_dataframe(test_path)
    predictions = predict_dataframe(
        dataframe=dataframe,
        model_path=model_path,
        base_model_name=args.base_model_name,
        threshold_path=threshold_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    submission = generate_submission(predictions, output_path)
    print(
        json.dumps(
            {
                "num_predictions": len(submission),
                "output_path": str(output_path),
                "model_path": str(model_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
