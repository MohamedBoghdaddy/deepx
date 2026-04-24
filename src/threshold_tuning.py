"""
Dynamic threshold tuning for Arabic ABSA.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import (
    ABDataset,
    ASPECT_SENTIMENT_LABELS,
    DEFAULT_VALIDATION_PATH,
    OUTPUTS_ROOT,
    VALID_SENTIMENTS,
    infer_column_mapping,
    load_dataframe,
    parse_json_column,
    parse_sentiment_dict,
    resolve_input_path,
)
from predict import (
    DEFAULT_MODEL_PATH,
    collect_probability_records,
    load_trained_model,
    resolve_tokenizer_source,
)
from preprocess import ArabicPreprocessor
from rules import apply_postprocessing, extract_rule_features, prediction_to_vector


DEFAULT_OUTPUT_PATH = OUTPUTS_ROOT / "best_thresholds.json"


def labels_from_dataframe(dataframe: pd.DataFrame) -> np.ndarray:
    """Build the gold label matrix from validation data."""
    from dataset import create_multi_label_vector

    mapping = infer_column_mapping(dataframe, require_labels=True)
    vectors = []
    for _, row in dataframe.iterrows():
        aspects = parse_json_column(row[mapping.aspects]) if mapping.aspects else []
        sentiments = parse_sentiment_dict(row[mapping.aspect_sentiments]) if mapping.aspect_sentiments else {}
        vectors.append(create_multi_label_vector(aspects, sentiments))
    return np.vstack(vectors).astype(np.float32)


def build_probability_records(
    dataframe: pd.DataFrame,
    model_path: Path,
    base_model_name: str | None,
    batch_size: int | None,
    max_length: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the model and collect cached probability records plus checkpoint metadata."""
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
    )
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    probability_records = collect_probability_records(model, dataloader, device)
    for record in probability_records:
        record["features"] = extract_rule_features(str(record.get("review_text", "")))
    return probability_records, checkpoint


def evaluate_threshold_config(
    probability_records: Sequence[Mapping[str, Any]],
    gold_matrix: np.ndarray,
    threshold_config: Mapping[str, Any],
) -> float:
    """Score a threshold configuration with final rule-aware predictions."""
    pred_vectors = []
    for record in probability_records:
        decision = apply_postprocessing(
            text=str(record.get("review_text", "")),
            label_probabilities=record["label_probabilities"],
            threshold_config=threshold_config,
            features=record.get("features"),
        )
        pred_vectors.append(prediction_to_vector(decision))
    pred_matrix = np.vstack(pred_vectors).astype(np.float32)
    return float(f1_score(gold_matrix.astype(int), pred_matrix.astype(int), average="micro", zero_division=0))


def derive_aspect_thresholds(label_thresholds: Mapping[str, float], default_threshold: float) -> Dict[str, float]:
    """Derive aspect-level thresholds by averaging the three sentiment thresholds."""
    aspect_thresholds: Dict[str, float] = {}
    for label_name in ASPECT_SENTIMENT_LABELS:
        aspect_name = label_name.rsplit("_", 1)[0]
        aspect_thresholds.setdefault(aspect_name, [])
        aspect_thresholds[aspect_name].append(float(label_thresholds.get(label_name, default_threshold)))
    return {
        aspect: round(float(sum(values) / len(values)), 6)
        for aspect, values in aspect_thresholds.items()
    }


def tune_global_threshold(
    probability_records: Sequence[Mapping[str, Any]],
    gold_matrix: np.ndarray,
    search_space: Sequence[float],
) -> tuple[float, float]:
    """Tune one global threshold."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in search_space:
        score = evaluate_threshold_config(
            probability_records,
            gold_matrix,
            {"global_threshold": float(threshold), "thresholds": {}, "aspect_thresholds": {}},
        )
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def tune_coordinate_descent(
    items: Sequence[str],
    probability_records: Sequence[Mapping[str, Any]],
    gold_matrix: np.ndarray,
    base_config: Dict[str, Any],
    search_space: Sequence[float],
    target_key: str,
    num_passes: int,
) -> tuple[Dict[str, Any], float]:
    """Greedy coordinate search over either per-label or per-aspect thresholds."""
    best_config = deepcopy(base_config)
    best_score = evaluate_threshold_config(probability_records, gold_matrix, best_config)

    for _ in range(num_passes):
        improved = False
        for item in items:
            item_best_threshold = best_config[target_key].get(item, best_config["global_threshold"])
            item_best_score = best_score
            for threshold in search_space:
                candidate_config = deepcopy(best_config)
                candidate_config[target_key][item] = float(threshold)
                score = evaluate_threshold_config(probability_records, gold_matrix, candidate_config)
                if score > item_best_score:
                    item_best_score = score
                    item_best_threshold = float(threshold)
            if item_best_score > best_score:
                best_config[target_key][item] = item_best_threshold
                best_score = item_best_score
                improved = True
        if not improved:
            break
    return best_config, best_score


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Tune rule-aware thresholds for Arabic ABSA.")
    parser.add_argument("--validation_path", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base_model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--granularity", choices=["global", "aspect", "label"], default="label")
    parser.add_argument("--num_passes", type=int, default=2)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    validation_path = resolve_input_path(args.validation_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH
    model_path = resolve_input_path(args.model_path, DEFAULT_MODEL_PATH) or DEFAULT_MODEL_PATH
    output_path = resolve_input_path(args.output_path, DEFAULT_OUTPUT_PATH) or DEFAULT_OUTPUT_PATH

    validation_df = load_dataframe(validation_path)
    gold_matrix = labels_from_dataframe(validation_df)
    probability_records, checkpoint = build_probability_records(
        validation_df,
        model_path,
        args.base_model_name,
        args.batch_size,
        args.max_length,
    )

    search_space = [round(float(value), 2) for value in np.arange(0.25, 0.76, 0.05)]
    checkpoint_threshold = float(checkpoint.get("best_threshold", 0.5))
    base_config = {
        "global_threshold": checkpoint_threshold,
        "thresholds": {},
        "aspect_thresholds": {},
    }

    if args.granularity == "global":
        best_threshold, best_micro_f1 = tune_global_threshold(probability_records, gold_matrix, search_space)
        best_config = {"global_threshold": best_threshold, "thresholds": {}, "aspect_thresholds": {}}
    elif args.granularity == "aspect":
        aspect_items = sorted({label.rsplit("_", 1)[0] for label in ASPECT_SENTIMENT_LABELS})
        best_config, best_micro_f1 = tune_coordinate_descent(
            aspect_items,
            probability_records,
            gold_matrix,
            base_config,
            search_space,
            target_key="aspect_thresholds",
            num_passes=args.num_passes,
        )
    else:
        best_config, best_micro_f1 = tune_coordinate_descent(
            ASPECT_SENTIMENT_LABELS,
            probability_records,
            gold_matrix,
            base_config,
            search_space,
            target_key="thresholds",
            num_passes=args.num_passes,
        )
        best_config["aspect_thresholds"] = derive_aspect_thresholds(
            best_config["thresholds"],
            best_config["global_threshold"],
        )

    if not best_config["aspect_thresholds"] and best_config["thresholds"]:
        best_config["aspect_thresholds"] = derive_aspect_thresholds(
            best_config["thresholds"],
            best_config["global_threshold"],
        )

    payload = {
        "metric": "micro_f1",
        "granularity": args.granularity,
        "search_space": search_space,
        "global_threshold": round(float(best_config["global_threshold"]), 6),
        "aspect_thresholds": {
            key: round(float(value), 6)
            for key, value in best_config["aspect_thresholds"].items()
        },
        "thresholds": {
            key: round(float(value), 6)
            for key, value in best_config["thresholds"].items()
        },
        "best_micro_f1": round(float(best_micro_f1), 6),
        "model_path": str(model_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
