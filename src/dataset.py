"""
Shared dataset, schema, and label utilities for Arabic ABSA.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from preprocess import ArabicPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

DEFAULT_TRAIN_PATH = DATASET_ROOT / "DeepX_train.xlsx"
DEFAULT_VALIDATION_PATH = DATASET_ROOT / "DeepX_validation.xlsx"
DEFAULT_UNLABELED_PATH = DATASET_ROOT / "DeepX_unlabeled.xlsx"
DEFAULT_SAMPLE_SUBMISSION_PATH = DATASET_ROOT / "sample_submission.json"

VALID_ASPECTS = [
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
VALID_SENTIMENTS = ["positive", "negative", "neutral"]
NON_NONE_ASPECTS = [aspect for aspect in VALID_ASPECTS if aspect != "none"]

ASPECT_SENTIMENT_LABELS = [
    f"{aspect}_{sentiment}"
    for aspect in VALID_ASPECTS
    for sentiment in VALID_SENTIMENTS
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ASPECT_SENTIMENT_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

COLUMN_ALIASES = {
    "review_id": ("review_id", "reviewid", "id", "sample_id"),
    "review_text": ("review_text", "reviewtext", "text", "review", "comment", "content"),
    "aspects": ("aspects", "aspect", "labels", "aspect_labels"),
    "aspect_sentiments": (
        "aspect_sentiments",
        "aspectsentiments",
        "sentiments",
        "aspect_polarity",
        "label_sentiments",
    ),
}


@dataclass(frozen=True)
class ColumnMapping:
    """Resolved dataset columns."""

    review_id: str
    review_text: str
    aspects: Optional[str] = None
    aspect_sentiments: Optional[str] = None


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(column_name).strip().lower())


def resolve_input_path(path: Optional[Path], default_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve a user-provided path against the project and dataset roots."""
    if path is None:
        return default_path
    if path.is_absolute():
        return path

    candidates = [path, PROJECT_ROOT / path, DATASET_ROOT / path, OUTPUTS_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a spreadsheet or JSON file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def infer_column_mapping(dataframe: pd.DataFrame, require_labels: bool = False) -> ColumnMapping:
    """Infer the canonical review/text/label columns without hardcoding exact names."""
    normalized_lookup = {
        normalize_column_name(column_name): column_name
        for column_name in dataframe.columns
    }

    def resolve(field_name: str, required: bool) -> Optional[str]:
        aliases = [normalize_column_name(alias) for alias in COLUMN_ALIASES[field_name]]
        for alias in aliases:
            if alias in normalized_lookup:
                return normalized_lookup[alias]

        for normalized_column, original_column in normalized_lookup.items():
            if any(alias in normalized_column for alias in aliases):
                return original_column

        if required:
            raise ValueError(
                f"Could not infer the '{field_name}' column from available columns: "
                f"{list(dataframe.columns)}"
            )
        return None

    return ColumnMapping(
        review_id=resolve("review_id", required=True) or "review_id",
        review_text=resolve("review_text", required=True) or "review_text",
        aspects=resolve("aspects", required=require_labels),
        aspect_sentiments=resolve("aspect_sentiments", required=require_labels),
    )


def parse_json_column(value: Any) -> List[str]:
    """Safely parse a JSON-like list cell."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or pd.isna(value):
        return []
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
                continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            if isinstance(parsed, dict):
                return [str(item) for item in parsed.keys()]
    return []


def parse_sentiment_dict(value: Any) -> Dict[str, str]:
    """Safely parse a JSON-like aspect sentiment cell."""
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    if value is None or pd.isna(value):
        return {}
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict):
                return {str(key): str(item) for key, item in parsed.items()}
    return {}


def sanitize_aspect_sentiments(
    aspects: Sequence[str],
    sentiments: Mapping[str, str],
) -> Tuple[List[str], Dict[str, str]]:
    """Force predictions and labels into the allowed taxonomy and consistency rules."""
    ordered_aspects: List[str] = []
    aspect_sentiments: Dict[str, str] = {}
    seen = set()

    for raw_aspect in aspects:
        aspect = str(raw_aspect)
        if aspect not in VALID_ASPECTS or aspect in seen:
            continue
        seen.add(aspect)
        ordered_aspects.append(aspect)
        sentiment = str(sentiments.get(aspect, "neutral"))
        if sentiment not in VALID_SENTIMENTS:
            sentiment = "neutral"
        aspect_sentiments[aspect] = sentiment

    if "none" in ordered_aspects and len(ordered_aspects) > 1:
        ordered_aspects = [aspect for aspect in ordered_aspects if aspect != "none"]
        aspect_sentiments.pop("none", None)

    if not ordered_aspects:
        return ["none"], {"none": "neutral"}

    if ordered_aspects == ["none"]:
        return ["none"], {"none": "neutral"}

    return ordered_aspects, aspect_sentiments


def create_multi_label_vector(
    aspects: Sequence[str],
    sentiments: Mapping[str, str],
) -> np.ndarray:
    """Encode an aspect-sentiment assignment into the 27-label multi-hot vector."""
    label_vector = np.zeros(len(ASPECT_SENTIMENT_LABELS), dtype=np.float32)
    safe_aspects, safe_sentiments = sanitize_aspect_sentiments(aspects, sentiments)

    for aspect in safe_aspects:
        sentiment = safe_sentiments.get(aspect, "neutral")
        label = f"{aspect}_{sentiment}"
        label_index = LABEL_TO_IDX.get(label)
        if label_index is not None:
            label_vector[label_index] = 1.0
    return label_vector


def decode_multi_label_vector(
    label_vector: Sequence[float],
    threshold: float = 0.5,
) -> Tuple[List[str], Dict[str, str]]:
    """Decode probabilities or logits into one sentiment per detected aspect."""
    label_scores = np.asarray(label_vector, dtype=np.float32)
    selected_sentiments: Dict[str, str] = {}

    for aspect in VALID_ASPECTS:
        best_sentiment = "neutral"
        best_score = float("-inf")
        for sentiment in VALID_SENTIMENTS:
            label = f"{aspect}_{sentiment}"
            score = float(label_scores[LABEL_TO_IDX[label]])
            if score > best_score:
                best_score = score
                best_sentiment = sentiment
        if best_score >= threshold:
            selected_sentiments[aspect] = best_sentiment

    return sanitize_aspect_sentiments(list(selected_sentiments.keys()), selected_sentiments)


def labels_to_dataframe(
    predictions: Sequence[Mapping[str, Any]],
    include_metadata: bool = True,
) -> pd.DataFrame:
    """Convert prediction dictionaries into a training-friendly DataFrame."""
    rows: List[Dict[str, Any]] = []
    for prediction in predictions:
        row = {
            "review_id": prediction.get("review_id"),
            "review_text": prediction.get("review_text", ""),
            "aspects": json.dumps(prediction.get("aspects", []), ensure_ascii=False),
            "aspect_sentiments": json.dumps(
                prediction.get("aspect_sentiments", {}),
                ensure_ascii=False,
            ),
        }
        if include_metadata:
            for key in ("confidence", "source", "model_name"):
                if key in prediction:
                    row[key] = prediction[key]
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_label_matrix(dataframe: pd.DataFrame) -> np.ndarray:
    """Convert a labeled DataFrame into a binary label matrix."""
    column_mapping = infer_column_mapping(dataframe, require_labels=True)
    vectors = []
    for _, row in dataframe.iterrows():
        aspects = parse_json_column(row[column_mapping.aspects]) if column_mapping.aspects else []
        sentiments = (
            parse_sentiment_dict(row[column_mapping.aspect_sentiments])
            if column_mapping.aspect_sentiments
            else {}
        )
        vectors.append(create_multi_label_vector(aspects, sentiments))
    if not vectors:
        return np.zeros((0, len(ASPECT_SENTIMENT_LABELS)), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def compute_class_distribution(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Compute label, aspect, and sentiment frequency statistics."""
    label_matrix = dataframe_to_label_matrix(dataframe)
    total_samples = int(label_matrix.shape[0])
    label_counts = label_matrix.sum(axis=0).astype(int)
    label_distribution = {
        label: int(label_counts[index])
        for index, label in enumerate(ASPECT_SENTIMENT_LABELS)
    }

    aspect_distribution = {}
    for aspect in VALID_ASPECTS:
        indices = [LABEL_TO_IDX[f"{aspect}_{sentiment}"] for sentiment in VALID_SENTIMENTS]
        aspect_distribution[aspect] = int(label_counts[indices].sum())

    sentiment_distribution = {}
    for sentiment in VALID_SENTIMENTS:
        indices = [LABEL_TO_IDX[f"{aspect}_{sentiment}"] for aspect in VALID_ASPECTS]
        sentiment_distribution[sentiment] = int(label_counts[indices].sum())

    raw_pos_weight = (
        (total_samples - label_counts + 1.0) / (label_counts + 1.0)
        if total_samples > 0
        else np.ones(len(ASPECT_SENTIMENT_LABELS), dtype=np.float32)
    )
    clipped_pos_weight = np.clip(raw_pos_weight, 1.0, 20.0)

    return {
        "num_samples": total_samples,
        "label_distribution": label_distribution,
        "aspect_distribution": aspect_distribution,
        "sentiment_distribution": sentiment_distribution,
        "positive_ratio": {
            label: round(float(label_counts[index] / max(total_samples, 1)), 6)
            for index, label in enumerate(ASPECT_SENTIMENT_LABELS)
        },
        "pos_weight_raw": {
            label: round(float(raw_pos_weight[index]), 6)
            for index, label in enumerate(ASPECT_SENTIMENT_LABELS)
        },
        "pos_weight_clipped": {
            label: round(float(clipped_pos_weight[index]), 6)
            for index, label in enumerate(ASPECT_SENTIMENT_LABELS)
        },
    }


def build_pos_weight_tensor(distribution: Mapping[str, Any]) -> torch.Tensor:
    """Create the BCE pos_weight tensor from distribution metadata."""
    pos_weight_values = distribution.get("pos_weight_clipped", {})
    values = [
        float(pos_weight_values.get(label, 1.0))
        for label in ASPECT_SENTIMENT_LABELS
    ]
    return torch.tensor(values, dtype=torch.float32)


def coerce_review_id(value: Any) -> Any:
    """Return an int review id when possible, otherwise a stable string."""
    if value is None or pd.isna(value):
        return ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


class ABDataset(Dataset):
    """PyTorch dataset for Arabic ABSA model training and inference."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        preprocessor: Optional[ArabicPreprocessor] = None,
        is_test: bool = False,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or ArabicPreprocessor()
        self.is_test = is_test
        self.column_mapping = infer_column_mapping(self.dataframe, require_labels=not is_test)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[idx]
        review_text = ""
        if pd.notna(row[self.column_mapping.review_text]):
            review_text = str(row[self.column_mapping.review_text])

        normalized_text = self.preprocessor.normalize(review_text)
        encoding = self.tokenizer(
            normalized_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "review_id": coerce_review_id(row[self.column_mapping.review_id]),
            "text": review_text,
            "normalized_text": normalized_text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if not self.is_test and self.column_mapping.aspects and self.column_mapping.aspect_sentiments:
            aspects = parse_json_column(row[self.column_mapping.aspects])
            sentiments = parse_sentiment_dict(row[self.column_mapping.aspect_sentiments])
            item["labels"] = torch.from_numpy(create_multi_label_vector(aspects, sentiments))

        return item


def load_data(
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VALIDATION_PATH,
    test_path: Path = DEFAULT_UNLABELED_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, validation, and unlabeled datasets."""
    train_df = load_dataframe(resolve_input_path(train_path, DEFAULT_TRAIN_PATH) or DEFAULT_TRAIN_PATH)
    val_df = load_dataframe(resolve_input_path(val_path, DEFAULT_VALIDATION_PATH) or DEFAULT_VALIDATION_PATH)
    test_df = load_dataframe(resolve_input_path(test_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH)
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    print(f"Train: {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Unlabeled: {test_df.shape}")
    print(json.dumps(compute_class_distribution(train_df), ensure_ascii=False, indent=2))
