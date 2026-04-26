"""
Shared utilities for cleaning, analyzing, and reusing unlabeled DeepX reviews.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import (
    ABDataset,
    DEFAULT_UNLABELED_PATH,
    OUTPUTS_ROOT,
    VALID_ASPECTS,
    VALID_SENTIMENTS,
    infer_column_mapping,
    load_dataframe,
    normalize_column_name,
    resolve_input_path,
)
from predict import (
    collect_probability_records,
    load_thresholds_for_checkpoint,
    load_trained_model,
    resolve_tokenizer_source,
)
from preprocess import ArabicPreprocessor, DEFAULT_FRANCO_THRESHOLD
from rules import apply_postprocessing, extract_rule_features


DEFAULT_MODEL_DIR = OUTPUTS_ROOT
DEFAULT_CLEAN_UNLABELED_PATH = OUTPUTS_ROOT / "clean_unlabeled.csv"
DEFAULT_PSEUDO_LABELED_PATH = OUTPUTS_ROOT / "pseudo_labeled.csv"
DEFAULT_ACTIVE_LEARNING_PATH = OUTPUTS_ROOT / "active_learning_samples.csv"
DEFAULT_KEYWORD_REPORT_PATH = OUTPUTS_ROOT / "aspect_keyword_report.json"
DEFAULT_STRESS_REPORT_PATH = OUTPUTS_ROOT / "unlabeled_stress_report.json"
DEFAULT_DOMAIN_ADAPTATION_CORPUS_PATH = OUTPUTS_ROOT / "domain_adaptation_corpus.txt"

ASPECT_KEYWORD_SEEDS = {
    "price": ["expensive", "cheap", "overpriced", "price", "السعر", "غالي", "رخيص"],
    "cleanliness": ["clean", "dirty", "toilets", "نظيف", "وسخ", "حمام", "حمامات"],
    "delivery": ["late", "arrived", "order", "driver", "تأخير", "وصل", "مندوب", "طلب"],
    "service": ["staff", "service", "employees", "موظفين", "تعامل", "خدمة"],
    "food": ["food", "meal", "taste", "أكل", "طعم", "وجبة", "مطعم"],
    "ambiance": ["place", "crowded", "quiet", "atmosphere", "مكان", "زحمة", "هادي", "جو"],
    "app_experience": ["app", "application", "website", "booking", "تطبيق", "ابلكيشن", "موقع", "حجز"],
    "general": ["good", "bad", "nice", "excellent", "حلو", "وحش", "ممتاز", "كويس"],
}

OPTIONAL_COLUMN_ALIASES = {
    "star_rating": ("star_rating", "starrating", "stars", "rating", "review_rating"),
    "business_category": ("business_category", "businesscategory", "category"),
    "platform": ("platform", "source_platform", "channel"),
    "business_name": ("business_name", "businessname", "store_name", "shop_name"),
    "date": ("date", "review_date", "created_at"),
}

ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
LATIN_CHAR_PATTERN = re.compile(r"[A-Za-z]")
TOKEN_PATTERN = re.compile(r"[\u0600-\u06FFA-Za-z0-9_]+")
FRANCO_HINT_PATTERN = re.compile(r"\b(?:msh|mesh|7aga|7لو|3shan|3ala|2a|2el|5ales|kwayes|helw)\b", re.IGNORECASE)


@dataclass(frozen=True)
class UnlabeledColumnMapping:
    """Resolved required and optional unlabeled-data columns."""

    review_id: str
    review_text: str
    star_rating: Optional[str] = None
    business_category: Optional[str] = None
    platform: Optional[str] = None
    business_name: Optional[str] = None
    date: Optional[str] = None


def format_json_cell(value: Any) -> str:
    """Serialize lists and dicts consistently for CSV outputs."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def save_csv(dataframe: pd.DataFrame, output_path: Path, json_columns: Optional[Sequence[str]] = None) -> Path:
    """Save a dataframe as UTF-8 CSV, serializing structured columns when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = dataframe.copy()
    for column_name in json_columns or []:
        if column_name in export_df.columns:
            export_df[column_name] = export_df[column_name].apply(format_json_cell)
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def save_json(data: Mapping[str, Any], output_path: Path) -> Path:
    """Save a UTF-8 JSON file with Arabic preserved."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return output_path


def resolve_optional_column(dataframe: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    """Find an optional column by fuzzy alias matching."""
    normalized_lookup = {
        normalize_column_name(column_name): column_name
        for column_name in dataframe.columns
    }
    normalized_aliases = [normalize_column_name(alias) for alias in aliases]
    for alias in normalized_aliases:
        if alias in normalized_lookup:
            return normalized_lookup[alias]
    for normalized_column, original_column in normalized_lookup.items():
        if any(alias in normalized_column for alias in normalized_aliases):
            return original_column
    return None


def infer_unlabeled_column_mapping(dataframe: pd.DataFrame) -> UnlabeledColumnMapping:
    """Infer required and optional unlabeled metadata columns."""
    required_mapping = infer_column_mapping(dataframe, require_labels=False)
    return UnlabeledColumnMapping(
        review_id=required_mapping.review_id,
        review_text=required_mapping.review_text,
        star_rating=resolve_optional_column(dataframe, OPTIONAL_COLUMN_ALIASES["star_rating"]),
        business_category=resolve_optional_column(dataframe, OPTIONAL_COLUMN_ALIASES["business_category"]),
        platform=resolve_optional_column(dataframe, OPTIONAL_COLUMN_ALIASES["platform"]),
        business_name=resolve_optional_column(dataframe, OPTIONAL_COLUMN_ALIASES["business_name"]),
        date=resolve_optional_column(dataframe, OPTIONAL_COLUMN_ALIASES["date"]),
    )


def parse_star_rating(value: Any) -> Optional[float]:
    """Parse star ratings conservatively while tolerating dirty spreadsheet cells."""
    if value is None or pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        parsed = float(match.group(0))
    if parsed < 0:
        return None
    return parsed


def weak_sentiment_from_star_rating(star_rating: Any) -> str:
    """Map stars to a weak overall sentiment signal without creating labels."""
    parsed = parse_star_rating(star_rating)
    if parsed is None:
        return "unknown"
    if parsed >= 4.0:
        return "positive"
    if parsed <= 2.0:
        return "negative"
    return "neutral"


def has_emoji_signal(emoji_tokens: Sequence[str], emoji_names: Sequence[str], raw_text: str) -> bool:
    """Return True when the review contains emoji-like sentiment cues."""
    _ = raw_text
    return bool(emoji_tokens or emoji_names)


def is_multilingual_text(text: str, language: str) -> bool:
    """Detect mixed-script or non-default-language content."""
    sample = str(text or "")
    has_arabic = bool(ARABIC_CHAR_PATTERN.search(sample))
    has_latin = bool(LATIN_CHAR_PATTERN.search(sample))
    if has_arabic and has_latin:
        return True
    return language not in {"ar", "en", "und"}


def is_franco_arabic_text(text: str, franco_ratio: float) -> bool:
    """Detect Franco-Arabic from ratio or common Arabizi markers."""
    sample = str(text or "")
    if franco_ratio > DEFAULT_FRANCO_THRESHOLD:
        return True
    if re.search(r"[23578]", sample) and LATIN_CHAR_PATTERN.search(sample):
        return True
    return bool(FRANCO_HINT_PATTERN.search(sample))


def tokenize_text(text: str) -> List[str]:
    """Lightweight multilingual tokenization for analytics."""
    return TOKEN_PATTERN.findall(str(text or ""))


def aggregate_predicted_sentiment(aspect_sentiments: Mapping[str, str]) -> str:
    """Collapse aspect sentiments into a coarse review-level polarity for weak-signal checks."""
    sentiments = [
        str(sentiment)
        for aspect, sentiment in aspect_sentiments.items()
        if aspect != "none" and sentiment in VALID_SENTIMENTS
    ]
    if not sentiments:
        return "neutral"
    unique_sentiments = set(sentiments)
    if "positive" in unique_sentiments and "negative" in unique_sentiments:
        return "mixed"
    if unique_sentiments == {"neutral"}:
        return "neutral"
    if unique_sentiments <= {"positive", "neutral"}:
        return "positive"
    if unique_sentiments <= {"negative", "neutral"}:
        return "negative"
    return "mixed"


def compare_weak_signal_to_prediction(weak_sentiment: str, aggregate_sentiment: str) -> str:
    """Compare star-derived weak sentiment to model output without overriding labels."""
    if weak_sentiment == "unknown":
        return "unknown"
    if aggregate_sentiment in {"mixed", "neutral"}:
        return "ambiguous"
    if weak_sentiment == aggregate_sentiment:
        return "aligned"
    return "contradictory"


def aspect_confusion_pair(aspect_probabilities: Mapping[str, float]) -> Optional[Tuple[str, str, float]]:
    """Return the closest top-two aspect pair when the model is uncertain."""
    candidates = [
        (aspect, float(score))
        for aspect, score in aspect_probabilities.items()
        if aspect != "none"
    ]
    if len(candidates) < 2:
        return None
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1]
    return ranked[0][0], ranked[1][0], margin


def prepare_domain_adaptation_corpus(cleaned_df: pd.DataFrame, output_path: Path) -> Path:
    """Write one conservatively cleaned review per line for future DAPT or MLM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        str(review).strip()
        for review in cleaned_df.get("review_text", pd.Series(dtype=str)).tolist()
        if str(review).strip()
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def clean_unlabeled_dataframe(
    unlabeled_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    preprocessor: Optional[ArabicPreprocessor] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean unlabeled reviews while preserving multilingual and emoji signals."""
    processor = preprocessor or ArabicPreprocessor()
    mapping = infer_unlabeled_column_mapping(unlabeled_df)
    cleaned_rows: List[Dict[str, Any]] = []
    seen_normalized_texts = set()

    total_rows = int(len(unlabeled_df))
    empty_removed = 0
    duplicate_removed = 0

    for row_index, (_, row) in enumerate(unlabeled_df.iterrows()):
        raw_text = ""
        if mapping.review_text in row and pd.notna(row[mapping.review_text]):
            raw_text = str(row[mapping.review_text]).strip()

        if not raw_text:
            empty_removed += 1
            continue

        cleaned_text = processor.clean(raw_text)
        analysis = processor.analyze(cleaned_text)
        normalized_text = analysis.normalized_text.strip()

        if not cleaned_text or not normalized_text:
            empty_removed += 1
            continue

        dedup_key = normalized_text.casefold()
        if dedup_key in seen_normalized_texts:
            duplicate_removed += 1
            continue

        seen_normalized_texts.add(dedup_key)

        star_rating = row[mapping.star_rating] if mapping.star_rating and mapping.star_rating in row else None
        token_count = len(tokenize_text(cleaned_text))
        cleaned_rows.append(
            {
                "review_id": row[mapping.review_id] if mapping.review_id in row else row_index,
                "review_text": cleaned_text,
                "normalized_review_text": normalized_text,
                "star_rating": parse_star_rating(star_rating),
                "weak_sentiment": weak_sentiment_from_star_rating(star_rating),
                "business_category": (
                    row[mapping.business_category]
                    if mapping.business_category and mapping.business_category in row
                    else None
                ),
                "platform": row[mapping.platform] if mapping.platform and mapping.platform in row else None,
                "business_name": (
                    row[mapping.business_name]
                    if mapping.business_name and mapping.business_name in row
                    else None
                ),
                "date": row[mapping.date] if mapping.date and mapping.date in row else None,
                "language": analysis.language,
                "franco_token_ratio": analysis.franco_token_ratio,
                "is_franco_arabic": is_franco_arabic_text(cleaned_text, analysis.franco_token_ratio),
                "is_multilingual": is_multilingual_text(cleaned_text, analysis.language),
                "contains_emoji": has_emoji_signal(analysis.emoji_tokens, analysis.emoji_names, raw_text),
                "token_count": token_count,
                "char_count": len(cleaned_text),
            }
        )

    cleaned_df = pd.DataFrame(cleaned_rows)

    if output_path is not None:
        save_csv(cleaned_df, output_path)

    summary = {
        "total_rows": total_rows,
        "clean_rows": int(len(cleaned_df)),
        "empty_removed": int(empty_removed),
        "duplicates_removed": int(duplicate_removed),
        "multilingual_rows": int(cleaned_df["is_multilingual"].sum()) if not cleaned_df.empty else 0,
        "franco_rows": int(cleaned_df["is_franco_arabic"].sum()) if not cleaned_df.empty else 0,
        "emoji_rows": int(cleaned_df["contains_emoji"].sum()) if not cleaned_df.empty else 0,
        "weak_sentiment_distribution": (
            cleaned_df["weak_sentiment"].value_counts(dropna=False).to_dict()
            if "weak_sentiment" in cleaned_df.columns
            else {}
        ),
        "output_path": str(output_path) if output_path is not None else None,
    }

    return cleaned_df, summary


def resolve_model_artifacts(
    model_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], Path]:
    """Resolve model, threshold, and model directory paths for unlabeled workflows."""
    resolved_model_dir = resolve_input_path(model_dir, DEFAULT_MODEL_DIR) or DEFAULT_MODEL_DIR
    if model_path is not None:
        resolved_model_path = resolve_input_path(model_path, model_path) or model_path
    else:
        resolved_model_path = resolve_input_path(resolved_model_dir / "model.pt", resolved_model_dir / "model.pt")
        if resolved_model_path is None:
            resolved_model_path = resolved_model_dir / "model.pt"

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Could not find a trained checkpoint at {resolved_model_path}. "
            "Pass --model_dir or --model_path pointing to a folder with model.pt."
        )

    if threshold_path is not None:
        resolved_threshold_path = resolve_input_path(threshold_path, threshold_path) or threshold_path
    else:
        candidate_threshold_path = resolved_model_path.parent / "best_thresholds.json"
        resolved_threshold_path = candidate_threshold_path if candidate_threshold_path.exists() else None

    return resolved_model_path, resolved_threshold_path, resolved_model_path.parent


def build_prediction_analysis_table(
    cleaned_df: pd.DataFrame,
    model_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Run the trained ABSA model and return prediction metadata for unlabeled reviews."""
    if cleaned_df.empty:
        return cleaned_df.copy()

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path, resolved_threshold_path, _ = resolve_model_artifacts(
        model_dir=model_dir,
        model_path=model_path,
        threshold_path=threshold_path,
    )
    model, checkpoint = load_trained_model(resolved_model_path, device=resolved_device)
    tokenizer_source = resolve_tokenizer_source(checkpoint, resolved_model_path, None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    checkpoint_config = checkpoint.get("config", {})
    effective_batch_size = int(batch_size or checkpoint_config.get("batch_size", 8))
    effective_max_length = int(max_length or checkpoint_config.get("max_length", 128))
    dataset = ABDataset(
        cleaned_df,
        tokenizer,
        max_length=effective_max_length,
        preprocessor=ArabicPreprocessor(),
        is_test=True,
    )
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    probability_records = collect_probability_records(model, dataloader, resolved_device)
    threshold_config = load_thresholds_for_checkpoint(checkpoint, resolved_threshold_path)

    if len(probability_records) != len(cleaned_df):
        raise ValueError(
            "Prediction count does not match cleaned unlabeled dataframe length. "
            f"Expected {len(cleaned_df)}, got {len(probability_records)}."
        )

    enriched_rows: List[Dict[str, Any]] = []
    for metadata, record in zip(cleaned_df.to_dict(orient="records"), probability_records):
        features = extract_rule_features(str(record.get("review_text", "")))
        decision = apply_postprocessing(
            text=str(record.get("review_text", "")),
            label_probabilities=record["label_probabilities"],
            threshold_config=threshold_config,
            features=features,
        )
        aggregate_sentiment = aggregate_predicted_sentiment(decision.aspect_sentiments)
        weak_signal_alignment = compare_weak_signal_to_prediction(
            str(metadata.get("weak_sentiment", "unknown")),
            aggregate_sentiment,
        )
        confusion = aspect_confusion_pair(decision.aspect_probabilities)
        predicted_labels = [
            f"{aspect}:{decision.aspect_sentiments.get(aspect, 'neutral')}"
            for aspect in decision.aspects
        ]
        try:
            json.dumps(
                {
                    "review_id": metadata.get("review_id"),
                    "aspects": decision.aspects,
                    "aspect_sentiments": decision.aspect_sentiments,
                },
                ensure_ascii=False,
            )
            prediction_json_valid = True
        except (TypeError, ValueError):
            prediction_json_valid = False

        enriched_rows.append(
            {
                **metadata,
                "predicted_aspects": list(decision.aspects),
                "predicted_sentiments": dict(decision.aspect_sentiments),
                "predicted_labels": predicted_labels,
                "confidence_score": float(decision.prediction_confidence),
                "aggregate_predicted_sentiment": aggregate_sentiment,
                "weak_signal_alignment": weak_signal_alignment,
                "num_predicted_aspects": int(len([aspect for aspect in decision.aspects if aspect != "none"])),
                "mixed_sentiment": bool(features.sentiment_conflict or aggregate_sentiment == "mixed"),
                "prediction_json_valid": prediction_json_valid,
                "matched_keywords": {k: v for k, v in decision.matched_keywords.items() if v},
                "rule_decisions": list(decision.rule_decisions),
                "emoji_tokens": list(decision.emoji_tokens),
                "emoji_names": list(decision.emoji_names),
                "sarcasm_candidate": bool(decision.sarcasm_candidate),
                "aspect_probabilities": dict(decision.aspect_probabilities),
                "sentiment_probabilities": dict(decision.sentiment_probabilities),
                "confused_aspect_pair": (
                    {"first": confusion[0], "second": confusion[1], "margin": round(float(confusion[2]), 6)}
                    if confusion is not None
                    else None
                ),
                "general_ambiance_app_confusion": (
                    confusion is not None
                    and {confusion[0], confusion[1]} <= {"general", "ambiance", "app_experience"}
                    and float(confusion[2]) < 0.08
                ),
                "short_meaningful": bool(int(metadata.get("token_count", 0)) <= 3 and decision.aspects != ["none"]),
            }
        )

    return pd.DataFrame(enriched_rows)


def load_and_clean_unlabeled_data(
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    output_path: Optional[Path] = DEFAULT_CLEAN_UNLABELED_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the DeepX unlabeled spreadsheet and save a cleaned CSV copy."""
    resolved_unlabeled_path = resolve_input_path(unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH
    unlabeled_df = load_dataframe(resolved_unlabeled_path)
    cleaned_df, summary = clean_unlabeled_dataframe(unlabeled_df, output_path=output_path)
    summary.update(
        {
            "input_path": str(resolved_unlabeled_path),
            "output_path": str(output_path) if output_path is not None else None,
        }
    )
    return cleaned_df, summary


def top_confused_aspects(prediction_df: pd.DataFrame, limit: int = 10) -> List[Dict[str, Any]]:
    """Summarize the most common near-tie aspect confusions."""
    counts: Dict[Tuple[str, str], int] = {}
    for confused_pair in prediction_df.get("confused_aspect_pair", pd.Series(dtype=object)).tolist():
        if not isinstance(confused_pair, dict):
            continue
        first = confused_pair.get("first")
        second = confused_pair.get("second")
        margin = float(confused_pair.get("margin", 1.0))
        if not first or not second or margin >= 0.1:
            continue
        key = tuple(sorted((str(first), str(second))))
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"pair": list(pair), "count": count}
        for pair, count in ranked[:limit]
    ]


def examples_from_dataframe(
    prediction_df: pd.DataFrame,
    mask: Iterable[bool],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Collect compact example records from a filtered prediction dataframe."""
    filtered_df = prediction_df.loc[list(mask)]
    examples: List[Dict[str, Any]] = []
    for _, row in filtered_df.head(limit).iterrows():
        examples.append(
            {
                "review_id": row.get("review_id"),
                "review_text": row.get("review_text"),
                "star_rating": row.get("star_rating"),
                "weak_sentiment": row.get("weak_sentiment"),
                "predicted_aspects": row.get("predicted_aspects"),
                "predicted_sentiments": row.get("predicted_sentiments"),
                "confidence_score": round(float(row.get("confidence_score", 0.0)), 6),
                "weak_signal_alignment": row.get("weak_signal_alignment"),
                "confused_aspect_pair": row.get("confused_aspect_pair"),
            }
        )
    return examples
