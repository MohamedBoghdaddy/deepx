from __future__ import annotations

import ast
import json
import math
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as exc:  # pragma: no cover - Tkinter is a runtime dependency for the dashboard.
    raise RuntimeError(f"Tkinter is required to run the DeepX dashboard: {exc}") from exc

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional at runtime.
    pd = None


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE_CANDIDATES = [
    ROOT / "submission.json",
    ROOT / "submission_arabert.json",
    ROOT / "submission_marbert.json",
    ROOT / "submission_xlmr.json",
    ROOT / "submission_hidden_arabert.json",
    ROOT / "submission_hidden_marbert.json",
    ROOT / "submission_hidden_xlmr.json",
]

REVIEW_FILE_CANDIDATES = [
    ROOT / "data" / "processed" / "DeepX_unlabeled.xlsx",
    ROOT / "data" / "processed" / "DeepX_validation.xlsx",
    ROOT / "dataset" / "DeepX_unlabeled.xlsx",
    ROOT / "dataset" / "DeepX_validation.xlsx",
]

MODEL_BENCHMARK_CONFIG = [
    {
        "key": "arabert",
        "label": "AraBERT",
        "family": "Arabic specialist",
        "manifest_paths": [
            ROOT / "outputs" / "arabert_final" / "training_manifest.pkl",
        ],
        "prediction_paths": [
            ROOT / "submission_arabert.json",
            ROOT / "submission_hidden_arabert.json",
        ],
    },
    {
        "key": "marbert",
        "label": "MARBERT",
        "family": "Dialect and social review specialist",
        "manifest_paths": [
            ROOT / "outputs" / "marbert_final" / "training_manifest.pkl",
        ],
        "prediction_paths": [
            ROOT / "submission_marbert.json",
            ROOT / "submission_hidden_marbert.json",
        ],
    },
    {
        "key": "xlmr",
        "label": "XLM-R",
        "family": "Multilingual generalist",
        "manifest_paths": [
            ROOT / "outputs" / "xlmr_final" / "training_manifest.pkl",
            ROOT / "outputs" / "training_manifest.pkl",
        ],
        "prediction_paths": [
            ROOT / "submission_xlmr.json",
            ROOT / "submission_hidden_xlmr.json",
        ],
    },
]

ASPECT_MEANINGS = {
    "food": "Product quality and taste",
    "service": "Staff behavior and response quality",
    "price": "Value for money and affordability",
    "cleanliness": "Hygiene and trust",
    "delivery": "Speed, accuracy, and logistics",
    "ambiance": "Place comfort and atmosphere",
    "app_experience": "Digital journey and usability",
    "general": "Overall brand perception",
}

ASPECT_DISPLAY_ORDER = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
]

ASPECT_ALIASES = {
    "app": "app_experience",
    "app experience": "app_experience",
    "app-experience": "app_experience",
    "application": "app_experience",
    "customer_service": "service",
    "hygiene": "cleanliness",
    "overall": "general",
    "brand": "general",
    "atmosphere": "ambiance",
    "environment": "ambiance",
}

PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

COLORS = {
    "background": "#0F172A",
    "panel": "#111827",
    "card": "#1E293B",
    "border": "#334155",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "blue": "#38BDF8",
    "purple": "#A78BFA",
    "green": "#22C55E",
    "yellow": "#FACC15",
    "orange": "#FB923C",
    "red": "#EF4444",
}

ROLE_CARDS = [
    (
        "CEO / Founder",
        "Understands brand risk, satisfaction shifts, and where customer trust is being won or lost.",
        COLORS["purple"],
    ),
    (
        "Operations Manager",
        "Finds process issues across delivery, cleanliness, and service before they damage repeat business.",
        COLORS["orange"],
    ),
    (
        "Marketing Manager",
        "Identifies strengths customers already love so campaigns can amplify the right messages.",
        COLORS["green"],
    ),
    (
        "Product Manager",
        "Improves app experience, ordering friction, and the digital customer journey with direct evidence.",
        COLORS["blue"],
    ),
]

RECOMMENDATION_LIBRARY = {
    "service": "Improve staff responsiveness, complaint handling, and customer communication.",
    "food": "Review food quality consistency, taste, freshness, and preparation standards.",
    "price": "Reassess value perception, bundle offers, and pricing communication.",
    "delivery": "Improve delivery speed, packaging reliability, and order tracking.",
    "app_experience": "Fix friction points in ordering, payment, navigation, and tracking.",
    "cleanliness": "Prioritize hygiene checks, visible cleaning routines, and quality control.",
    "ambiance": "Improve seating comfort, noise level, lighting, and atmosphere.",
    "general": "Audit the full customer journey and tighten the overall brand promise at every touchpoint.",
}

FONT_TITLE = ("Bahnschrift", 30, "bold")
FONT_H1 = ("Bahnschrift", 22, "bold")
FONT_H2 = ("Bahnschrift", 16, "bold")
FONT_CARD = ("Aptos", 12)
FONT_BODY = ("Aptos", 11)
FONT_SMALL = ("Aptos", 10)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def metric_text(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def short_text(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def as_ascii_safe(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def standardize_review_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if pd is not None:
        try:
            if pd.isna(value):
                return True
        except Exception:
            pass
    return False


def try_parse_structured_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text[0] not in "[{\"'":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def normalize_aspect_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.strip("\"'")
    text = text.replace("/", "_").replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = ASPECT_ALIASES.get(text, text)
    if text in {"", "none", "no_aspect", "null", "nan", "na", "n_a"}:
        return ""
    return text


def normalize_sentiment(value: Any) -> str:
    text = str(value or "").strip().lower().strip("\"'")
    if not text or text in {"none", "null", "nan", "unknown", "mixed", "unclear"}:
        return "neutral"
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "good": "positive",
        "negative": "negative",
        "neg": "negative",
        "bad": "negative",
        "neutral": "neutral",
        "neu": "neutral",
    }
    if text in mapping:
        return mapping[text]
    if "pos" in text:
        return "positive"
    if "neg" in text:
        return "negative"
    return "neutral"


def parse_aspects(value: Any) -> list[str]:
    items: list[str] = []

    def collect(item: Any) -> None:
        if item is None or is_missing_value(item):
            return
        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                collect(sub_item)
            return
        if isinstance(item, dict):
            if "aspect" in item:
                collect(item.get("aspect"))
                return
            for key, sub_value in item.items():
                aspect_name = normalize_aspect_name(key)
                if aspect_name and not isinstance(sub_value, (list, tuple, set, dict)):
                    items.append(aspect_name)
                else:
                    collect(sub_value)
            return
        if isinstance(item, str):
            parsed = try_parse_structured_value(item)
            if parsed is not None:
                collect(parsed)
                return
            if any(separator in item for separator in [",", ";", "|"]):
                for part in re.split(r"[;,|]+", item):
                    collect(part)
                return
        aspect_name = normalize_aspect_name(item)
        if aspect_name:
            items.append(aspect_name)

    collect(value)
    unique_items: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def parse_aspect_sentiments(value: Any) -> dict[str, str]:
    sentiments: dict[str, str] = {}

    def collect(item: Any) -> None:
        if item is None or is_missing_value(item):
            return
        if isinstance(item, dict):
            if "aspect" in item and ("sentiment" in item or "label" in item):
                aspect_name = normalize_aspect_name(item.get("aspect"))
                if aspect_name:
                    sentiments[aspect_name] = normalize_sentiment(item.get("sentiment") or item.get("label"))
                return
            for key, sub_value in item.items():
                if isinstance(sub_value, (dict, list, tuple, set)):
                    if key.lower() in {"prediction", "predictions", "aspect_sentiments", "sentiments", "data"}:
                        collect(sub_value)
                        continue
                aspect_name = normalize_aspect_name(key)
                if aspect_name:
                    sentiments[aspect_name] = normalize_sentiment(sub_value)
            return
        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                collect(sub_item)
            return
        if isinstance(item, str):
            parsed = try_parse_structured_value(item)
            if parsed is not None:
                collect(parsed)
                return
            matched = False
            for part in re.split(r"[;,|]+", item):
                if ":" not in part:
                    continue
                aspect_text, sentiment_text = part.split(":", 1)
                aspect_name = normalize_aspect_name(aspect_text)
                if aspect_name:
                    sentiments[aspect_name] = normalize_sentiment(sentiment_text)
                    matched = True
            if matched:
                return

    collect(value)
    return sentiments


def extract_prediction_records(raw_data: Any, inherited_review_id: str | None = None) -> list[dict[str, Any]]:
    if raw_data is None:
        return []
    if isinstance(raw_data, list):
        records: list[dict[str, Any]] = []
        for item in raw_data:
            records.extend(extract_prediction_records(item, inherited_review_id))
        return records
    if isinstance(raw_data, dict):
        direct_prediction_keys = {
            "review_id",
            "aspects",
            "aspect_sentiments",
            "sentiments",
            "predicted_aspects",
            "predicted_sentiments",
        }
        if direct_prediction_keys.intersection(raw_data.keys()):
            record = dict(raw_data)
            if inherited_review_id and not record.get("review_id"):
                record["review_id"] = inherited_review_id
            return [record]
        for key in ("predictions", "records", "data", "items", "results", "submission", "submissions"):
            if key in raw_data:
                return extract_prediction_records(raw_data.get(key), inherited_review_id)
        records = []
        for key, value in raw_data.items():
            if not isinstance(value, (dict, list)):
                continue
            next_review_id = standardize_review_id(key) or inherited_review_id
            records.extend(extract_prediction_records(value, next_review_id))
        return records
    return []


def normalize_prediction_records(raw_data: Any) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    raw_records = extract_prediction_records(raw_data)
    for index, record in enumerate(raw_records, start=1):
        review_id = (
            standardize_review_id(record.get("review_id"))
            or standardize_review_id(record.get("id"))
            or standardize_review_id(record.get("reviewId"))
            or f"row-{index}"
        )
        nested_prediction = record.get("prediction") or record.get("predictions")
        aspects = parse_aspects(
            record.get("aspects")
            or record.get("predicted_aspects")
            or record.get("labels")
            or (nested_prediction.get("aspects") if isinstance(nested_prediction, dict) else None)
        )
        aspect_sentiments = parse_aspect_sentiments(
            record.get("aspect_sentiments")
            or record.get("sentiments")
            or record.get("predicted_sentiments")
            or (nested_prediction.get("aspect_sentiments") if isinstance(nested_prediction, dict) else None)
        )
        if not aspects and aspect_sentiments:
            aspects = list(aspect_sentiments.keys())
        if aspects and not aspect_sentiments:
            aspect_sentiments = {aspect: "neutral" for aspect in aspects}
        for aspect in aspects:
            aspect_sentiments.setdefault(aspect, "neutral")
        aspects = [aspect for aspect in aspects if aspect]
        aspect_sentiments = {
            normalize_aspect_name(aspect): normalize_sentiment(sentiment)
            for aspect, sentiment in aspect_sentiments.items()
            if normalize_aspect_name(aspect)
        }
        if not aspects and not aspect_sentiments:
            continue
        aspects = [aspect for aspect in aspects if aspect in aspect_sentiments or aspect]
        normalized_records.append(
            {
                "review_id": review_id,
                "aspects": aspects,
                "aspect_sentiments": aspect_sentiments,
            }
        )
    return normalized_records


def discover_prediction_files() -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in PREDICTION_FILE_CANDIDATES:
        if path.exists() and path not in seen:
            seen.add(path)
            discovered.append(path)
    return discovered


def load_prediction_file(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return [], f"Unable to read prediction file {path.name}: {exc}"
    try:
        raw_data = json.loads(raw_text)
    except Exception as exc:
        return [], f"Unable to parse prediction JSON from {path.name}: {exc}"
    try:
        records = normalize_prediction_records(raw_data)
    except Exception as exc:
        return [], f"Unable to normalize predictions from {path.name}: {exc}"
    for record in records:
        record["source_file"] = path.name
    return records, None


def load_review_dataframe() -> tuple[Any, list[Path], list[str]]:
    if pd is None:
        return None, [], ["pandas is not installed. Review text loading is disabled."]
    frames = []
    loaded_paths: list[Path] = []
    errors: list[str] = []
    for path in REVIEW_FILE_CANDIDATES:
        if not path.exists():
            continue
        try:
            frame = pd.read_excel(path)
        except Exception as exc:
            errors.append(f"Unable to load review file {path.name}: {exc}")
            continue
        lower_map = {str(column).strip().lower(): column for column in frame.columns}
        rename_map = {}
        for target_name, aliases in {
            "review_id": ["review_id", "id", "reviewid"],
            "review_text": ["review_text", "review", "text", "content", "comment"],
            "business_name": ["business_name", "business", "store_name", "entity_name"],
            "business_category": ["business_category", "category"],
            "platform": ["platform", "source"],
            "star_rating": ["star_rating", "rating", "stars"],
        }.items():
            for alias in aliases:
                if alias in lower_map:
                    rename_map[lower_map[alias]] = target_name
                    break
        frame = frame.rename(columns=rename_map)
        if "review_id" not in frame.columns or "review_text" not in frame.columns:
            errors.append(f"Skipping {path.name} because review_id/review_text columns were not found.")
            continue
        available_columns = [
            column
            for column in [
                "review_id",
                "review_text",
                "business_name",
                "business_category",
                "platform",
                "star_rating",
            ]
            if column in frame.columns
        ]
        frame = frame[available_columns].copy()
        frame["review_id"] = frame["review_id"].apply(standardize_review_id)
        frame = frame[frame["review_id"].notna()].copy()
        frame["__source_path"] = str(path)
        frames.append(frame)
        loaded_paths.append(path)
    if not frames:
        return None, loaded_paths, errors
    reviews_df = pd.concat(frames, ignore_index=True, sort=False)
    reviews_df = reviews_df.drop_duplicates(subset=["review_id"], keep="first")
    return reviews_df, loaded_paths, errors


def compute_aspect_summary(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "aspect": "",
            "aspect_label": "",
            "mentions": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "neutral_rate": 0.0,
            "business_meaning": "",
            "priority": "Low",
        }
    )
    for prediction in predictions:
        sentiments = prediction.get("aspect_sentiments", {}) or {}
        for aspect, sentiment in sentiments.items():
            aspect_name = normalize_aspect_name(aspect)
            if not aspect_name:
                continue
            row = summary[aspect_name]
            row["aspect"] = aspect_name
            row["aspect_label"] = aspect_name.replace("_", " ").title()
            row["mentions"] += 1
            row[normalize_sentiment(sentiment)] += 1
    if not summary:
        return []
    max_mentions = max(row["mentions"] for row in summary.values()) or 1
    for row in summary.values():
        mentions = row["mentions"] or 1
        row["positive_rate"] = row["positive"] / mentions
        row["negative_rate"] = row["negative"] / mentions
        row["neutral_rate"] = row["neutral"] / mentions
        row["business_meaning"] = ASPECT_MEANINGS.get(row["aspect"], "Customer experience driver")
        high_volume = row["mentions"] >= max(3, math.ceil(max_mentions * 0.55))
        medium_volume = row["mentions"] >= max(2, math.ceil(max_mentions * 0.30))
        if row["negative_rate"] >= 0.45 and high_volume:
            row["priority"] = "Critical"
        elif row["negative_rate"] >= 0.30 and (high_volume or medium_volume or row["negative"] >= 3):
            row["priority"] = "High"
        elif row["neutral_rate"] >= 0.35 or row["negative_rate"] >= 0.15:
            row["priority"] = "Medium"
        else:
            row["priority"] = "Low"
    ordered_aspects = {aspect: index for index, aspect in enumerate(ASPECT_DISPLAY_ORDER)}
    return sorted(
        summary.values(),
        key=lambda row: (
            PRIORITY_RANK.get(row["priority"], 99),
            -row["negative_rate"],
            -row["mentions"],
            ordered_aspects.get(row["aspect"], 999),
            row["aspect"],
        ),
    )


def compute_business_kpis(aspect_summary: list[dict[str, Any]]) -> dict[str, Any]:
    positive_count = sum(item["positive"] for item in aspect_summary)
    negative_count = sum(item["negative"] for item in aspect_summary)
    neutral_count = sum(item["neutral"] for item in aspect_summary)
    total_mentions = positive_count + negative_count + neutral_count
    satisfaction_score = positive_count / total_mentions if total_mentions else 0.0
    risk_score = negative_count / total_mentions if total_mentions else 0.0
    neutral_rate = neutral_count / total_mentions if total_mentions else 0.0
    most_mentioned = max(aspect_summary, key=lambda item: item["mentions"], default=None)
    most_negative = max(
        aspect_summary,
        key=lambda item: (item["negative_rate"], item["negative"], item["mentions"]),
        default=None,
    )
    most_positive = max(
        aspect_summary,
        key=lambda item: (item["positive_rate"], item["positive"], item["mentions"]),
        default=None,
    )
    urgent_issue_count = sum(1 for item in aspect_summary if item["priority"] in {"Critical", "High"})
    if risk_score > 0.45:
        urgency = "High"
    elif risk_score > 0.25:
        urgency = "Medium"
    else:
        urgency = "Low"
    if risk_score >= 0.45 or urgent_issue_count >= 2:
        recommended_priority = "Critical"
    elif risk_score >= 0.25 or urgent_issue_count >= 1:
        recommended_priority = "High"
    elif neutral_rate >= 0.25:
        recommended_priority = "Medium"
    else:
        recommended_priority = "Low"
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_aspect_mentions": total_mentions,
        "satisfaction_score": satisfaction_score,
        "risk_score": risk_score,
        "neutral_rate": neutral_rate,
        "urgency": urgency,
        "most_mentioned_aspect": most_mentioned,
        "most_negative_aspect": most_negative,
        "most_positive_aspect": most_positive,
        "dominant_pain_point": most_negative["aspect_label"] if most_negative else "Not enough data",
        "strongest_satisfaction_driver": most_positive["aspect_label"] if most_positive else "Not enough data",
        "urgent_issue_count": urgent_issue_count,
        "recommended_priority": recommended_priority,
    }


def generate_business_recommendations(aspect_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for item in aspect_summary:
        priority = item["priority"]
        if priority not in {"Critical", "High"}:
            continue
        aspect = item["aspect"]
        action = RECOMMENDATION_LIBRARY.get(
            aspect,
            "Run a focused quality review on this aspect and assign an owner to fix the root cause.",
        )
        timeline = "Immediate" if priority == "Critical" else "Next sprint"
        recommendations.append(
            {
                "title": f"{item['aspect_label']} recovery plan",
                "priority": priority,
                "owner": "Operations" if aspect in {"service", "delivery", "cleanliness", "food"} else "Leadership",
                "timeline": timeline,
                "detail": (
                    f"{action} Current signal: {pct(item['negative_rate'])} negative across "
                    f"{item['mentions']} mentions."
                ),
            }
        )
    strengths = [
        item
        for item in aspect_summary
        if item["positive_rate"] >= 0.60 and item["mentions"] >= 2 and item["priority"] == "Low"
    ]
    for item in strengths[:2]:
        recommendations.append(
            {
                "title": f"Promote {item['aspect_label']}",
                "priority": "Opportunity",
                "owner": "Marketing",
                "timeline": "This quarter",
                "detail": (
                    f"Customer feedback is {pct(item['positive_rate'])} positive across "
                    f"{item['mentions']} mentions. Turn this into campaigns, retention messaging, or proof points."
                ),
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "title": "Maintain current customer strengths",
                "priority": "Low",
                "owner": "Leadership",
                "timeline": "Ongoing",
                "detail": "Signals are mostly stable. Keep monitoring weekly and protect the strongest parts of the experience.",
            }
        )
    priority_order = {"Critical": 0, "High": 1, "Opportunity": 2, "Medium": 3, "Low": 4}
    return sorted(recommendations, key=lambda item: priority_order.get(item["priority"], 99))


def build_review_lookup(reviews_df: Any) -> dict[str, dict[str, Any]]:
    if reviews_df is None or pd is None or reviews_df.empty:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for row in reviews_df.to_dict(orient="records"):
        review_id = standardize_review_id(row.get("review_id"))
        if not review_id:
            continue
        cleaned_row = {}
        for key, value in row.items():
            if is_missing_value(value):
                cleaned_row[key] = None
            else:
                cleaned_row[key] = value
        lookup.setdefault(review_id, cleaned_row)
    return lookup


def get_customer_voice_examples(
    predictions: list[dict[str, Any]],
    reviews_df: Any,
    target_aspect: str | None = None,
) -> Any:
    review_lookup = build_review_lookup(reviews_df)
    enriched_predictions = []
    for prediction in predictions:
        review_context = review_lookup.get(prediction.get("review_id"), {})
        review_text = review_context.get("review_text") or prediction.get("review_text")
        if not review_text:
            continue
        sentiments = prediction.get("aspect_sentiments", {}) or {}
        if target_aspect and target_aspect not in sentiments:
            continue
        positive_hits = sum(1 for sentiment in sentiments.values() if sentiment == "positive")
        negative_hits = sum(1 for sentiment in sentiments.values() if sentiment == "negative")
        enriched_predictions.append(
            {
                "review_id": prediction.get("review_id"),
                "review_text": review_text,
                "business_name": review_context.get("business_name"),
                "platform": review_context.get("platform"),
                "aspect_sentiments": sentiments,
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
            }
        )
    if target_aspect:
        target_examples = []
        for item in enriched_predictions:
            aspect_sentiment = item["aspect_sentiments"].get(target_aspect)
            if not aspect_sentiment:
                continue
            score = item["negative_hits"] if aspect_sentiment == "negative" else item["positive_hits"]
            target_examples.append((score, item))
        target_examples.sort(
            key=lambda pair: (
                -pair[0],
                -len(pair[1]["aspect_sentiments"]),
                len(str(pair[1]["review_text"])),
            )
        )
        return [example for _, example in target_examples[:3]]
    positive_examples = sorted(
        [item for item in enriched_predictions if item["positive_hits"] > 0],
        key=lambda item: (-item["positive_hits"], len(str(item["review_text"]))),
    )[:3]
    negative_examples = sorted(
        [item for item in enriched_predictions if item["negative_hits"] > 0],
        key=lambda item: (-item["negative_hits"], len(str(item["review_text"]))),
    )[:3]
    return {"positive": positive_examples, "negative": negative_examples}


def compute_business_insights(predictions: list[dict[str, Any]], reviews_df: Any = None) -> dict[str, Any]:
    review_lookup = build_review_lookup(reviews_df)
    enriched_predictions = []
    for prediction in predictions:
        record = dict(prediction)
        review_context = review_lookup.get(record.get("review_id"), {})
        for key in ("review_text", "business_name", "business_category", "platform", "star_rating"):
            if key not in record or not record.get(key):
                record[key] = review_context.get(key)
        enriched_predictions.append(record)
    aspect_summary = compute_aspect_summary(enriched_predictions)
    kpis = compute_business_kpis(aspect_summary)
    customer_voice = get_customer_voice_examples(enriched_predictions, reviews_df)
    most_negative_aspect = kpis.get("most_negative_aspect")
    most_negative_examples = []
    if most_negative_aspect:
        most_negative_examples = get_customer_voice_examples(
            enriched_predictions,
            reviews_df,
            target_aspect=most_negative_aspect["aspect"],
        )
    return {
        "predictions": enriched_predictions,
        "aspect_summary": aspect_summary,
        "kpis": kpis,
        "recommendations": generate_business_recommendations(aspect_summary),
        "customer_voice": customer_voice,
        "most_negative_aspect_examples": most_negative_examples,
        "review_text_available": any(record.get("review_text") for record in enriched_predictions),
        "total_reviews": len(enriched_predictions),
    }


def load_pickle_file(path: Path) -> tuple[Any, str | None]:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle), None
    except Exception as exc:
        return None, f"Unable to read benchmark manifest {path.name}: {exc}"


def load_model_benchmark_records() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for config in MODEL_BENCHMARK_CONFIG:
        manifest_data = None
        manifest_path = None
        for candidate in config["manifest_paths"]:
            if not candidate.exists():
                continue
            manifest_data, error = load_pickle_file(candidate)
            if error:
                errors.append(error)
                continue
            manifest_path = candidate
            break
        metrics = {}
        checkpoint_path = None
        if isinstance(manifest_data, dict):
            metrics = manifest_data.get("metrics", {}) or {}
            checkpoint_path = manifest_data.get("checkpoint_path")
        prediction_exports = [path for path in config["prediction_paths"] if path.exists()]
        record = {
            "key": config["key"],
            "label": config["label"],
            "family": config["family"],
            "manifest_path": manifest_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_exists": bool(checkpoint_path and Path(checkpoint_path).exists()),
            "prediction_exports": prediction_exports,
            "prediction_export_exists": bool(prediction_exports),
            "micro_f1": safe_float(metrics.get("micro_f1")),
            "macro_f1": safe_float(metrics.get("macro_f1")),
            "precision": safe_float(metrics.get("micro_precision")),
            "recall": safe_float(metrics.get("micro_recall")),
            "subset_accuracy": safe_float(metrics.get("subset_accuracy")),
        }
        if any(record.get(metric_name) is not None for metric_name in ["micro_f1", "macro_f1", "precision", "recall"]):
            records.append(record)
    records.sort(key=lambda item: (item["micro_f1"] is None, -(item["micro_f1"] or 0.0)))
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
        record["is_best"] = rank == 1
    return records, errors


def choose_primary_prediction_source(
    loaded_prediction_files: list[dict[str, Any]],
    benchmark_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not loaded_prediction_files:
        return None
    for item in loaded_prediction_files:
        if item["path"].name == "submission.json" and item["records"]:
            return item
    best_model = benchmark_records[0]["key"] if benchmark_records else None
    model_to_files = {
        "arabert": {"submission_arabert.json", "submission_hidden_arabert.json"},
        "marbert": {"submission_marbert.json", "submission_hidden_marbert.json"},
        "xlmr": {"submission_xlmr.json", "submission_hidden_xlmr.json"},
    }
    if best_model:
        for item in loaded_prediction_files:
            if item["path"].name in model_to_files.get(best_model, set()) and item["records"]:
                return item
    return max(loaded_prediction_files, key=lambda item: len(item["records"]))


def priority_color(priority: str) -> str:
    return {
        "Critical": COLORS["red"],
        "High": COLORS["orange"],
        "Medium": COLORS["yellow"],
        "Low": COLORS["green"],
        "Opportunity": COLORS["blue"],
    }.get(priority, COLORS["blue"])


def create_chip(parent: tk.Widget, text: str, bg_color: str, fg_color: str | None = None) -> tk.Label:
    return tk.Label(
        parent,
        text=text,
        bg=bg_color,
        fg=fg_color or COLORS["panel"],
        padx=10,
        pady=4,
        font=FONT_SMALL,
    )


def create_panel(parent: tk.Widget, padding: int = 18) -> tk.Frame:
    return tk.Frame(
        parent,
        bg=COLORS["panel"],
        highlightbackground=COLORS["border"],
        highlightthickness=1,
        padx=padding,
        pady=padding,
    )


def create_section_header(parent: tk.Widget, title: str, subtitle: str) -> None:
    tk.Label(parent, text=title, bg=COLORS["background"], fg=COLORS["text"], font=FONT_H1).pack(anchor="w")
    tk.Label(parent, text=subtitle, bg=COLORS["background"], fg=COLORS["muted"], font=FONT_BODY).pack(anchor="w", pady=(4, 14))


def create_business_kpi_card(
    parent: tk.Widget,
    title: str,
    value: str,
    subtitle: str,
    accent: str,
    chip_text: str | None = None,
) -> tk.Frame:
    frame = tk.Frame(
        parent,
        bg=COLORS["card"],
        highlightbackground=COLORS["border"],
        highlightthickness=1,
        padx=16,
        pady=16,
    )
    frame.grid_columnconfigure(1, weight=1)
    tk.Frame(frame, bg=accent, width=8, height=80).grid(row=0, column=0, rowspan=3, sticky="ns", padx=(0, 14))
    header = tk.Frame(frame, bg=COLORS["card"])
    header.grid(row=0, column=1, sticky="ew")
    tk.Label(header, text=title, bg=COLORS["card"], fg=COLORS["muted"], font=FONT_BODY).pack(side="left")
    if chip_text:
        create_chip(header, chip_text, accent).pack(side="right")
    tk.Label(frame, text=value, bg=COLORS["card"], fg=COLORS["text"], font=("Bahnschrift", 20, "bold")).grid(
        row=1,
        column=1,
        sticky="w",
        pady=(10, 6),
    )
    tk.Label(
        frame,
        text=subtitle,
        bg=COLORS["card"],
        fg=COLORS["muted"],
        font=FONT_SMALL,
        justify="left",
        wraplength=260,
    ).grid(row=2, column=1, sticky="w")
    return frame


def create_customer_needs_table(parent: tk.Widget, aspect_summary: list[dict[str, Any]]) -> tk.Frame:
    frame = create_panel(parent)
    tk.Label(frame, text="Customer Needs Analysis", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
    tk.Label(
        frame,
        text="Aspect-level demand, risk, and business meaning derived directly from predicted sentiments.",
        bg=COLORS["panel"],
        fg=COLORS["muted"],
        font=FONT_BODY,
    ).pack(anchor="w", pady=(4, 14))
    columns = (
        "Aspect",
        "Total Mentions",
        "Positive",
        "Negative",
        "Neutral",
        "Negative Rate",
        "Business Meaning",
        "Priority",
    )
    table_container = tk.Frame(frame, bg=COLORS["panel"])
    table_container.pack(fill="both", expand=True)
    tree = ttk.Treeview(table_container, columns=columns, show="headings", style="Dark.Treeview", height=12)
    widths = {
        "Aspect": 130,
        "Total Mentions": 110,
        "Positive": 80,
        "Negative": 80,
        "Neutral": 80,
        "Negative Rate": 110,
        "Business Meaning": 260,
        "Priority": 90,
    }
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=widths[column], anchor="w")
    scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    for item in aspect_summary:
        row_values = (
            item["aspect_label"],
            item["mentions"],
            item["positive"],
            item["negative"],
            item["neutral"],
            pct(item["negative_rate"]),
            item["business_meaning"],
            item["priority"],
        )
        tag = item["priority"].lower()
        tree.insert("", "end", values=row_values, tags=(tag,))
    tree.tag_configure("critical", foreground=COLORS["red"])
    tree.tag_configure("high", foreground=COLORS["orange"])
    tree.tag_configure("medium", foreground=COLORS["yellow"])
    tree.tag_configure("low", foreground=COLORS["green"])
    return frame


def create_priority_board(parent: tk.Widget, aspect_summary: list[dict[str, Any]]) -> tk.Frame:
    frame = create_panel(parent)
    tk.Label(frame, text="Pain Point Radar / Priority Board", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
    tk.Label(
        frame,
        text="A management-ready view of what needs urgent action, what should improve next, and what can be promoted.",
        bg=COLORS["panel"],
        fg=COLORS["muted"],
        font=FONT_BODY,
    ).pack(anchor="w", pady=(4, 14))
    board = tk.Frame(frame, bg=COLORS["panel"])
    board.pack(fill="both", expand=True)
    buckets = {
        "Critical Issues": {
            "color": COLORS["red"],
            "items": [item for item in aspect_summary if item["priority"] == "Critical"],
            "description": "High-volume problems already hurting satisfaction and trust.",
        },
        "High Priority Improvements": {
            "color": COLORS["orange"],
            "items": [item for item in aspect_summary if item["priority"] == "High"],
            "description": "Important weaknesses that should be assigned to the next execution cycle.",
        },
        "Strengths to Promote": {
            "color": COLORS["green"],
            "items": [
                item
                for item in aspect_summary
                if item["positive_rate"] >= 0.60 and item["mentions"] >= 2 and item["priority"] == "Low"
            ],
            "description": "Positive proof points that marketing and retention teams should amplify.",
        },
        "Stable Areas": {
            "color": COLORS["blue"],
            "items": [
                item
                for item in aspect_summary
                if item["priority"] in {"Medium", "Low"} and item["positive_rate"] < 0.60
            ],
            "description": "Signals that are currently manageable but still worth monitoring.",
        },
    }
    for index, (title, bucket) in enumerate(buckets.items()):
        card = tk.Frame(
            board,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
        board.grid_columnconfigure(index % 2, weight=1)
        board.grid_rowconfigure(index // 2, weight=1)
        header = tk.Frame(card, bg=COLORS["card"])
        header.pack(fill="x")
        tk.Label(header, text=title, bg=COLORS["card"], fg=COLORS["text"], font=FONT_H2).pack(side="left")
        create_chip(header, str(len(bucket["items"])), bucket["color"]).pack(side="right")
        tk.Label(
            card,
            text=bucket["description"],
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
            justify="left",
            wraplength=320,
        ).pack(anchor="w", pady=(8, 12))
        if not bucket["items"]:
            tk.Label(
                card,
                text="No major signals in this category right now.",
                bg=COLORS["card"],
                fg=COLORS["muted"],
                font=FONT_BODY,
            ).pack(anchor="w")
            continue
        for item in bucket["items"][:4]:
            text = (
                f"{item['aspect_label']}: {item['mentions']} mentions, "
                f"{pct(item['negative_rate'])} negative. {item['business_meaning']}."
            )
            tk.Label(
                card,
                text=text,
                bg=COLORS["card"],
                fg=COLORS["text"],
                font=FONT_BODY,
                justify="left",
                wraplength=320,
            ).pack(anchor="w", pady=3)
    return frame


def create_action_plan(parent: tk.Widget, recommendations: list[dict[str, Any]]) -> tk.Frame:
    frame = create_panel(parent)
    tk.Label(frame, text="Management Action Plan", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
    tk.Label(
        frame,
        text="Deterministic recommendations generated from aspect sentiment distribution, without any LLM post-processing.",
        bg=COLORS["panel"],
        fg=COLORS["muted"],
        font=FONT_BODY,
    ).pack(anchor="w", pady=(4, 14))
    grid = tk.Frame(frame, bg=COLORS["panel"])
    grid.pack(fill="both", expand=True)
    for index, item in enumerate(recommendations):
        card = tk.Frame(
            grid,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
        grid.grid_columnconfigure(index % 2, weight=1)
        header = tk.Frame(card, bg=COLORS["card"])
        header.pack(fill="x")
        tk.Label(header, text=item["title"], bg=COLORS["card"], fg=COLORS["text"], font=FONT_H2).pack(side="left")
        create_chip(header, item["priority"], priority_color(item["priority"])).pack(side="right")
        meta = f"Owner: {item['owner']}  |  Timeline: {item['timeline']}"
        tk.Label(card, text=meta, bg=COLORS["card"], fg=COLORS["muted"], font=FONT_SMALL).pack(anchor="w", pady=(8, 8))
        tk.Label(
            card,
            text=item["detail"],
            bg=COLORS["card"],
            fg=COLORS["text"],
            font=FONT_BODY,
            justify="left",
            wraplength=420,
        ).pack(anchor="w")
    return frame


def create_role_cards(parent: tk.Widget) -> tk.Frame:
    frame = create_panel(parent)
    tk.Label(frame, text="How a Business Team Uses DeepX", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
    tk.Label(
        frame,
        text="Each team sees the same ABSA signals through a role-specific decision lens.",
        bg=COLORS["panel"],
        fg=COLORS["muted"],
        font=FONT_BODY,
    ).pack(anchor="w", pady=(4, 14))
    grid = tk.Frame(frame, bg=COLORS["panel"])
    grid.pack(fill="both", expand=True)
    for index, (title, description, accent) in enumerate(ROLE_CARDS):
        card = tk.Frame(
            grid,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=16,
            pady=16,
        )
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
        grid.grid_columnconfigure(index % 2, weight=1)
        tk.Frame(card, bg=accent, height=6).pack(fill="x", pady=(0, 12))
        tk.Label(card, text=title, bg=COLORS["card"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
        tk.Label(
            card,
            text=description,
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=FONT_BODY,
            justify="left",
            wraplength=380,
        ).pack(anchor="w", pady=(8, 0))
    return frame


def create_message_panel(parent: tk.Widget, title: str, message: str, accent: str = COLORS["blue"]) -> tk.Frame:
    frame = create_panel(parent)
    tk.Frame(frame, bg=accent, height=6).pack(fill="x", pady=(0, 14))
    tk.Label(frame, text=title, bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
    tk.Label(frame, text=message, bg=COLORS["panel"], fg=COLORS["muted"], font=FONT_BODY, justify="left", wraplength=880).pack(anchor="w", pady=(8, 0))
    return frame


def create_animated_metric_bar(parent: tk.Widget, label: str, value: float | None, accent: str) -> None:
    row = tk.Frame(parent, bg=COLORS["card"])
    row.pack(fill="x", pady=5)
    tk.Label(row, text=label, bg=COLORS["card"], fg=COLORS["muted"], font=FONT_SMALL).pack(side="left")
    tk.Label(row, text=metric_text(value), bg=COLORS["card"], fg=COLORS["text"], font=FONT_SMALL).pack(side="right")
    bar = tk.Canvas(row, width=240, height=10, bg=COLORS["card"], highlightthickness=0)
    bar.pack(fill="x", pady=(6, 0))
    full_width = 240
    bar.create_rectangle(0, 0, full_width, 10, fill=COLORS["border"], outline="")
    fill = bar.create_rectangle(0, 0, 0, 10, fill=accent, outline="")
    target_ratio = max(0.0, min(1.0, value or 0.0))
    steps = 18

    def animate(step: int = 0) -> None:
        current_ratio = target_ratio * (step / steps)
        bar.coords(fill, 0, 0, full_width * current_ratio, 10)
        if step < steps:
            bar.after(18, animate, step + 1)

    animate()


def create_model_benchmark_section(parent: tk.Widget, benchmark_records: list[dict[str, Any]]) -> tk.Frame:
    frame = create_panel(parent)
    tk.Label(frame, text="Model Reliability & Benchmark", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
    tk.Label(
        frame,
        text="The best-ranked validation model produces the most reliable customer insight layer for the business dashboard.",
        bg=COLORS["panel"],
        fg=COLORS["muted"],
        font=FONT_BODY,
    ).pack(anchor="w", pady=(4, 14))
    if not benchmark_records:
        tk.Label(
            frame,
            text="No saved benchmark manifests were found. Add training manifests to compare model reliability.",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_BODY,
        ).pack(anchor="w")
        return frame
    best_model = benchmark_records[0]
    recommendation = (
        f"Use {best_model['label']} for this run because it has the highest validation micro-F1 "
        f"({pct(best_model['micro_f1'])})."
    )
    create_chip(frame, "Benchmark Recommendation", COLORS["blue"]).pack(anchor="w")
    tk.Label(frame, text=recommendation, bg=COLORS["panel"], fg=COLORS["text"], font=FONT_BODY).pack(anchor="w", pady=(8, 18))
    cards = tk.Frame(frame, bg=COLORS["panel"])
    cards.pack(fill="x")
    metric_colors = {
        "Micro F1": COLORS["blue"],
        "Macro F1": COLORS["purple"],
        "Precision": COLORS["green"],
        "Recall": COLORS["yellow"],
        "Subset Accuracy": COLORS["orange"],
    }
    for index, record in enumerate(benchmark_records):
        card = tk.Frame(
            cards,
            bg=COLORS["card"],
            highlightbackground=priority_color("Opportunity" if record["is_best"] else "Low"),
            highlightthickness=2 if record["is_best"] else 1,
            padx=16,
            pady=16,
        )
        card.grid(row=0, column=index, sticky="nsew", padx=8, pady=8)
        cards.grid_columnconfigure(index, weight=1)
        header = tk.Frame(card, bg=COLORS["card"])
        header.pack(fill="x")
        tk.Label(card, text=record["label"], bg=COLORS["card"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
        tk.Label(card, text=record["family"], bg=COLORS["card"], fg=COLORS["muted"], font=FONT_SMALL).pack(anchor="w", pady=(3, 10))
        badge_frame = tk.Frame(card, bg=COLORS["card"])
        badge_frame.pack(fill="x", pady=(0, 10))
        create_chip(badge_frame, f"Rank #{record['rank']}", COLORS["purple"]).pack(side="left", padx=(0, 6))
        if record["is_best"]:
            create_chip(badge_frame, "Most Reliable", COLORS["green"]).pack(side="left", padx=(0, 6))
        create_chip(
            badge_frame,
            "Checkpoint Ready" if record["checkpoint_exists"] else "Checkpoint Missing",
            COLORS["green"] if record["checkpoint_exists"] else COLORS["red"],
        ).pack(side="left")
        create_animated_metric_bar(card, "Micro F1", record["micro_f1"], metric_colors["Micro F1"])
        create_animated_metric_bar(card, "Macro F1", record["macro_f1"], metric_colors["Macro F1"])
        create_animated_metric_bar(card, "Precision", record["precision"], metric_colors["Precision"])
        create_animated_metric_bar(card, "Recall", record["recall"], metric_colors["Recall"])
        create_animated_metric_bar(card, "Subset Accuracy", record["subset_accuracy"], metric_colors["Subset Accuracy"])
        export_text = ", ".join(path.name for path in record["prediction_exports"]) if record["prediction_exports"] else "No model-specific export"
        tk.Label(
            card,
            text=f"Prediction export: {export_text}",
            bg=COLORS["card"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
            justify="left",
            wraplength=280,
        ).pack(anchor="w", pady=(10, 0))
    table_frame = tk.Frame(frame, bg=COLORS["panel"])
    table_frame.pack(fill="both", expand=True, pady=(14, 0))
    columns = ("Model", "Micro F1", "Macro F1", "Precision", "Recall", "Subset Accuracy", "Status")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Dark.Treeview", height=6)
    for column, width in {
        "Model": 130,
        "Micro F1": 95,
        "Macro F1": 95,
        "Precision": 95,
        "Recall": 95,
        "Subset Accuracy": 110,
        "Status": 240,
    }.items():
        tree.heading(column, text=column)
        tree.column(column, width=width, anchor="w")
    scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    for record in benchmark_records:
        status_bits = []
        status_bits.append("best" if record["is_best"] else "validated")
        status_bits.append("checkpoint ready" if record["checkpoint_exists"] else "checkpoint missing")
        status_bits.append("prediction export found" if record["prediction_export_exists"] else "no model-specific export")
        tree.insert(
            "",
            "end",
            values=(
                record["label"],
                metric_text(record["micro_f1"]),
                metric_text(record["macro_f1"]),
                metric_text(record["precision"]),
                metric_text(record["recall"]),
                metric_text(record["subset_accuracy"]),
                ", ".join(status_bits),
            ),
        )
    return frame


class ScrollableTab(tk.Frame):
    def __init__(self, master: tk.Widget, **kwargs: Any) -> None:
        super().__init__(master, bg=COLORS["background"], **kwargs)
        self.canvas = tk.Canvas(
            self,
            bg=COLORS["background"],
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg=COLORS["background"])
        self.content.bind("<Configure>", self._on_frame_configure)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

    def _on_frame_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self.winfo_ismapped():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class DeepXDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DeepX - AI Customer Needs Intelligence")
        self.geometry("1440x940")
        self.minsize(1220, 820)
        self.configure(bg=COLORS["background"])
        self.data_state: dict[str, Any] = {}
        self.status_text = tk.StringVar(value="Loading business signals...")
        self._pulse_index = 0
        self._configure_styles()
        self._build_shell()
        self.refresh_dashboard()
        self._pulse_title()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=COLORS["panel"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            padding=(18, 10),
            font=FONT_BODY,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["card"])],
            foreground=[("selected", COLORS["text"])],
        )
        style.configure(
            "Dark.Treeview",
            background=COLORS["card"],
            fieldbackground=COLORS["card"],
            foreground=COLORS["text"],
            rowheight=30,
            font=FONT_SMALL,
            borderwidth=0,
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=FONT_SMALL,
            relief="flat",
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", COLORS["blue"])],
            foreground=[("selected", COLORS["panel"])],
        )
        style.configure(
            "Vertical.TScrollbar",
            background=COLORS["card"],
            troughcolor=COLORS["panel"],
            arrowcolor=COLORS["muted"],
            bordercolor=COLORS["panel"],
        )

    def _build_shell(self) -> None:
        header = tk.Frame(self, bg=COLORS["panel"], padx=26, pady=22)
        header.pack(fill="x")
        left = tk.Frame(header, bg=COLORS["panel"])
        left.pack(side="left", fill="x", expand=True)
        self.title_label = tk.Label(left, text="DeepX", bg=COLORS["panel"], fg=COLORS["blue"], font=FONT_TITLE)
        self.title_label.pack(anchor="w")
        tk.Label(
            left,
            text="AI Customer Needs Intelligence for Arabic and Multilingual Reviews",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=FONT_H2,
        ).pack(anchor="w", pady=(4, 2))
        tk.Label(
            left,
            text="Arabic ABSA for business decision makers",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_BODY,
        ).pack(anchor="w")
        self.header_chips = tk.Frame(left, bg=COLORS["panel"])
        self.header_chips.pack(anchor="w", pady=(14, 0))
        right = tk.Frame(header, bg=COLORS["panel"])
        right.pack(side="right", anchor="ne")
        tk.Label(right, textvariable=self.status_text, bg=COLORS["panel"], fg=COLORS["muted"], font=FONT_SMALL).pack(anchor="e", pady=(0, 10))
        self.refresh_button = tk.Button(
            right,
            text="Refresh Data",
            command=self.refresh_dashboard,
            bg=COLORS["blue"],
            fg=COLORS["panel"],
            activebackground=COLORS["purple"],
            activeforeground=COLORS["text"],
            relief="flat",
            padx=16,
            pady=10,
            font=FONT_BODY,
            cursor="hand2",
        )
        self.refresh_button.pack(anchor="e")
        notebook_host = tk.Frame(self, bg=COLORS["background"], padx=20, pady=16)
        notebook_host.pack(fill="both", expand=True)
        self.notebook = ttk.Notebook(notebook_host)
        self.notebook.pack(fill="both", expand=True)
        self.tabs: dict[str, ScrollableTab] = {}
        for key, label in [
            ("overview", "Business Overview"),
            ("needs", "Customer Needs"),
            ("action", "Action Plan"),
            ("benchmark", "Model Benchmark"),
        ]:
            tab = ScrollableTab(self.notebook)
            self.tabs[key] = tab
            self.notebook.add(tab, text=label)
        self.footer = tk.Label(
            self,
            text="DeepX turns multilingual customer reviews into business actions.",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
            padx=22,
            pady=10,
        )
        self.footer.pack(fill="x", side="bottom")

    def _pulse_title(self) -> None:
        palette = [COLORS["blue"], COLORS["purple"], COLORS["blue"], COLORS["green"]]
        self.title_label.configure(fg=palette[self._pulse_index % len(palette)])
        self._pulse_index += 1
        self.after(750, self._pulse_title)

    def refresh_dashboard(self) -> None:
        self.refresh_button.configure(state="disabled")
        self.status_text.set("Refreshing business insights and benchmark data...")
        self.update_idletasks()
        self.data_state = self._load_dashboard_state()
        self._render_dashboard()
        self.status_text.set("Dashboard ready")
        self.refresh_button.configure(state="normal")

    def _load_dashboard_state(self) -> dict[str, Any]:
        benchmark_records, benchmark_errors = load_model_benchmark_records()
        prediction_errors: list[str] = []
        loaded_prediction_files: list[dict[str, Any]] = []
        for path in discover_prediction_files():
            records, error = load_prediction_file(path)
            if error:
                prediction_errors.append(error)
                continue
            loaded_prediction_files.append({"path": path, "records": records})
        active_prediction_source = choose_primary_prediction_source(loaded_prediction_files, benchmark_records)
        active_predictions = active_prediction_source["records"] if active_prediction_source else []
        reviews_df, review_paths, review_errors = load_review_dataframe()
        business_insights = compute_business_insights(active_predictions, reviews_df)
        source_message = (
            f"Business insights are using {active_prediction_source['path'].name}."
            if active_prediction_source
            else "Prediction files were not found. The dashboard is in benchmark-only mode."
        )
        best_model = benchmark_records[0] if benchmark_records else None
        return {
            "benchmark_records": benchmark_records,
            "benchmark_errors": benchmark_errors,
            "prediction_errors": prediction_errors,
            "loaded_prediction_files": loaded_prediction_files,
            "active_prediction_source": active_prediction_source,
            "review_paths": review_paths,
            "review_errors": review_errors,
            "reviews_df": reviews_df,
            "business_insights": business_insights,
            "source_message": source_message,
            "best_model": best_model,
        }

    def _render_dashboard(self) -> None:
        for tab in self.tabs.values():
            for child in tab.content.winfo_children():
                child.destroy()
        self._render_header_chips()
        self._render_overview_tab()
        self._render_customer_needs_tab()
        self._render_action_plan_tab()
        self._render_benchmark_tab()

    def _render_header_chips(self) -> None:
        for child in self.header_chips.winfo_children():
            child.destroy()
        loaded_files = self.data_state.get("loaded_prediction_files", [])
        review_paths = self.data_state.get("review_paths", [])
        benchmark_records = self.data_state.get("benchmark_records", [])
        active_prediction_source = self.data_state.get("active_prediction_source")
        chip_specs = [
            (
                f"Insight Source: {active_prediction_source['path'].name}" if active_prediction_source else "Insight Source: benchmark only",
                COLORS["blue"],
            ),
            (
                f"Review Text: {len(review_paths)} file(s) connected" if review_paths else "Review Text: unavailable",
                COLORS["green"] if review_paths else COLORS["orange"],
            ),
            (
                f"Prediction Files: {len(loaded_files)} found" if loaded_files else "Prediction Files: missing",
                COLORS["purple"] if loaded_files else COLORS["red"],
            ),
            (
                f"Models Benchmarked: {len(benchmark_records)}",
                COLORS["yellow"],
            ),
        ]
        for text, color in chip_specs:
            create_chip(self.header_chips, text, color).pack(side="left", padx=(0, 8))

    def _render_overview_tab(self) -> None:
        tab = self.tabs["overview"].content
        tab.configure(padx=6, pady=6)
        create_section_header(
            tab,
            "Executive Header",
            "DeepX turns ABSA predictions into customer pain points, satisfaction drivers, and next actions for business teams.",
        )
        insights = self.data_state["business_insights"]
        kpis = insights["kpis"]
        best_model = self.data_state.get("best_model")
        executive = create_panel(tab, padding=20)
        executive.pack(fill="x", pady=(0, 16))
        tk.Label(
            executive,
            text="DeepX",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Bahnschrift", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            executive,
            text="AI Customer Needs Intelligence",
            bg=COLORS["panel"],
            fg=COLORS["blue"],
            font=FONT_H2,
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            executive,
            text="Arabic ABSA for business decision makers",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_BODY,
        ).pack(anchor="w", pady=(0, 12))
        meta_grid = tk.Frame(executive, bg=COLORS["panel"])
        meta_grid.pack(fill="x")
        executive_metrics = [
            ("Total reviews analyzed", str(insights["total_reviews"])),
            ("Total detected customer needs", str(kpis["total_aspect_mentions"])),
            ("Dominant customer pain point", kpis["dominant_pain_point"]),
            ("Strongest customer satisfaction driver", kpis["strongest_satisfaction_driver"]),
        ]
        for index, (label, value) in enumerate(executive_metrics):
            card = tk.Frame(
                meta_grid,
                bg=COLORS["card"],
                highlightbackground=COLORS["border"],
                highlightthickness=1,
                padx=16,
                pady=16,
            )
            card.grid(row=0, column=index, sticky="nsew", padx=6, pady=6)
            meta_grid.grid_columnconfigure(index, weight=1)
            tk.Label(card, text=label, bg=COLORS["card"], fg=COLORS["muted"], font=FONT_SMALL).pack(anchor="w")
            tk.Label(card, text=value, bg=COLORS["card"], fg=COLORS["text"], font=FONT_H2, wraplength=250, justify="left").pack(anchor="w", pady=(8, 0))
        tk.Label(
            executive,
            text=self.data_state["source_message"],
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_BODY,
        ).pack(anchor="w", pady=(14, 0))
        if best_model:
            tk.Label(
                executive,
                text=(
                    f"Most reliable benchmark model: {best_model['label']} "
                    f"with validation micro-F1 of {pct(best_model['micro_f1'])}."
                ),
                bg=COLORS["panel"],
                fg=COLORS["text"],
                font=FONT_BODY,
            ).pack(anchor="w", pady=(6, 0))
        kpi_section = tk.Frame(tab, bg=COLORS["background"])
        kpi_section.pack(fill="x", pady=(0, 16))
        kpi_items = [
            (
                "Customer Satisfaction Signal",
                pct(kpis["satisfaction_score"]),
                "Share of positive aspect mentions across all detected needs.",
                COLORS["green"],
                "Positive",
            ),
            (
                "Negative Review Risk",
                pct(kpis["risk_score"]),
                "Share of negative aspect mentions that may hurt retention.",
                COLORS["red"],
                kpis["urgency"],
            ),
            (
                "Most Mentioned Aspect",
                kpis["most_mentioned_aspect"]["aspect_label"] if kpis["most_mentioned_aspect"] else "N/A",
                "The issue customers talk about most often.",
                COLORS["blue"],
                None,
            ),
            (
                "Most Negative Aspect",
                kpis["most_negative_aspect"]["aspect_label"] if kpis["most_negative_aspect"] else "N/A",
                "The sharpest pain point based on negative rate and volume.",
                COLORS["orange"],
                None,
            ),
            (
                "Most Positive Aspect",
                kpis["most_positive_aspect"]["aspect_label"] if kpis["most_positive_aspect"] else "N/A",
                "The strongest satisfaction driver to protect or promote.",
                COLORS["purple"],
                None,
            ),
            (
                "Neutral / unclear feedback rate",
                pct(kpis["neutral_rate"]),
                "Signals that need better operational clarity or more feedback.",
                COLORS["yellow"],
                None,
            ),
            (
                "Number of urgent issues",
                str(kpis["urgent_issue_count"]),
                "Aspects currently marked Critical or High priority.",
                COLORS["red"],
                None,
            ),
            (
                "Recommended priority",
                kpis["recommended_priority"],
                "Management urgency derived from the overall negative rate.",
                priority_color(kpis["recommended_priority"]),
                None,
            ),
        ]
        for index, item in enumerate(kpi_items):
            card = create_business_kpi_card(kpi_section, *item)
            card.grid(row=index // 4, column=index % 4, sticky="nsew", padx=8, pady=8)
            kpi_section.grid_columnconfigure(index % 4, weight=1)
        if not self.data_state.get("loaded_prediction_files"):
            create_message_panel(
                tab,
                "Benchmark-only mode",
                "Business insights need prediction files. Connect any submission export to activate customer-needs analytics.",
                COLORS["orange"],
            ).pack(fill="x", pady=(0, 16))
        if self.data_state["prediction_errors"] or self.data_state["review_errors"]:
            details = self.data_state["prediction_errors"] + self.data_state["review_errors"]
            create_message_panel(
                tab,
                "Data loading notes",
                "\n".join(details[:5]),
                COLORS["yellow"],
            ).pack(fill="x", pady=(0, 16))
        create_role_cards(tab).pack(fill="x", pady=(0, 16))

    def _render_customer_needs_tab(self) -> None:
        tab = self.tabs["needs"].content
        tab.configure(padx=6, pady=6)
        create_section_header(
            tab,
            "Customer Needs",
            "Understand what customers complain about, what they love, and which aspects deserve intervention first.",
        )
        insights = self.data_state["business_insights"]
        aspect_summary = insights["aspect_summary"]
        if not aspect_summary:
            create_message_panel(
                tab,
                "Business insights unavailable",
                "No prediction records are currently available. The model benchmark section below still shows reliability information.",
                COLORS["orange"],
            ).pack(fill="x", pady=(0, 16))
            return
        create_customer_needs_table(tab, aspect_summary).pack(fill="both", expand=True, pady=(0, 16))
        create_priority_board(tab, aspect_summary).pack(fill="x", pady=(0, 16))
        voice_panel = create_panel(tab)
        voice_panel.pack(fill="x", pady=(0, 16))
        tk.Label(voice_panel, text="Customer Voice", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
        if not insights["review_text_available"]:
            tk.Label(
                voice_panel,
                text="Review text not found. Connect processed review file to enable customer voice examples.",
                bg=COLORS["panel"],
                fg=COLORS["muted"],
                font=FONT_BODY,
            ).pack(anchor="w", pady=(8, 0))
            return
        tk.Label(
            voice_panel,
            text="Representative reviews selected through aspect and sentiment matching.",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_BODY,
        ).pack(anchor="w", pady=(4, 12))
        grids = tk.Frame(voice_panel, bg=COLORS["panel"])
        grids.pack(fill="x")
        voice_sections = [
            ("Top positive review signals", insights["customer_voice"].get("positive", []), COLORS["green"]),
            ("Top negative review signals", insights["customer_voice"].get("negative", []), COLORS["red"]),
            (
                f"Reviews related to {insights['kpis']['dominant_pain_point']}",
                insights["most_negative_aspect_examples"],
                COLORS["orange"],
            ),
        ]
        for index, (title, examples, accent) in enumerate(voice_sections):
            card = tk.Frame(
                grids,
                bg=COLORS["card"],
                highlightbackground=COLORS["border"],
                highlightthickness=1,
                padx=16,
                pady=16,
            )
            card.grid(row=0, column=index, sticky="nsew", padx=8, pady=8)
            grids.grid_columnconfigure(index, weight=1)
            tk.Frame(card, bg=accent, height=6).pack(fill="x", pady=(0, 12))
            tk.Label(card, text=title, bg=COLORS["card"], fg=COLORS["text"], font=FONT_H2, wraplength=320, justify="left").pack(anchor="w")
            if not examples:
                tk.Label(card, text="No matching reviews found yet.", bg=COLORS["card"], fg=COLORS["muted"], font=FONT_BODY).pack(anchor="w", pady=(10, 0))
                continue
            for example in examples:
                business_name = example.get("business_name") or "Unknown business"
                aspects = ", ".join(
                    f"{aspect}: {sentiment}"
                    for aspect, sentiment in list(example.get("aspect_sentiments", {}).items())[:3]
                )
                tk.Label(
                    card,
                    text=short_text(example.get("review_text"), limit=210),
                    bg=COLORS["card"],
                    fg=COLORS["text"],
                    font=FONT_BODY,
                    justify="left",
                    wraplength=320,
                ).pack(anchor="w", pady=(10, 4))
                meta = f"Review {example.get('review_id')} | {business_name} | {aspects}"
                tk.Label(card, text=short_text(meta, limit=120), bg=COLORS["card"], fg=COLORS["muted"], font=FONT_SMALL, justify="left", wraplength=320).pack(anchor="w")

    def _render_action_plan_tab(self) -> None:
        tab = self.tabs["action"].content
        tab.configure(padx=6, pady=6)
        create_section_header(
            tab,
            "Action Plan",
            "Move from sentiment signals to management actions with operational, pricing, product, and marketing guidance.",
        )
        insights = self.data_state["business_insights"]
        kpis = insights["kpis"]
        summary_panel = create_panel(tab)
        summary_panel.pack(fill="x", pady=(0, 16))
        tk.Label(summary_panel, text="Recommended next move", bg=COLORS["panel"], fg=COLORS["text"], font=FONT_H2).pack(anchor="w")
        tk.Label(
            summary_panel,
            text=(
                f"Current management priority: {kpis['recommended_priority']}. "
                f"Negative risk is {pct(kpis['risk_score'])}, and there are {kpis['urgent_issue_count']} urgent issue(s)."
            ),
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_BODY,
            wraplength=980,
            justify="left",
        ).pack(anchor="w", pady=(8, 10))
        strongest_driver = kpis["strongest_satisfaction_driver"]
        tk.Label(
            summary_panel,
            text=f"Protect the strongest satisfaction driver: {strongest_driver}. Use it as a reference point while fixing weaker areas.",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=FONT_BODY,
            wraplength=980,
            justify="left",
        ).pack(anchor="w")
        create_action_plan(tab, insights["recommendations"]).pack(fill="x", pady=(0, 16))
        create_role_cards(tab).pack(fill="x", pady=(0, 16))

    def _render_benchmark_tab(self) -> None:
        tab = self.tabs["benchmark"].content
        tab.configure(padx=6, pady=6)
        create_section_header(
            tab,
            "Model Benchmark",
            "Keep the model comparison visible so business teams can trust how customer insights are generated.",
        )
        create_model_benchmark_section(tab, self.data_state["benchmark_records"]).pack(fill="x", pady=(0, 16))
        if self.data_state["benchmark_errors"]:
            create_message_panel(
                tab,
                "Benchmark loading notes",
                "\n".join(self.data_state["benchmark_errors"][:5]),
                COLORS["yellow"],
            ).pack(fill="x", pady=(0, 16))


def main() -> None:
    app = DeepXDashboard()
    if os.environ.get("DEEPX_SMOKE_TEST"):
        app.after(1200, app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
