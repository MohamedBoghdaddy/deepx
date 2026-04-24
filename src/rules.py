"""
Rule-based post-processing and explainability utilities for Arabic ABSA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dataset import (
    ASPECT_SENTIMENT_LABELS,
    LABEL_TO_IDX,
    NON_NONE_ASPECTS,
    OUTPUTS_ROOT,
    VALID_ASPECTS,
    VALID_SENTIMENTS,
    create_multi_label_vector,
    sanitize_aspect_sentiments,
)
from preprocess import ArabicPreprocessor, get_rule_preprocessor


DEFAULT_THRESHOLD_PATH = OUTPUTS_ROOT / "best_thresholds.json"
DEFAULT_GLOBAL_THRESHOLD = 0.5
LOCAL_SENTIMENT_WINDOW = 4

RAW_ASPECT_KEYWORDS = {
    "food": [
        "اكل",
        "الاكل",
        "أكل",
        "طعم",
        "طعمه",
        "نكهه",
        "وجبه",
        "وجبات",
        "ساندوتش",
        "ساندويتش",
        "برجر",
        "بيتزا",
        "قهوه",
        "مشروب",
        "حلويات",
        "حلو",
        "طعام",
        "meal",
        "food",
        "taste",
        "burger",
        "pizza",
        "coffee",
        "dessert",
    ],
    "service": [
        "خدمه",
        "الخدمه",
        "موظف",
        "موظفين",
        "استقبال",
        "تعامل",
        "الكاشير",
        "ويتر",
        "النادل",
        "طاقم",
        "service",
        "staff",
        "employees",
        "waiter",
        "support",
    ],
    "price": [
        "سعر",
        "السعر",
        "اسعار",
        "غالي",
        "مره غالي",
        "رخيص",
        "قيمه",
        "قيمة",
        "مبالغ",
        "خصم",
        "price",
        "prices",
        "expensive",
        "cheap",
        "overpriced",
    ],
    "cleanliness": [
        "نظيف",
        "نضافه",
        "نظافه",
        "نظافة",
        "وصخ",
        "وسخ",
        "قذر",
        "حمام",
        "الحمام",
        "دورات المياه",
        "toilet",
        "toilets",
        "bathroom",
        "bathrooms",
        "clean",
        "dirty",
        "propre",
    ],
    "delivery": [
        "توصيل",
        "الدليفري",
        "دليفري",
        "طلب",
        "الطلب",
        "مندوب",
        "السائق",
        "وصل",
        "التسليم",
        "delivery",
        "driver",
        "shipping",
        "courier",
        "arrival",
    ],
    "ambiance": [
        "اجواء",
        "أجواء",
        "جو",
        "ديكور",
        "قعده",
        "قعدة",
        "جلسه",
        "جلسة",
        "مكان",
        "المكان",
        "الموسيقى",
        "اضاءه",
        "هدوء",
        "زحمه",
        "زحمة",
        "vibe",
        "atmosphere",
        "ambiance",
        "decor",
    ],
    "app_experience": [
        "تطبيق",
        "التطبيق",
        "ابلكيشن",
        "الموقع",
        "سيستم",
        "الدفع",
        "بحث",
        "الخريطه",
        "الخريطة",
        "تسجيل",
        "دخول",
        "واجهة",
        "يوزر",
        "checkout",
        "login",
        "search",
        "map",
        "maps",
        "payment",
        "app",
        "application",
        "website",
        "site",
    ],
    "general": [
        "تجربه",
        "تجربة",
        "بصراحه",
        "بصراحة",
        "عموما",
        "بشكل عام",
        "مكان",
        "المطعم",
        "الخدمه كلها",
        "everything",
        "overall",
        "experience",
        "branch",
        "store",
        "restaurant",
    ],
}

RAW_SENTIMENT_LEXICONS = {
    "positive": [
        "ممتاز",
        "رائع",
        "تحفه",
        "تحفة",
        "جميل",
        "حلو",
        "لذيذ",
        "طيب",
        "نظيف",
        "نضيف",
        "مرتب",
        "سريع",
        "احترافي",
        "فوق الممتاز",
        "روعة",
        "يجنن",
        "يعجب",
        "مريح",
        "awesome",
        "great",
        "excellent",
        "good",
        "clean",
        "perfect",
        "best",
        "EMO_POS",
    ],
    "negative": [
        "سيء",
        "سيئ",
        "سيئه",
        "سيئة",
        "وحش",
        "رديء",
        "زفت",
        "مقرف",
        "خايس",
        "بارد",
        "محروق",
        "غالي",
        "وسخ",
        "وصخ",
        "قذر",
        "بطيء",
        "بطيء",
        "متاخر",
        "متأخر",
        "زحمه",
        "زحمة",
        "مشكله",
        "مشكلة",
        "bad",
        "worst",
        "dirty",
        "expensive",
        "slow",
        "EMO_NEG",
    ],
    "neutral": [
        "عادي",
        "مقبول",
        "متوسط",
        "اوكي",
        "اوك",
        "ok",
        "normal",
        "neutral",
        "EMO_NEU",
    ],
}

NEGATION_WORDS = {
    "مو",
    "مش",
    "ما",
    "ماهو",
    "ماهي",
    "مافي",
    "مب",
    "موب",
    "بدون",
    "غير",
    "ليس",
    "not",
    "no",
    "never",
}
SARCASM_MARKERS = {
    "هههه",
    "ههه",
    "يا سلام",
    "اكيد",
    "أكيد",
    "طبعا",
    "طبعا",
    "lol",
    "lmao",
}
CONTRAST_MARKERS = {"لكن", "بس", "مع ان", "مع إن", "بالرغم", "رغم"}
EMOJI_SENTIMENT = {"EMO_POS": "positive", "EMO_NEG": "negative", "EMO_NEU": "neutral"}

RULE_PREPROCESSOR = get_rule_preprocessor()


def _normalize_terms(values: Sequence[str]) -> List[str]:
    normalized = []
    for value in values:
        normalized_value = RULE_PREPROCESSOR.normalize(str(value)).strip()
        if normalized_value and normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


ASPECT_KEYWORDS = {
    aspect: _normalize_terms(terms)
    for aspect, terms in RAW_ASPECT_KEYWORDS.items()
}
SENTIMENT_LEXICONS = {
    sentiment: _normalize_terms(terms)
    for sentiment, terms in RAW_SENTIMENT_LEXICONS.items()
}
NORMALIZED_NEGATION_WORDS = set(_normalize_terms(list(NEGATION_WORDS)))
NORMALIZED_SARCASM_MARKERS = set(_normalize_terms(list(SARCASM_MARKERS)))
NORMALIZED_CONTRAST_MARKERS = set(_normalize_terms(list(CONTRAST_MARKERS)))


@dataclass
class RuleFeatures:
    """Cached rule features for a single text sample."""

    original_text: str
    normalized_text: str
    tokens: List[str]
    emoji_tokens: List[str]
    emoji_names: List[str]
    aspect_matches: Dict[str, List[str]]
    aspect_positions: Dict[str, List[int]]
    global_sentiment_terms: Dict[str, List[str]]
    aspect_sentiment_terms: Dict[str, Dict[str, List[str]]]
    sarcasm_candidate: bool
    sentiment_conflict: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class PredictionDecision:
    """Schema-safe prediction enriched with rule explanations."""

    aspects: List[str]
    aspect_sentiments: Dict[str, str]
    label_probabilities: Dict[str, float]
    aspect_probabilities: Dict[str, float]
    sentiment_probabilities: Dict[str, Dict[str, float]]
    matched_keywords: Dict[str, List[str]]
    matched_sentiment_terms: Dict[str, Dict[str, List[str]]]
    emoji_tokens: List[str]
    emoji_names: List[str]
    rule_decisions: List[str]
    sarcasm_candidate: bool
    sentiment_conflict: bool
    prediction_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decision for JSON output."""
        return asdict(self)


def load_threshold_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load tuned thresholds when available, otherwise return a safe default."""
    resolved_path = path or DEFAULT_THRESHOLD_PATH
    if resolved_path and Path(resolved_path).exists():
        with Path(resolved_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "thresholds" in data:
            return {
                "global_threshold": float(data.get("global_threshold", DEFAULT_GLOBAL_THRESHOLD)),
                "thresholds": {str(key): float(value) for key, value in data.get("thresholds", {}).items()},
                "aspect_thresholds": {
                    str(key): float(value)
                    for key, value in data.get("aspect_thresholds", {}).items()
                },
            }
        if isinstance(data, dict) and any(key in ASPECT_SENTIMENT_LABELS for key in data):
            return {
                "global_threshold": DEFAULT_GLOBAL_THRESHOLD,
                "thresholds": {str(key): float(value) for key, value in data.items()},
                "aspect_thresholds": {},
            }
    return {
        "global_threshold": DEFAULT_GLOBAL_THRESHOLD,
        "thresholds": {},
        "aspect_thresholds": {},
    }


def resolve_label_threshold(threshold_config: Mapping[str, Any], label_name: str) -> float:
    """Resolve the active threshold for a specific aspect-sentiment label."""
    thresholds = threshold_config.get("thresholds", {})
    if label_name in thresholds:
        return float(thresholds[label_name])

    aspect_name = label_name.rsplit("_", 1)[0]
    aspect_thresholds = threshold_config.get("aspect_thresholds", {})
    if aspect_name in aspect_thresholds:
        return float(aspect_thresholds[aspect_name])

    return float(threshold_config.get("global_threshold", DEFAULT_GLOBAL_THRESHOLD))


def resolve_aspect_threshold(threshold_config: Mapping[str, Any], aspect_name: str) -> float:
    """Resolve an aspect-level threshold from aspect or label thresholds."""
    aspect_thresholds = threshold_config.get("aspect_thresholds", {})
    if aspect_name in aspect_thresholds:
        return float(aspect_thresholds[aspect_name])

    label_thresholds = [
        resolve_label_threshold(threshold_config, f"{aspect_name}_{sentiment}")
        for sentiment in VALID_SENTIMENTS
    ]
    return float(sum(label_thresholds) / len(label_thresholds))


def build_probability_maps(label_probabilities: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Convert the flat probability vector into label-, aspect-, and sentiment-level maps."""
    probabilities = np.asarray(label_probabilities, dtype=np.float32)
    label_probability_map = {
        label_name: round(float(probabilities[index]), 6)
        for index, label_name in enumerate(ASPECT_SENTIMENT_LABELS)
    }
    aspect_probabilities: Dict[str, float] = {}
    sentiment_probabilities: Dict[str, Dict[str, float]] = {}

    for aspect in VALID_ASPECTS:
        per_sentiment = {
            sentiment: float(probabilities[LABEL_TO_IDX[f"{aspect}_{sentiment}"]])
            for sentiment in VALID_SENTIMENTS
        }
        sentiment_probabilities[aspect] = {
            sentiment: round(score, 6)
            for sentiment, score in per_sentiment.items()
        }
        aspect_probabilities[aspect] = round(max(per_sentiment.values()), 6)

    return label_probability_map, aspect_probabilities, sentiment_probabilities


def find_term_positions(tokens: Sequence[str], term: str) -> List[int]:
    """Find all starting token positions for a term or phrase."""
    term_tokens = term.split()
    if not term_tokens or len(term_tokens) > len(tokens):
        return []
    positions = []
    for index in range(len(tokens) - len(term_tokens) + 1):
        if list(tokens[index : index + len(term_tokens)]) == term_tokens:
            positions.append(index)
    return positions


def has_negation(tokens: Sequence[str], start_index: int, window: int = 3) -> bool:
    """Check whether a sentiment term is locally negated."""
    left = max(0, start_index - window)
    context = tokens[left:start_index]
    return any(token in NORMALIZED_NEGATION_WORDS for token in context)


def flip_sentiment(sentiment: str) -> str:
    """Swap positive and negative under negation."""
    if sentiment == "positive":
        return "negative"
    if sentiment == "negative":
        return "positive"
    return sentiment


def collect_sentiment_terms(
    tokens: Sequence[str],
    anchor_positions: Optional[Sequence[int]] = None,
    window: int = LOCAL_SENTIMENT_WINDOW,
) -> Dict[str, List[str]]:
    """Collect matched sentiment terms globally or near aspect keyword anchors."""
    matched = {sentiment: [] for sentiment in VALID_SENTIMENTS}

    for sentiment, lexicon in SENTIMENT_LEXICONS.items():
        for term in lexicon:
            positions = find_term_positions(tokens, term)
            for position in positions:
                if anchor_positions:
                    min_distance = min(abs(position - anchor) for anchor in anchor_positions)
                    if min_distance > window:
                        continue
                effective_sentiment = flip_sentiment(sentiment) if has_negation(tokens, position) else sentiment
                matched[effective_sentiment].append(term)

    return matched


def detect_sarcasm(normalized_text: str, emoji_names: Sequence[str], global_terms: Mapping[str, Sequence[str]]) -> bool:
    """Detect sarcasm candidates using laughter markers plus literal polarity cues."""
    normalized_lower = normalized_text.lower()
    has_marker = any(marker in normalized_lower for marker in NORMALIZED_SARCASM_MARKERS)
    has_marker = has_marker or any(
        ArabicPreprocessor.is_sarcasm_emoji(emoji_name)
        for emoji_name in emoji_names
    )
    literal_polarity = bool(global_terms.get("positive") or global_terms.get("negative"))
    return has_marker and literal_polarity


def extract_rule_features(
    text: str,
    preprocessor: Optional[ArabicPreprocessor] = None,
) -> RuleFeatures:
    """Extract aspect, sentiment, emoji, and sarcasm cues from text."""
    processor = preprocessor or RULE_PREPROCESSOR
    analysis = processor.analyze(text)
    tokens = analysis.normalized_text.split()

    aspect_matches: Dict[str, List[str]] = {}
    aspect_positions: Dict[str, List[int]] = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        matched_terms = [term for term in keywords if term and term in analysis.normalized_text]
        positions: List[int] = []
        for term in matched_terms:
            positions.extend(find_term_positions(tokens, term))
        aspect_matches[aspect] = sorted(set(matched_terms))
        aspect_positions[aspect] = sorted(set(positions))

    global_sentiment_terms = collect_sentiment_terms(tokens)
    aspect_sentiment_terms = {
        aspect: collect_sentiment_terms(tokens, positions)
        for aspect, positions in aspect_positions.items()
        if aspect != "none"
    }

    positive_cues = bool(global_sentiment_terms["positive"])
    negative_cues = bool(global_sentiment_terms["negative"])
    sentiment_conflict = positive_cues and negative_cues

    if any(token == "EMO_NEG" for token in analysis.emoji_tokens) and positive_cues:
        sentiment_conflict = True
    if any(token == "EMO_POS" for token in analysis.emoji_tokens) and negative_cues:
        sentiment_conflict = True

    notes: List[str] = []
    if sentiment_conflict:
        notes.append("Mixed positive and negative sentiment cues detected.")
    if any(marker in analysis.normalized_text for marker in NORMALIZED_CONTRAST_MARKERS):
        notes.append("Contrast marker detected in review text.")

    sarcasm_candidate = detect_sarcasm(
        analysis.normalized_text,
        analysis.emoji_names,
        global_sentiment_terms,
    )
    if sarcasm_candidate:
        notes.append("Sarcasm candidate detected from laughter or ironic markers.")

    return RuleFeatures(
        original_text=text,
        normalized_text=analysis.normalized_text,
        tokens=tokens,
        emoji_tokens=analysis.emoji_tokens,
        emoji_names=analysis.emoji_names,
        aspect_matches=aspect_matches,
        aspect_positions=aspect_positions,
        global_sentiment_terms=global_sentiment_terms,
        aspect_sentiment_terms=aspect_sentiment_terms,
        sarcasm_candidate=sarcasm_candidate,
        sentiment_conflict=sentiment_conflict,
        notes=notes,
    )


def score_sentiment_for_aspect(
    aspect_name: str,
    features: RuleFeatures,
) -> Dict[str, float]:
    """Score per-aspect sentiment using local cues, emojis, and global fallbacks."""
    local_terms = features.aspect_sentiment_terms.get(
        aspect_name,
        {sentiment: [] for sentiment in VALID_SENTIMENTS},
    )
    scores = {sentiment: 0.0 for sentiment in VALID_SENTIMENTS}

    for sentiment in VALID_SENTIMENTS:
        scores[sentiment] += 1.0 * len(local_terms.get(sentiment, []))
        scores[sentiment] += 0.25 * len(features.global_sentiment_terms.get(sentiment, []))

    for emoji_token in features.emoji_tokens:
        emoji_sentiment = EMOJI_SENTIMENT.get(emoji_token)
        if emoji_sentiment:
            scores[emoji_sentiment] += 0.6

    if features.sentiment_conflict:
        scores["neutral"] += 0.5
    if features.sarcasm_candidate:
        scores["neutral"] += 0.75

    return scores


def finalize_prediction(
    aspects: Sequence[str],
    aspect_sentiments: Mapping[str, str],
) -> Tuple[List[str], Dict[str, str]]:
    """Apply the final consistency rules for the competition schema."""
    return sanitize_aspect_sentiments(list(aspects), dict(aspect_sentiments))


def apply_postprocessing(
    text: str,
    label_probabilities: Sequence[float],
    threshold_config: Optional[Mapping[str, Any]] = None,
    features: Optional[RuleFeatures] = None,
) -> PredictionDecision:
    """Combine model probabilities with rules and return a schema-safe decision."""
    thresholds = dict(threshold_config or load_threshold_config())
    cached_features = features or extract_rule_features(text)
    label_probability_map, aspect_probabilities, sentiment_probabilities = build_probability_maps(
        label_probabilities
    )

    rule_decisions: List[str] = list(cached_features.notes)
    selected_aspects: List[str] = []

    for aspect in NON_NONE_ASPECTS:
        per_sentiment = sentiment_probabilities[aspect]
        model_sentiment = max(per_sentiment, key=per_sentiment.get)
        model_probability = per_sentiment[model_sentiment]
        label_threshold = resolve_label_threshold(thresholds, f"{aspect}_{model_sentiment}")
        if model_probability >= label_threshold:
            selected_aspects.append(aspect)
            rule_decisions.append(
                f"Model kept aspect '{aspect}' at probability {model_probability:.3f} "
                f"above threshold {label_threshold:.3f}."
            )

    aspect_scores: Dict[str, float] = {}
    for aspect in NON_NONE_ASPECTS:
        keyword_hits = len(cached_features.aspect_matches.get(aspect, []))
        nearby_sentiment_hits = sum(
            len(values)
            for values in cached_features.aspect_sentiment_terms.get(
                aspect,
                {sentiment: [] for sentiment in VALID_SENTIMENTS},
            ).values()
        )
        aspect_score = float(aspect_probabilities[aspect]) + 0.12 * keyword_hits + 0.08 * nearby_sentiment_hits
        if aspect == "general" and not any(
            cached_features.aspect_matches.get(other_aspect)
            for other_aspect in NON_NONE_ASPECTS
            if other_aspect != "general"
        ):
            aspect_score += 0.15 * sum(
                len(cached_features.global_sentiment_terms[sentiment])
                for sentiment in VALID_SENTIMENTS
            )
        aspect_scores[aspect] = aspect_score

        if aspect in selected_aspects:
            continue

        recovery_threshold = max(0.2, resolve_aspect_threshold(thresholds, aspect) - 0.15)
        strong_keyword_signal = keyword_hits >= 2 or (
            keyword_hits >= 1 and nearby_sentiment_hits >= 1
        )
        if strong_keyword_signal and aspect_score >= recovery_threshold:
            selected_aspects.append(aspect)
            rule_decisions.append(
                f"Recovered aspect '{aspect}' from keywords/rules "
                f"(score={aspect_score:.3f}, threshold={recovery_threshold:.3f})."
            )

    if not selected_aspects and any(cached_features.aspect_matches.values()):
        best_aspect = max(
            NON_NONE_ASPECTS,
            key=lambda aspect: aspect_scores.get(aspect, 0.0),
        )
        if aspect_scores.get(best_aspect, 0.0) >= 0.2:
            selected_aspects.append(best_aspect)
            rule_decisions.append(
                f"Forced best aspect '{best_aspect}' instead of 'none' because keywords were present."
            )

    if not selected_aspects and sum(
        len(cached_features.global_sentiment_terms[sentiment])
        for sentiment in VALID_SENTIMENTS
    ) > 0:
        selected_aspects.append("general")
        rule_decisions.append("Assigned 'general' because sentiment cues existed without a specific aspect.")

    if not selected_aspects:
        none_probability = float(aspect_probabilities.get("none", 1.0 - max(aspect_probabilities.values())))
        return PredictionDecision(
            aspects=["none"],
            aspect_sentiments={"none": "neutral"},
            label_probabilities=label_probability_map,
            aspect_probabilities=aspect_probabilities,
            sentiment_probabilities=sentiment_probabilities,
            matched_keywords={aspect: values for aspect, values in cached_features.aspect_matches.items() if values},
            matched_sentiment_terms=cached_features.aspect_sentiment_terms,
            emoji_tokens=cached_features.emoji_tokens,
            emoji_names=cached_features.emoji_names,
            rule_decisions=rule_decisions + ["Returned 'none' because no aspect signal survived post-processing."],
            sarcasm_candidate=cached_features.sarcasm_candidate,
            sentiment_conflict=cached_features.sentiment_conflict,
            prediction_confidence=round(max(0.05, min(1.0, none_probability)), 6),
        )

    selected_aspects = sorted(
        set(selected_aspects),
        key=lambda aspect: (-aspect_scores.get(aspect, 0.0), aspect),
    )
    predicted_sentiments: Dict[str, str] = {}
    confidences: List[float] = []

    for aspect in selected_aspects:
        per_sentiment = sentiment_probabilities[aspect]
        model_sentiment = max(per_sentiment, key=per_sentiment.get)
        model_probability = float(per_sentiment[model_sentiment])
        model_margin = model_probability - sorted(per_sentiment.values())[-2]

        rule_scores = score_sentiment_for_aspect(aspect, cached_features)
        rule_sentiment = max(rule_scores, key=rule_scores.get)
        sorted_rule_scores = sorted(rule_scores.values(), reverse=True)
        rule_margin = sorted_rule_scores[0] - sorted_rule_scores[1]

        chosen_sentiment = model_sentiment
        if sum(rule_scores.values()) > 0:
            if rule_margin >= 1.0 and rule_sentiment != model_sentiment:
                chosen_sentiment = rule_sentiment
                rule_decisions.append(
                    f"Overrode sentiment for '{aspect}' from '{model_sentiment}' to '{rule_sentiment}' via rules."
                )
            elif model_probability < 0.45 and rule_scores[rule_sentiment] >= 0.75:
                chosen_sentiment = rule_sentiment
                rule_decisions.append(
                    f"Used rule sentiment '{rule_sentiment}' for '{aspect}' because model confidence was low."
                )

        adjusted_confidence = model_probability
        if cached_features.sentiment_conflict:
            adjusted_confidence = max(0.05, adjusted_confidence - 0.10)
            if (
                chosen_sentiment != "neutral"
                and abs(rule_scores["positive"] - rule_scores["negative"]) <= 0.5
                and model_margin < 0.15
            ):
                chosen_sentiment = "neutral"
                rule_decisions.append(
                    f"Shifted '{aspect}' to 'neutral' because positive and negative cues conflicted."
                )
        if cached_features.sarcasm_candidate:
            adjusted_confidence = max(0.05, adjusted_confidence - 0.15)
            rule_decisions.append(f"Lowered confidence for '{aspect}' due to sarcasm candidate cues.")

        predicted_sentiments[aspect] = chosen_sentiment
        confidences.append(round(min(1.0, adjusted_confidence), 6))

    final_aspects, final_sentiments = finalize_prediction(selected_aspects, predicted_sentiments)
    prediction_confidence = round(min(confidences) if confidences else 0.05, 6)

    return PredictionDecision(
        aspects=final_aspects,
        aspect_sentiments=final_sentiments,
        label_probabilities=label_probability_map,
        aspect_probabilities=aspect_probabilities,
        sentiment_probabilities=sentiment_probabilities,
        matched_keywords={aspect: values for aspect, values in cached_features.aspect_matches.items() if values},
        matched_sentiment_terms={
            aspect: {
                sentiment: values
                for sentiment, values in per_aspect.items()
                if values
            }
            for aspect, per_aspect in cached_features.aspect_sentiment_terms.items()
            if any(per_aspect.values())
        },
        emoji_tokens=cached_features.emoji_tokens,
        emoji_names=cached_features.emoji_names,
        rule_decisions=rule_decisions,
        sarcasm_candidate=cached_features.sarcasm_candidate,
        sentiment_conflict=cached_features.sentiment_conflict,
        prediction_confidence=prediction_confidence,
    )


def prediction_to_vector(decision: PredictionDecision) -> np.ndarray:
    """Encode a post-processed prediction back into the multi-label space."""
    return create_multi_label_vector(decision.aspects, decision.aspect_sentiments)


def make_submission_record(review_id: Any, decision: PredictionDecision, review_text: Optional[str] = None) -> Dict[str, Any]:
    """Create a submission-compatible record with optional explainability metadata."""
    record = {
        "review_id": review_id,
        "aspects": decision.aspects,
        "aspect_sentiments": decision.aspect_sentiments,
    }
    if review_text is not None:
        record["review_text"] = review_text
    record["confidence"] = decision.prediction_confidence
    return record


if __name__ == "__main__":
    samples = [
        "الأكل تحفة 😍",
        "الخدمة سيئة 😡",
        "الخدمة ممتازة 😂",
        "المكان نظيف لكن الأسعار غالية",
    ]
    fake_probabilities = np.zeros(len(ASPECT_SENTIMENT_LABELS), dtype=np.float32)
    for sample in samples:
        features = extract_rule_features(sample)
        decision = apply_postprocessing(sample, fake_probabilities, features=features)
        print(sample)
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
        print("---")
