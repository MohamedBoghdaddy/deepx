"""
Mine aspect-related keyword expansions from unlabeled DeepX reviews.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from dataset import DEFAULT_UNLABELED_PATH, resolve_input_path
from preprocess import ArabicPreprocessor
from unlabeled_utils import (
    ASPECT_KEYWORD_SEEDS,
    DEFAULT_CLEAN_UNLABELED_PATH,
    DEFAULT_KEYWORD_REPORT_PATH,
    load_and_clean_unlabeled_data,
    save_json,
)


def _normalize_seed_terms(seeds: Sequence[str], preprocessor: ArabicPreprocessor) -> List[str]:
    normalized = []
    for seed in seeds:
        normalized_seed = preprocessor.normalize(str(seed)).strip()
        if normalized_seed and normalized_seed not in normalized:
            normalized.append(normalized_seed)
    return normalized


def _phrase_positions(tokens: Sequence[str], phrase: str) -> List[int]:
    phrase_tokens = phrase.split()
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return []
    positions = []
    for index in range(len(tokens) - len(phrase_tokens) + 1):
        if list(tokens[index : index + len(phrase_tokens)]) == phrase_tokens:
            positions.append(index)
    return positions


def _valid_term(token: str, seed_terms: Sequence[str]) -> bool:
    if not token or len(token) <= 1:
        return False
    if token.startswith("EMO_"):
        return False
    if token in seed_terms:
        return False
    return True


def mine_aspect_keyword_report(
    normalized_reviews: Sequence[str],
    original_reviews: Sequence[str],
    seeds: Mapping[str, Sequence[str]] = ASPECT_KEYWORD_SEEDS,
    window: int = 4,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Extract frequent neighboring terms and phrases around seed keywords."""
    preprocessor = ArabicPreprocessor()
    report: Dict[str, Any] = {
        "total_reviews": int(len(normalized_reviews)),
        "window_size": int(window),
        "top_k": int(top_k),
        "aspects": {},
    }

    tokenized_reviews = [str(text or "").split() for text in normalized_reviews]

    for aspect, seed_terms in seeds.items():
        normalized_seeds = _normalize_seed_terms(seed_terms, preprocessor)
        unigram_counter: Counter[str] = Counter()
        phrase_counter: Counter[str] = Counter()
        matched_reviews = 0
        example_reviews: List[str] = []

        for original_review, tokens in zip(original_reviews, tokenized_reviews):
            if not tokens:
                continue

            review_has_seed = False
            context_tokens: List[str] = []
            for seed in normalized_seeds:
                positions = _phrase_positions(tokens, seed)
                if not positions:
                    continue

                review_has_seed = True
                seed_length = len(seed.split())
                for position in positions:
                    left = max(0, position - window)
                    right = min(len(tokens), position + seed_length + window)
                    neighborhood = tokens[left:position] + tokens[position + seed_length : right]
                    context_tokens.extend(token for token in neighborhood if _valid_term(token, normalized_seeds))

                    for phrase_length in (2, 3):
                        for phrase_start in range(left, max(left, right - phrase_length) + 1):
                            phrase_tokens = tokens[phrase_start : phrase_start + phrase_length]
                            if len(phrase_tokens) != phrase_length:
                                continue
                            phrase = " ".join(phrase_tokens)
                            if any(seed == phrase for seed in normalized_seeds):
                                continue
                            if all(_valid_term(token, normalized_seeds) for token in phrase_tokens):
                                phrase_counter[phrase] += 1

            if review_has_seed:
                matched_reviews += 1
                if len(example_reviews) < 5:
                    example_reviews.append(str(original_review))
                unigram_counter.update(context_tokens)

        report["aspects"][aspect] = {
            "seed_keywords": list(seed_terms),
            "normalized_seed_keywords": normalized_seeds,
            "matched_reviews": int(matched_reviews),
            "top_neighbor_terms": [
                {"term": term, "count": int(count)}
                for term, count in unigram_counter.most_common(top_k)
            ],
            "top_neighbor_phrases": [
                {"phrase": phrase, "count": int(count)}
                for phrase, count in phrase_counter.most_common(top_k)
            ],
            "example_reviews": example_reviews,
        }

    return report


def print_keyword_summary(report: Mapping[str, Any]) -> None:
    """Print a readable terminal summary."""
    print("=" * 80)
    print("Aspect keyword mining summary")
    print("=" * 80)
    for aspect, details in report.get("aspects", {}).items():
        top_terms = ", ".join(
            f"{item['term']} ({item['count']})"
            for item in details.get("top_neighbor_terms", [])[:8]
        ) or "No strong neighbors found"
        top_phrases = ", ".join(
            f"{item['phrase']} ({item['count']})"
            for item in details.get("top_neighbor_phrases", [])[:5]
        ) or "No strong phrases found"
        print(f"[{aspect}] matched_reviews={details.get('matched_reviews', 0)}")
        print(f"  top terms   : {top_terms}")
        print(f"  top phrases : {top_phrases}")


def run_keyword_mining(
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    output_path: Path = DEFAULT_KEYWORD_REPORT_PATH,
    clean_output_path: Optional[Path] = DEFAULT_CLEAN_UNLABELED_PATH,
    top_k: int = 15,
    window: int = 4,
) -> Dict[str, Any]:
    """Clean unlabeled reviews and mine aspect-related keyword expansions."""
    cleaned_df, cleaning_summary = load_and_clean_unlabeled_data(
        unlabeled_path=unlabeled_path,
        output_path=clean_output_path,
    )
    report = mine_aspect_keyword_report(
        normalized_reviews=cleaned_df.get("normalized_review_text", []).tolist(),
        original_reviews=cleaned_df.get("review_text", []).tolist(),
        seeds=ASPECT_KEYWORD_SEEDS,
        window=window,
        top_k=top_k,
    )
    report["cleaning_summary"] = cleaning_summary
    report["input_path"] = str(resolve_input_path(unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH)
    report["output_path"] = str(output_path)
    save_json(report, output_path)
    print_keyword_summary(report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Mine aspect keyword expansions from the unlabeled DeepX reviews."
    )
    parser.add_argument("--unlabeled_path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_KEYWORD_REPORT_PATH)
    parser.add_argument("--clean_output_path", type=Path, default=DEFAULT_CLEAN_UNLABELED_PATH)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--window", type=int, default=4)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    report = run_keyword_mining(
        unlabeled_path=resolve_input_path(args.unlabeled_path, DEFAULT_UNLABELED_PATH) or DEFAULT_UNLABELED_PATH,
        output_path=resolve_input_path(args.output_path, DEFAULT_KEYWORD_REPORT_PATH) or DEFAULT_KEYWORD_REPORT_PATH,
        clean_output_path=resolve_input_path(args.clean_output_path, DEFAULT_CLEAN_UNLABELED_PATH) or DEFAULT_CLEAN_UNLABELED_PATH,
        top_k=args.top_k,
        window=args.window,
    )
    print(json.dumps({"output_path": report["output_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
