"""
JSON Validator Module for Arabic ABSA
=====================================
Validates submission JSON against the expected schema.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_TEST_PATH = DATASET_ROOT / "DeepX_unlabeled.xlsx"
DEFAULT_SAMPLE_SUBMISSION_PATH = DATASET_ROOT / "sample_submission.json"


class ValidationError(Exception):
    """Custom exception for validation errors."""


def validate_schema(data: Any) -> Tuple[bool, List[str]]:
    """Validate the overall JSON schema."""
    errors = []

    if not isinstance(data, list):
        errors.append("Root must be a list")
        return False, errors

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {index}: Must be a dictionary")
            continue

        if "review_id" not in item:
            errors.append(f"Item {index}: Missing 'review_id'")
        if "aspects" not in item:
            errors.append(f"Item {index}: Missing 'aspects'")
        if "aspect_sentiments" not in item:
            errors.append(f"Item {index}: Missing 'aspect_sentiments'")

    return len(errors) == 0, errors


def validate_review_entry(entry: Dict, index: int) -> List[str]:
    """Validate a single review entry."""
    errors = []

    review_id = entry.get("review_id")
    if review_id is None:
        errors.append(f"Entry {index}: review_id is None")
    elif not isinstance(review_id, (int, str)):
        errors.append(f"Entry {index}: review_id must be int or str, got {type(review_id)}")

    aspects = entry.get("aspects", [])
    if not isinstance(aspects, list):
        errors.append(f"Entry {index}: aspects must be a list, got {type(aspects)}")
    else:
        for aspect in aspects:
            if aspect not in VALID_ASPECTS:
                errors.append(f"Entry {index}: Invalid aspect '{aspect}'")

    sentiments = entry.get("aspect_sentiments", {})
    if not isinstance(sentiments, dict):
        errors.append(f"Entry {index}: aspect_sentiments must be a dict, got {type(sentiments)}")
    else:
        for aspect, sentiment in sentiments.items():
            if aspect not in aspects:
                errors.append(
                    f"Entry {index}: aspect '{aspect}' in aspect_sentiments but not in aspects"
                )
            if aspect not in VALID_ASPECTS:
                errors.append(f"Entry {index}: Invalid aspect '{aspect}' in aspect_sentiments")
            if sentiment not in VALID_SENTIMENTS:
                errors.append(
                    f"Entry {index}: Invalid sentiment '{sentiment}' for aspect '{aspect}'"
                )

    if isinstance(aspects, list) and isinstance(sentiments, dict):
        if set(aspects) != set(sentiments.keys()):
            errors.append(
                f"Entry {index}: aspects and aspect_sentiments keys don't match. "
                f"aspects: {set(aspects)}, sentiments: {set(sentiments.keys())}"
            )

    if "none" in aspects:
        if len(aspects) > 1:
            errors.append(f"Entry {index}: 'none' aspect should be the only aspect when present")
        if sentiments.get("none") != "neutral":
            errors.append(f"Entry {index}: 'none' aspect should have 'neutral' sentiment")

    return errors


def validate_submission(
    submission_path: str,
    sample_submission_path: Optional[str] = None,
    check_review_ids: bool = True,
    test_df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, Dict]:
    """Validate a submission JSON file."""
    errors = []
    warnings = []

    if not os.path.exists(submission_path):
        errors.append(f"Submission file not found: {submission_path}")
        return False, {"errors": errors, "warnings": warnings}

    try:
        with open(submission_path, "r", encoding="utf-8") as handle:
            submission = json.load(handle)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON: {exc}")
        return False, {"errors": errors, "warnings": warnings}

    schema_valid, schema_errors = validate_schema(submission)
    if not schema_valid:
        errors.extend(schema_errors)

    for index, entry in enumerate(submission):
        errors.extend(validate_review_entry(entry, index))

    if check_review_ids and test_df is not None:
        submission_ids = set(entry["review_id"] for entry in submission if "review_id" in entry)
        expected_ids = set(test_df["review_id"].tolist())

        missing_ids = expected_ids - submission_ids
        extra_ids = submission_ids - expected_ids
        if missing_ids:
            warnings.append(f"Missing review_ids: {sorted(list(missing_ids))[:10]}...")
        if extra_ids:
            warnings.append(f"Extra review_ids: {sorted(list(extra_ids))[:10]}...")

    if sample_submission_path and os.path.exists(sample_submission_path):
        with open(sample_submission_path, "r", encoding="utf-8") as handle:
            sample = json.load(handle)

        sample_ids = set(entry["review_id"] for entry in sample)
        submission_ids = set(entry["review_id"] for entry in submission if "review_id" in entry)
        if sample_ids != submission_ids:
            warnings.append(
                "review_ids don't match sample submission. "
                f"Sample has {len(sample_ids)}, submission has {len(submission_ids)}"
            )

    report = {
        "is_valid": len(errors) == 0,
        "total_entries": len(submission),
        "errors": errors,
        "warnings": warnings,
        "valid_aspects": VALID_ASPECTS,
        "valid_sentiments": VALID_SENTIMENTS,
    }
    return report["is_valid"], report


def print_validation_report(report: Dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"\nTotal entries: {report.get('total_entries', 0)}")
    print(f"Valid: {report.get('is_valid', False)}")

    if report.get("errors"):
        print(f"\nErrors ({len(report['errors'])}):")
        for error in report["errors"][:20]:
            print(f"  - {error}")
        if len(report["errors"]) > 20:
            print(f"  ... and {len(report['errors']) - 20} more")

    if report.get("warnings"):
        print(f"\nWarnings ({len(report['warnings'])}):")
        for warning in report["warnings"][:10]:
            print(f"  - {warning}")
        if len(report["warnings"]) > 10:
            print(f"  ... and {len(report['warnings']) - 10} more")

    if not report.get("errors") and not report.get("warnings"):
        print("\nNo issues found!")

    print("\n" + "=" * 50)


def fix_submission(
    submission_path: str,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Attempt to fix common issues in a submission file."""
    with open(submission_path, "r", encoding="utf-8") as handle:
        submission = json.load(handle)

    fixed_submission = []
    for entry in submission:
        fixed_entry = {"review_id": entry.get("review_id")}

        aspects = entry.get("aspects", [])
        if not isinstance(aspects, list):
            aspects = []
        valid_aspects = [aspect for aspect in aspects if aspect in VALID_ASPECTS]

        sentiments = entry.get("aspect_sentiments", {})
        if not isinstance(sentiments, dict):
            sentiments = {}

        fixed_sentiments = {}
        for aspect in valid_aspects:
            sentiment = sentiments.get(aspect, "neutral")
            if sentiment not in VALID_SENTIMENTS:
                sentiment = "neutral"
            fixed_sentiments[aspect] = sentiment

        if not valid_aspects:
            valid_aspects = ["none"]
            fixed_sentiments = {"none": "neutral"}

        if "none" in valid_aspects and len(valid_aspects) > 1:
            valid_aspects = ["none"]
            fixed_sentiments = {"none": "neutral"}

        fixed_entry["aspects"] = valid_aspects
        fixed_entry["aspect_sentiments"] = fixed_sentiments
        fixed_submission.append(fixed_entry)

    fixed_submission.sort(key=lambda item: item["review_id"])
    output_file = output_path or submission_path
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(fixed_submission, handle, indent=2, ensure_ascii=False)

    print(f"Fixed submission saved to: {output_file}")
    return fixed_submission


def resolve_input_path(path_str: Optional[str], default_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve a user-provided path against common project locations."""
    if not path_str:
        return default_path

    candidates = [Path(path_str), PROJECT_ROOT / path_str, DATASET_ROOT / path_str]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return Path(path_str)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for submission validation."""
    parser = argparse.ArgumentParser(description="Validate a DeepX submission JSON file.")
    parser.add_argument("submission_path")
    parser.add_argument(
        "sample_submission_path",
        nargs="?",
        default=str(DEFAULT_SAMPLE_SUBMISSION_PATH),
    )
    parser.add_argument("--test_path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--skip_review_ids", action="store_true")
    return parser


def main():
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    submission_path = resolve_input_path(args.submission_path)
    sample_submission_path = resolve_input_path(
        args.sample_submission_path,
        DEFAULT_SAMPLE_SUBMISSION_PATH,
    )
    test_path = resolve_input_path(args.test_path, DEFAULT_TEST_PATH)
    test_df = pd.read_excel(test_path) if test_path and test_path.exists() else None

    is_valid, report = validate_submission(
        str(submission_path),
        sample_submission_path=str(sample_submission_path) if sample_submission_path else None,
        check_review_ids=not args.skip_review_ids,
        test_df=test_df,
    )
    print_validation_report(report)
    raise SystemExit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
