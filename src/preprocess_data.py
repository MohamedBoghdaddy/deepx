"""
Offline dataset preprocessing for multilingual Arabic ABSA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from dataset import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_UNLABELED_PATH,
    DEFAULT_VALIDATION_PATH,
    OUTPUTS_ROOT,
    infer_column_mapping,
    load_dataframe,
    resolve_input_path,
)
from preprocess import ArabicPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_LANGUAGE_STATS_PATH = OUTPUTS_ROOT / "language_stats.json"


def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Persist a DataFrame using the same file extension as the source file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        dataframe.to_excel(output_path, index=False)
        return
    if suffix == ".csv":
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    if suffix == ".json":
        dataframe.to_json(output_path, orient="records", force_ascii=False, indent=2)
        return
    raise ValueError(f"Unsupported output file type: {output_path}")


def candidate_text_columns(dataframe: pd.DataFrame) -> List[str]:
    """Find dataset columns that should go through text normalization."""
    mapping = infer_column_mapping(dataframe, require_labels=False)
    candidates = [mapping.review_text]
    for column_name in dataframe.columns:
        normalized = str(column_name).strip().lower()
        if column_name in candidates:
            continue
        if any(token in normalized for token in ("review_text", "review", "comment", "content", "text")):
            candidates.append(column_name)
    return candidates


def process_dataframe(
    dataframe: pd.DataFrame,
    preprocessor: ArabicPreprocessor,
) -> tuple[pd.DataFrame, Counter]:
    """Normalize text columns and collect detected-language counts."""
    processed = dataframe.copy()
    language_counts: Counter = Counter()

    for column_name in candidate_text_columns(processed):
        processed_values = []
        for value in processed[column_name].tolist():
            if pd.isna(value):
                processed_values.append(value)
                continue

            analysis = preprocessor.analyze(str(value))
            language_counts[analysis.language] += 1
            processed_values.append(analysis.normalized_text)
        processed[column_name] = processed_values

    return processed, language_counts


def iter_default_inputs() -> Iterable[Path]:
    """Yield the competition datasets in the expected order."""
    yield DEFAULT_TRAIN_PATH
    yield DEFAULT_VALIDATION_PATH
    yield DEFAULT_UNLABELED_PATH


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess DeepX datasets for multilingual ABSA.")
    parser.add_argument("--input_files", nargs="*", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--language_stats_path", type=Path, default=DEFAULT_LANGUAGE_STATS_PATH)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    requested_inputs = args.input_files or list(iter_default_inputs())
    input_files = [
        resolve_input_path(path, path)
        for path in requested_inputs
    ]
    input_files = [path for path in input_files if path is not None]

    preprocessor = ArabicPreprocessor()
    overall_language_counts: Counter = Counter()
    per_file_language_counts: Dict[str, Dict[str, int]] = {}

    output_dir = resolve_input_path(args.output_dir, DEFAULT_PROCESSED_DIR) or DEFAULT_PROCESSED_DIR
    language_stats_path = (
        resolve_input_path(args.language_stats_path, DEFAULT_LANGUAGE_STATS_PATH)
        or DEFAULT_LANGUAGE_STATS_PATH
    )

    for input_path in input_files:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        dataframe = load_dataframe(input_path)
        processed_df, language_counts = process_dataframe(dataframe, preprocessor)
        save_dataframe(processed_df, output_dir / input_path.name)

        per_file_language_counts[input_path.name] = {
            language: int(count)
            for language, count in sorted(language_counts.items())
        }
        overall_language_counts.update(language_counts)

    language_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with language_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "overall": {
                    language: int(count)
                    for language, count in sorted(overall_language_counts.items())
                },
                "per_file": per_file_language_counts,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(
            {
                "processed_files": [path.name for path in input_files],
                "output_dir": str(output_dir),
                "language_stats_path": str(language_stats_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
