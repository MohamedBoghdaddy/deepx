"""
Generate the Franco-Arabic seed lexicon CSV used by preprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from franco import DEFAULT_FRANCO_SEED_PATH, get_franco_entries, write_franco_seed_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Franco-Arabic seed CSV.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_FRANCO_SEED_PATH)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = write_franco_seed_csv(args.output_path)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "num_rows": len(get_franco_entries()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
