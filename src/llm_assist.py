"""
Optional safe-mode LLM assist for edge-case ABSA reviews.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Mapping


def llm_assist_enabled() -> bool:
    """Return True only when explicitly enabled."""
    return os.getenv("USE_LLM_ASSIST", "false").strip().lower() == "true"


def maybe_refine_prediction(
    text: str,
    prediction: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """No-op by default; ready for guarded future use on edge cases only."""
    if not llm_assist_enabled():
        return {
            "used_llm": False,
            "reason": "USE_LLM_ASSIST is not enabled.",
            "prediction": dict(prediction),
        }

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "used_llm": False,
            "reason": "OPENAI_API_KEY is missing; returning the original prediction.",
            "prediction": dict(prediction),
        }

    del text, context
    return {
        "used_llm": False,
        "reason": "Safe-mode scaffold is enabled, but active refinement is intentionally disabled by default.",
        "prediction": dict(prediction),
    }


def main() -> None:
    """CLI entry point for quick inspection."""
    parser = argparse.ArgumentParser(description="Inspect the safe-mode LLM assist status.")
    parser.add_argument("--text", default="الخدمة ممتازة لكن التوصيل متأخر 😂")
    parser.add_argument("--prediction", default='{"aspects":["service"],"aspect_sentiments":{"service":"positive"}}')
    args = parser.parse_args()

    prediction = json.loads(args.prediction)
    result = maybe_refine_prediction(args.text, prediction)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
