import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.qwen_zero_shot import parse_and_repair_json  # noqa: E402


class QwenZeroShotTests(unittest.TestCase):
    def test_parse_and_repair_json_handles_code_fences(self) -> None:
        raw = """```json
        {"aspects":["service"],"aspect_sentiments":{"service":"negative"}}
        ```"""
        parsed = parse_and_repair_json(raw)
        self.assertEqual(parsed["aspects"], ["service"])
        self.assertEqual(parsed["aspect_sentiments"]["service"], "negative")

    def test_parse_and_repair_json_sanitizes_invalid_labels(self) -> None:
        raw = '{"aspects":["unknown_aspect"],"aspect_sentiments":{"unknown_aspect":"happy"}}'
        parsed = parse_and_repair_json(raw)
        self.assertEqual(parsed["aspects"], ["none"])
        self.assertEqual(parsed["aspect_sentiments"], {"none": "neutral"})


if __name__ == "__main__":
    unittest.main()
