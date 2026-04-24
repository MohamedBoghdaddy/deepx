"""Benchmark report and comparison table generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from benchmark.metrics import build_model_summary, flatten_benchmark_result


def select_best_model(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose the best model using aspect detection first, then joint quality and speed."""
    if not results:
        raise ValueError("No benchmark results were provided.")

    return max(
        results,
        key=lambda item: (
            float(item["metrics"]["aspect_detection"]["micro_f1"]),
            float(item["metrics"]["joint"]["micro_f1"]),
            float(item["metrics"]["sentiment_classification"]["micro_f1"]),
            -float(item["timing"]["avg_inference_time_ms"]),
        ),
    )


def format_cell(value: Any) -> str:
    """Format a comparison-table cell."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def render_markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render a compact Markdown table without extra dependencies."""
    if not rows:
        return ""

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def build_comparison_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build rows for the printed comparison table."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        rows.append(build_model_summary(result))
    return rows


def build_csv_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build flattened CSV rows for aggregate result exports."""
    return [flatten_benchmark_result(result) for result in results]


def explain_best_model(best_result: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> List[str]:
    """Generate short narrative bullets about the best model."""
    aspect_micro_f1 = best_result["metrics"]["aspect_detection"]["micro_f1"]
    sentiment_micro_f1 = best_result["metrics"]["sentiment_classification"]["micro_f1"]
    coverage = best_result["metrics"]["sentiment_classification"]["coverage"]
    inference_ms = best_result["timing"]["avg_inference_time_ms"]

    bullets = [
        (
            f"`{best_result['model_name']}` ranked first because it achieved the strongest aspect detection "
            f"micro F1 (`{aspect_micro_f1:.4f}`), which is the primary bottleneck in end-to-end ABSA."
        ),
        (
            f"It paired that with sentiment micro F1 of `{sentiment_micro_f1:.4f}` over matched aspects "
            f"at coverage `{coverage:.4f}`."
        ),
        f"Its average inference time was `{inference_ms:.4f}` ms per review on the validation benchmark run.",
    ]

    sorted_by_aspect = sorted(
        results,
        key=lambda item: float(item["metrics"]["aspect_detection"]["micro_f1"]),
        reverse=True,
    )
    if len(sorted_by_aspect) > 1:
        runner_up = sorted_by_aspect[1]
        gap = float(best_result["metrics"]["aspect_detection"]["micro_f1"]) - float(runner_up["metrics"]["aspect_detection"]["micro_f1"])
        bullets.append(
            f"The margin over the runner-up (`{runner_up['model_name']}`) on aspect micro F1 was `{gap:.4f}`."
        )

    return bullets


def render_benchmark_report(results: Sequence[Mapping[str, Any]], output_dir: Path) -> str:
    """Render the final benchmark markdown report."""
    best_result = select_best_model(results)
    comparison_rows = build_comparison_rows(results)
    report_lines = [
        "# Arabic ABSA Benchmark Report",
        "",
        "## Benchmark Table",
        render_markdown_table(comparison_rows),
        "",
        "## Best Model",
        f"`{best_result['model_name']}` was the strongest overall benchmark run.",
        "",
    ]

    report_lines.extend(f"- {bullet}" for bullet in explain_best_model(best_result, results))
    report_lines.extend(
        [
            "",
            "## Per-Model Notes",
        ]
    )

    for result in results:
        report_lines.extend(
            [
                f"### {result['model_name']}",
                (
                    f"- Aspect detection: micro F1 `{result['metrics']['aspect_detection']['micro_f1']:.4f}`, "
                    f"macro F1 `{result['metrics']['aspect_detection']['macro_f1']:.4f}`, "
                    f"PR-AUC micro `{format_cell(result['metrics']['aspect_detection']['pr_auc']['micro'])}`"
                ),
                (
                    f"- Sentiment classification: micro F1 `{result['metrics']['sentiment_classification']['micro_f1']:.4f}`, "
                    f"macro F1 `{result['metrics']['sentiment_classification']['macro_f1']:.4f}`, "
                    f"coverage `{result['metrics']['sentiment_classification']['coverage']:.4f}`"
                ),
                (
                    f"- End-to-end joint micro F1 `{result['metrics']['joint']['micro_f1']:.4f}` "
                    f"with exact-match accuracy `{result['metrics']['joint']['exact_match_accuracy']:.4f}`"
                ),
                f"- Average inference time: `{result['timing']['avg_inference_time_ms']:.4f}` ms/review",
                f"- Artifacts: `{result['artifacts']['model_output_dir']}`",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Output Files",
            f"- Aggregate CSV: `{output_dir / 'benchmark_results.csv'}`",
            f"- Aggregate JSON: `{output_dir / 'benchmark_results.json'}`",
            f"- Per-model artifacts: `{output_dir / 'benchmark'}`",
        ]
    )
    return "\n".join(report_lines).strip() + "\n"


def save_report(report_text: str, output_path: Path) -> None:
    """Save the markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def print_terminal_table(results: Sequence[Mapping[str, Any]]) -> None:
    """Print the compact benchmark table to the terminal."""
    print(render_markdown_table(build_comparison_rows(results)))


def save_aggregate_json(results: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Save the aggregate benchmark JSON payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")
