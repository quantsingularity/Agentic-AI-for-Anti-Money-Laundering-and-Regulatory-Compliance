"""
Populate the paper's results table from an experiment results JSON.

Reads ``full_experiments.json`` produced by ``run_experiments.py`` and renders a
LaTeX table comparing the baseline detectors against the agentic system on the
standard detection metrics (precision, recall, F1, and any extra metrics that are
present, e.g. AUC). Designed to be re-run whenever experiments are refreshed so
the paper always reflects the latest numbers.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Human-readable labels for known baseline keys.
_LABELS = {
    "rule_based": "Rule-Based",
    "isolation_forest": "Isolation Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}

# Metrics shown in the table, in order, with the JSON keys that may hold them.
_METRIC_COLUMNS = [
    ("Precision", ["precision"]),
    ("Recall", ["recall"]),
    ("F1", ["f1", "f1_score"]),
    ("AUC", ["auc", "roc_auc", "auc_roc"]),
]


def _pretty_label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " ").title())


def _get_metric(metrics: Dict[str, Any], candidates) -> Optional[float]:
    for key in candidates:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    return None


def _format_value(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "--"


def _row(label: str, metrics: Dict[str, Any], bold: bool = False) -> str:
    cells = [label]
    for _, candidates in _METRIC_COLUMNS:
        value = _get_metric(metrics, candidates)
        formatted = _format_value(value)
        if bold and value is not None:
            formatted = f"\\textbf{{{formatted}}}"
        cells.append(formatted)
    label_cell = f"\\textbf{{{label}}}" if bold else label
    cells[0] = label_cell
    return " & ".join(cells) + " \\\\"


def build_latex_table(results: Dict[str, Any]) -> str:
    """Build a LaTeX ``table`` environment from an experiment results dict."""
    baseline_results = results.get("baseline_results", {}) or {}
    agentic_results = results.get("agentic_results", {}) or {}

    header_cols = " & ".join(["Model"] + [name for name, _ in _METRIC_COLUMNS])
    col_spec = "l" + "c" * len(_METRIC_COLUMNS)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Detection performance: baselines vs. the agentic AML system.}",
        "\\label{tab:results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"{header_cols} \\\\",
        "\\midrule",
    ]

    # Baseline rows in a stable, sensible order.
    preferred = ["rule_based", "isolation_forest", "xgboost"]
    ordered_keys = [k for k in preferred if k in baseline_results]
    ordered_keys += [k for k in baseline_results if k not in ordered_keys]
    for key in ordered_keys:
        metrics = baseline_results[key]
        if isinstance(metrics, dict):
            lines.append(_row(_pretty_label(key), metrics))

    lines.append("\\midrule")
    lines.append(_row("Agentic (Ours)", agentic_results, bold=True))

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main(
    results_dir: str = "results",
    output_file: str = "paper_results.tex",
    results_filename: str = "full_experiments.json",
) -> str:
    """Render the results LaTeX table and write it to ``output_file``.

    Args:
        results_dir: Directory containing the experiment results JSON.
        output_file: Path of the LaTeX file to write.
        results_filename: Name of the results JSON within ``results_dir``.

    Returns:
        The path to the written LaTeX file.
    """
    results_path = Path(results_dir) / results_filename
    with open(results_path) as f:
        results = json.load(f)

    latex = build_latex_table(results)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate the paper's results table from experiment JSON."
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument(
        "--output-file", default="paper_results.tex", help="Output LaTeX file"
    )
    parser.add_argument(
        "--results-filename",
        default="full_experiments.json",
        help="Name of the results JSON file",
    )
    args = parser.parse_args()

    out = main(
        results_dir=args.results_dir,
        output_file=args.output_file,
        results_filename=args.results_filename,
    )
    print(f"Wrote results table to {out}")
