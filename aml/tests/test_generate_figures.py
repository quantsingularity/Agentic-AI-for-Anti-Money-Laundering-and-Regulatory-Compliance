import json

import pytest

from aml.scripts import generate_figures as gf


@pytest.fixture
def sample_results(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    results = {
        "baseline_results": {
            "rule_based": {
                "precision": 0.34,
                "recall": 0.89,
                "f1": 0.495,
                "roc_auc": 0.673,
                "pr_auc": 0.412,
            },
            "isolation_forest": {
                "precision": 0.456,
                "recall": 0.634,
                "f1": 0.531,
                "roc_auc": 0.762,
                "pr_auc": 0.509,
            },
            "xgboost": {
                "precision": 0.723,
                "recall": 0.812,
                "f1": 0.765,
                "roc_auc": 0.894,
                "pr_auc": 0.781,
            },
        },
        "agentic_results": {
            "precision": 0.847,
            "recall": 0.893,
            "f1": 0.869,
            "roc_auc": 0.921,
            "pr_auc": 0.856,
            "sar_generation_time_mean": 4.2,
            "sar_generation_time_std": 1.1,
        },
    }
    (d / "full_experiments.json").write_text(json.dumps(results))
    return d


def test_generate_roc_pr_and_metrics(tmp_path, sample_results):
    out = tmp_path / "figs"
    out.mkdir()

    # Generate a subset of figures
    gf.generate_roc_pr_curves(sample_results, out)
    gf.generate_metrics_comparison(sample_results, out)

    assert (out / "eval_roc_pr.png").exists()
    assert (out / "metrics_comparison.png").exists()


def test_generate_latency_and_explain(tmp_path, sample_results):
    out = tmp_path / "figs2"
    out.mkdir()

    gf.generate_sar_latency_throughput(sample_results, out)
    gf.generate_explainability_annotation(out)

    assert (out / "sar_latency_throughput.png").exists()
    assert (out / "explainability_annotation.png").exists()


def test_architecture_diagram_skips_without_graphviz(tmp_path):
    out = tmp_path / "figs3"
    out.mkdir()

    try:

        has_graphviz = True
    except Exception:
        has_graphviz = False

    gf.generate_architecture_diagram(out)
    if has_graphviz:
        assert (out / "system_architecture.svg").exists()
    else:
        # no file created, but function should not raise
        assert True
