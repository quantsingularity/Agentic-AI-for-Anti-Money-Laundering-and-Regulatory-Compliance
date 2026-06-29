import json
from pathlib import Path

from aml.scripts import populate_paper as pp


def test_populate_paper_writes_file(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    results = {
        "baseline_results": {"xgboost": {"precision": 0.7, "recall": 0.8, "f1": 0.75}},
        "agentic_results": {"precision": 0.8, "recall": 0.85, "f1": 0.825},
    }

    (results_dir / "full_experiments.json").write_text(json.dumps(results))

    out = pp.main(results_dir=str(results_dir), output_file=str(tmp_path / "paper.tex"))

    assert Path(out).exists()
    content = Path(out).read_text()
    assert "XGBoost" in content
    assert "Agentic" in content
