from aml.scripts.run_experiments import ExperimentRunner


def test_run_full_pipeline_smoke(tmp_path):
    outdir = tmp_path / "results"
    outdir.mkdir()

    runner = ExperimentRunner(
        seed=42, output_dir=str(outdir), model_params={"n_estimators": 10}
    )
    results = runner.run_full_pipeline(n_transactions=200, fraud_rate=0.02)

    assert isinstance(results, dict)
    assert "baseline_results" in results
    assert "agentic_results" in results
    assert (outdir / "full_experiments.json").exists()
