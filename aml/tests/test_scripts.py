def test_generate_quick_results_importable():
    import scripts.generate_quick_results as gq

    assert hasattr(gq, "main")


def test_generate_deterministic_results_importable():
    import scripts.generate_deterministic_results as gd

    assert hasattr(gd, "main")


def test_quick_main_smoke_runs(tmp_path):
    # Smoke-run quick main with small number of transactions to ensure it executes
    import scripts.generate_quick_results as gq

    outdir = tmp_path / "quick_test"
    outdir.mkdir()

    # Run with a very small dataset to keep test fast
    results = gq.main(
        seed=42, n_transactions=200, fraud_rate=0.02, output_dir=str(outdir)
    )

    assert isinstance(results, dict)
    assert "config" in results
    assert (outdir / "full_experiments.json").exists()


def test_deterministic_main_smoke_runs(tmp_path):
    import scripts.generate_deterministic_results as gd

    outdir = tmp_path / "deterministic_test"
    outdir.mkdir()

    results = gd.main(
        seed=42, n_transactions=200, fraud_rate=0.02, output_dir=str(outdir)
    )

    assert isinstance(results, dict)
    assert "config" in results
    assert (outdir / "full_experiments.json").exists()
