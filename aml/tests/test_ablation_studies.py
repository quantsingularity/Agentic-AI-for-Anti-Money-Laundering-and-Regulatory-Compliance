from aml.scripts import ablation_studies as ab


def test_ablation_runs(tmp_path):
    outdir = tmp_path / "abl"
    outdir.mkdir()

    results = ab.main(
        seed=42, n_transactions=200, fraud_rate=0.02, output_dir=str(outdir)
    )

    assert isinstance(results, dict)
    assert (outdir / "ablation_results.json").exists()
