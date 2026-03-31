"""Run ablation studies and save results.

This script runs ablation variants using the `ExperimentRunner` to allow
reproducible ablation experiments. It's lightweight and intended to be
invoked from `run_full.sh` (or directly) with a small number of transactions
for quick checks.
"""

import argparse
import json
import logging
from code.scripts.run_experiments import ExperimentRunner
from pathlib import Path

logger = logging.getLogger(__name__)


def main(
    seed: int = 42,
    n_transactions: int = 10000,
    fraud_rate: float = 0.023,
    output_dir: str = "results/ablation_studies",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner(seed=seed, output_dir=str(output_dir))

    # Generate data and split
    df = runner._generate_data(n_transactions, fraud_rate)
    train_df, test_df = runner._time_based_split(df, test_ratio=0.3)

    # Run ablation studies (uses internal helper)
    ablations = runner._run_ablations(train_df, test_df)

    out_file = output_dir / "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(ablations, f, indent=2)

    logger.info("Ablation results saved to: %s", out_file)
    return ablations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-transactions", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.023)
    parser.add_argument("--output-dir", type=str, default="results/ablation_studies")

    args = parser.parse_args()
    main(
        seed=args.seed,
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        output_dir=args.output_dir,
    )
