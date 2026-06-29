"""
Quick results generator.

Runs a reduced experimental pipeline (small transaction count) and writes
``full_experiments.json`` to the output directory. This is the canonical
implementation; ``code.scripts.generate_quick_results`` is a thin compatibility
shim that re-exports :func:`main` from here.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Make the project root importable so ``code.*`` packages resolve regardless of
# the current working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code.scripts.run_experiments import ExperimentRunner


def main(
    seed: int = 42,
    n_transactions: int = 10000,
    fraud_rate: float = 0.023,
    output_dir: str = "results/quick_run",
) -> Dict[str, Any]:
    """Run a quick experiment and return the results dictionary.

    Args:
        seed: Random seed for reproducibility.
        n_transactions: Number of synthetic transactions to generate.
        fraud_rate: Target fraud rate.
        output_dir: Directory in which to write ``full_experiments.json``.

    Returns:
        The results dictionary (also written to ``output_dir/full_experiments.json``).
    """
    os.makedirs(output_dir, exist_ok=True)
    runner = ExperimentRunner(seed=seed, output_dir=output_dir)
    results = runner.run_full_pipeline(
        n_transactions=n_transactions, fraud_rate=fraud_rate
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate quick experiment results.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-transactions", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.023)
    parser.add_argument("--output-dir", default="results/quick_run")
    args = parser.parse_args()

    main(
        seed=args.seed,
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        output_dir=args.output_dir,
    )
