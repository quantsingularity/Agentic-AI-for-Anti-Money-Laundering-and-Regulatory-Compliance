# Reproducibility Checklist

## Overview
This document provides step-by-step instructions to reproduce all experimental results reported in the paper "Agentic AI for Anti-Money Laundering and Regulatory Compliance."

## System Requirements

### Hardware
- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8 CPU cores, 16GB RAM, GPU (optional)
- **Storage**: 5GB free disk space

### Software
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+ with WSL2
- **Python**: 3.10 or 3.11
- **Docker**: 20.10+ (optional but recommended)

## Quick Reproducibility Test (30 minutes)

### Option 1: Using Docker (Recommended)

```bash
# Clone repository
git clone <repo-url>
cd aml_agentic_system

# Build Docker image
docker-compose build

# Run quick experiment
docker-compose run aml-system ./run_quick.sh

# Verify results
ls results/
ls figures/
```

### Option 2: Native Python

```bash
# Clone repository
git clone <repo-url>
cd aml_agentic_system

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run quick experiment
./run_quick.sh

# Verify results
cat results/full_experiments.json
```

Expected output:
- `results/full_experiments.json` with metrics matching Table 1 in paper
- `figures/*.png` matching Figures 1-5 in paper

## Full Reproducibility (4-8 hours)

### Step 1: Environment Setup

```bash
# Activate environment (if using venv)
source venv/bin/activate

# Verify Python version
python --version  # Should be 3.10.x or 3.11.x

# Verify packages
pip list | grep -E "xgboost|scikit-learn|pandas|numpy"
```

### Step 2: Data Generation

```bash
# Generate synthetic transaction data
python -c "
from code.data.synthetic_generator import generate_and_save
df = generate_and_save('data/synthetic/transactions.csv', 
                       n_transactions=100000, 
                       fraud_rate=0.023, 
                       seed=42)
"

# Verify data
python -c "
import pandas as pd
df = pd.read_csv('data/synthetic/transactions.csv')
print(f'Transactions: {len(df)}')
print(f'Fraud rate: {df[\"is_fraud\"].mean():.4f}')
"
```

Expected output:
- 100,000 transactions (±100 due to sampling)
- Fraud rate: 0.023 (±0.001)

### Step 3: Run Experiments

```bash
# Run full experimental suite
./run_full.sh

# This script executes:
# 1. Train-test split (70/30 temporal)
# 2. Rule-based baseline
# 3. Isolation Forest baseline
# 4. XGBoost baseline
# 5. Full agentic system
# 6. Ablation studies
# 7. Statistical significance tests
# 8. Figure generation

# Monitor progress
tail -f logs/experiment.log  # If available
```

Expected runtime:
- CPU-only: 4-6 hours
- With GPU: 2-3 hours

### Step 4: Verify Results

```bash
# Check results file
cat results/full_experiments.json

# Verify key metrics (should match Table 1 in paper)
python -c "
import json
with open('results/full_experiments.json') as f:
    results = json.load(f)

print('F1 Scores:')
print(f\"  Rule-Based:       {results['baseline_results']['rule_based']['f1']:.3f}\")
print(f\"  Isolation Forest: {results['baseline_results']['isolation_forest']['f1']:.3f}\")
print(f\"  XGBoost:          {results['baseline_results']['xgboost']['f1']:.3f}\")
print(f\"  Agentic System:   {results['agentic_results']['f1']:.3f}\")
"
```

Expected values (±0.005 tolerance):
- Rule-Based F1: 0.495
- Isolation Forest F1: 0.531
- XGBoost F1: 0.765
- Agentic System F1: 0.869

### Step 5: Verify Figures

```bash
# List generated figures
ls -lh figures/

# Verify checksums (optional)
sha256sum figures/*.png > figures/checksums.txt
```

Expected figures:
- `eval_roc_pr.png`: ROC and PR curves (Figure 3 in paper)
- `sar_latency_throughput.png`: Performance metrics (Figure 4)
- `metrics_comparison.png`: Bar chart comparison (Figure 2)
- `explainability_annotation.png`: Annotated SAR (Figure 5)
- `system_architecture.svg`: Architecture diagram (Figure 1)

## Deterministic Results Guarantee

All results are deterministic with the following guarantees:

### Random Seeds
- Global seed: 42
- NumPy: `np.random.seed(42)`
- Scikit-learn: `random_state=42`
- XGBoost: `seed=42`

### Data Generation
```python
# Deterministic transaction generator
generator = SyntheticTransactionGenerator(
    n_transactions=100000,
    fraud_rate=0.023,
    seed=42  # Fixed seed
)
df = generator.generate()  # Always produces same data
```

### Model Training
```python
# All models use fixed random states
xgb_model = XGBClassifier(
    seed=42,
    tree_method='hist',  # Deterministic
    # ... other params
)
```

### Splits
```python
# Time-based split (deterministic)
df_sorted = df.sort_values('timestamp')  # Deterministic order
split_idx = int(len(df) * 0.7)
train = df_sorted.iloc[:split_idx]
test = df_sorted.iloc[split_idx:]
```

## Troubleshooting

### Issue: Different metrics than reported

**Check**:
1. Python version (must be 3.10 or 3.11)
2. Package versions match `requirements.txt`
3. Seed is set to 42 in all scripts
4. No GPU randomness (set `CUBLAS_WORKSPACE_CONFIG=:4096:8`)

**Resolution**:
```bash
# Verify environment
python -c "
import numpy as np
import xgboost as xgb
print(f'NumPy: {np.__version__}')
print(f'XGBoost: {xgb.__version__}')
"

# Should output:
# NumPy: 1.24.3
# XGBoost: 1.7.6
```

### Issue: Out of memory

**Resolution**:
```bash
# Reduce dataset size
python code/scripts/run_experiments.py \
    --n-transactions 50000 \  # Reduced from 100K
    --output-dir results/reduced
```

### Issue: Figures don't match paper

**Check**:
1. Results JSON is from correct experiment
2. Matplotlib version matches (3.7.2)
3. DPI setting is 300

**Resolution**:
```bash
# Regenerate figures
python code/scripts/generate_figures.py \
    --results-dir results/full_experiments \
    --output-dir figures \
    --high-dpi
```

## Checksum Verification

To verify exact reproducibility, compare checksums:

```bash
# Generate checksums
sha256sum results/full_experiments.json > checksums.txt
sha256sum figures/*.png >> checksums.txt

# Compare with reference
diff checksums.txt reproducibility/reference_checksums.txt
```

Reference checksums are provided in `reproducibility/reference_checksums.txt`.

## Alternative: Pre-computed Results

If you cannot run experiments (e.g., no compute resources), pre-computed results are available:

```bash
# Download pre-computed results
wget <repo-url>/releases/precomputed/results.tar.gz
tar -xzf results.tar.gz

# Verify checksums
sha256sum -c reproducibility/reference_checksums.txt
```

## Contact

For reproducibility issues:
- Open a GitHub issue with tag `reproducibility`
- Include: OS, Python version, error logs
- Attach: `results/full_experiments.json`, checksums

## Checklist

- [ ] Python 3.10/3.11 installed
- [ ] All dependencies installed from `requirements.txt`
- [ ] Random seed set to 42
- [ ] Data generated successfully
- [ ] Experiments completed without errors
- [ ] Results match expected values (±0.005)
- [ ] All 5 figures generated
- [ ] Checksums verified (if applicable)

## Expected Execution Time

| Component | CPU-only | With GPU | Notes |
|-----------|----------|----------|-------|
| Data generation | 2 min | 2 min | I/O bound |
| Rule-based baseline | 5 min | 5 min | No training |
| Isolation Forest | 15 min | 10 min | Moderate |
| XGBoost | 45 min | 20 min | Heavy |
| Agentic system | 60 min | 25 min | Includes SAR generation |
| Ablations | 30 min | 15 min | 3 variants |
| Statistical tests | 10 min | 10 min | Bootstrap sampling |
| Figure generation | 5 min | 5 min | Plotting |
| **Total** | **~3-4 hrs** | **~1.5 hrs** | |

Quick run (reduced dataset): ~30 minutes

---

**Last updated**: 2024-01-01
**Version**: 1.0
