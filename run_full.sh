#!/bin/bash
# Full experimental suite (4-8 hours)

set -e

echo "=========================================="
echo "AML Agentic System - Full Experiments"
echo "Estimated time: 4-8 hours"
echo "=========================================="

# Create directories
mkdir -p results/full_experiments
mkdir -p results/models
mkdir -p results/logs
mkdir -p data/synthetic
mkdir -p figures

echo "[1/5] Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export RANDOM_SEED=42

echo "[2/5] Running full experimental suite..."
python code/scripts/run_experiments.py \
    --seed 42 \
    --n-transactions 100000 \
    --fraud-rate 0.023 \
    --output-dir results/full_experiments

echo "[3/5] Running ablation studies..."
python code/scripts/ablation_studies.py \
    --results-dir results/full_experiments \
    --output-dir results/ablation_studies

echo "[4/5] Generating all publication figures..."
python code/scripts/generate_figures.py \
    --results-dir results/full_experiments \
    --output-dir figures \
    --high-dpi

echo "[5/5] Running complete test suite..."
pytest tests/ -v --cov=code --cov-report=html --cov-report=term

echo ""
echo "=========================================="
echo "Full experiments completed!"
echo "=========================================="
echo "Results: results/full_experiments/"
echo "Figures: figures/"
echo "Models: results/models/"
echo "Coverage: htmlcov/index.html"
echo ""
echo "To populate papers with results:"
echo "  python code/scripts/populate_paper.py"
echo ""
