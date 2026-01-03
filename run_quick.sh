#!/bin/bash
# Quick 30-minute experiment runner

set -e

echo "=========================================="
echo "AML Agentic System - Quick Run (30 min)"
echo "=========================================="

# Create directories
mkdir -p results/quick_run
mkdir -p data/synthetic
mkdir -p figures
mkdir -p logs

echo "[1/4] Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export RANDOM_SEED=42

echo "[2/4] Running quick experiments (reduced dataset)..."
python code/scripts/run_experiments.py \
    --seed 42 \
    --n-transactions 10000 \
    --fraud-rate 0.023 \
    --output-dir results/quick_run

echo "[3/4] Generating figures..."
python code/scripts/generate_figures.py \
    --results-dir results/quick_run \
    --output-dir figures

echo "[4/4] Running tests..."
pytest tests/ -v --tb=short

echo ""
echo "=========================================="
echo "Quick run completed successfully!"
echo "=========================================="
echo "Results: results/quick_run/"
echo "Figures: figures/"
echo "Logs: logs/"
echo ""
echo "To run full experiments (4-8 hours):"
echo "  ./run_full.sh"
echo ""
