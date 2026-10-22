#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Restrict each process to a single math thread to avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Execute full parallel grid evaluation across datasets (64 workers)
python run_grid_parallel.py --config configs/main_fashion.yaml --output ../results/grid_fashion_full.csv --workers 64
python run_grid_parallel.py --config configs/main_har.yaml --output ../results/grid_har_full.csv --workers 64

# Generate summary metrics
python analyze_results.py ../results/grid_fashion_full.csv --out ../results/summary_fashion_full.csv
python analyze_results.py ../results/grid_har_full.csv --out ../results/summary_har_full.csv
