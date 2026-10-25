#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_grid.py --config configs/main_fashion.yaml --output ../results/grid_fashion.csv --quick
python run_grid.py --config configs/main_har.yaml --output ../results/grid_har.csv --quick
python analyze_results.py ../results/grid_fashion.csv --out ../results/summary_fashion.csv
python analyze_results.py ../results/grid_har.csv --out ../results/summary_har.csv
