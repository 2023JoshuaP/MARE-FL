#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
python main.py --config configs/smoke_digits.yaml --strategy mare_hierarchical --output results/smoke_mare_hierarchical.csv
python main.py --config configs/smoke_digits.yaml --strategy mare_centralized --output results/smoke_mare_centralized.csv
python scripts/check_paired_faults.py
