#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ../results/ablations
# Main architecture ablation: centralized vs hierarchical.
python main.py --config configs/main_fashion.yaml --strategy mare_hierarchical --seed 3 --output ../results/ablations/hierarchical.csv
python main.py --config configs/main_fashion.yaml --strategy mare_centralized --seed 3 --output ../results/ablations/centralized.csv
# Knowledge ablation: effectively forget history after each observation.
python main.py --config configs/ablation_no_history.yaml --strategy mare_hierarchical --seed 3 --output ../results/ablations/no_history.csv
# Planner ablation: static recovery policy.
python main.py --config configs/main_fashion.yaml --strategy static_replacement --seed 3 --output ../results/ablations/static_plan.csv
