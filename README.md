# MARE-FL: MAPE-K-Driven Adaptive Resilience for Federated Learning Across the Cloud–Edge Continuum

---

## Overview

MARE-FL provides a self-adaptive fault-management framework for Federated Learning (FL) deployments across heterogeneous cloud–edge environments. Rather than applying static heuristics to participant failures (such as unconditionally dropping or waiting for stragglers), MARE-FL leverages a Monitor-Analyze-Plan-Execute over Knowledge (MAPE-K) autonomic control loop.

The controller dynamically selects cost-optimal recovery actions (including adaptive wait, retry, replacement, edge rerouting, and partial aggregation) based on runtime operational telemetry and estimated recovery costs.

### Main Contributions
- Decoupled Autonomic Control: Independent execution of local edge loops for rapid client fault handling and global cloud loops for edge-level outages.
- Paired Fault-Injection Methodology: Deterministic and isolated random generation ensuring identical, fair fault traces across all evaluated strategies.
- High-Throughput Parallel Simulator: Multiprocessing grid runner optimized to execute paired simulation rounds efficiently across CPU clusters.

---

## Repository Structure

```text
├── marefl/                 # Core library: MAPE-K loop, simulator, fault models, networks
├── configs/                # Experiment configurations (Fashion-MNIST, UCI-HAR, ablations)
├── scripts/                # Shell scripts for environment setup, smoke tests, and benchmarks
├── tests/                  # Unit tests for MAPE-K logic and invariant validation
├── main.py                 # Single simulation entrypoint
├── run_grid.py             # Sequential grid runner
├── run_grid_parallel.py    # High-throughput parallel grid runner
├── analyze_results.py      # Statistical summarization and aggregation script
├── plot_results.py         # Publication figure generation utilities
└── requirements.txt        # Python package dependencies
```

---

## Requirements and Installation

### Prerequisites
- Linux OS (tested on Ubuntu 22.04 and 24.04 LTS)
- Python 3.10+
- PyTorch 2.0+

### Setup
Clone the repository and set up the local environment:

```bash
git clone https://github.com/2023JoshuaP/MARE-FL.git
cd MARE-FL

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, run the automated setup script on Ubuntu:
```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Run unit tests and a fast smoke test to verify system integrity:

```bash
pytest tests/
bash scripts/smoke_test.sh
```

---

## Reproducing Results

### 1. Running Benchmark Experiments
Execute the complete paired evaluation across Fashion-MNIST and UCI-HAR:

```bash
# Runs the full parallel simulation grid across all failure distributions
bash scripts/run_main_experiments_parallel.sh
```

### 2. Generating Summaries and Metrics
To compute aggregate metrics (`final_accuracy`, `latency`, `control_overhead`, etc.):

```bash
python analyze_results.py ../results/grid_fashion_full.csv --out ../results/summary_fashion_full.csv
python analyze_results.py ../results/grid_har_full.csv --out ../results/summary_har_full.csv
```

### 3. Generating Figures
To reproduce the evaluation plots:

```bash
python plot_results.py
```

---

## Summary of Results

Key comparative metrics obtained under non-IID data distributions and mixed failure conditions:

| Strategy | Accuracy (HAR) | Accuracy (Fashion-MNIST) | Mean Round Latency | Control Overhead |
| :--- | :---: | :---: | :---: | :---: |
| FedAvg-Drop | 70.0% | 77.6% | 0.24s | 0.00s |
| FedAvg-Wait | 74.3% | 80.2% | 0.35s | 0.00s |
| Minimal Repair | 73.5% | 80.7% | 0.36s | 0.00s |
| MARE-Centralized | 75.3% | 79.9% | 0.87s | 0.39s |
| **MARE-Hierarchical** | **75.6%** | **81.3%** | **0.49s** | **0.08s** |

Findings indicate that MARE-Hierarchical attains the highest final accuracy while reducing control overhead by 4.3x and latency by 43% compared to centralized adaptation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
