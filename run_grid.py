from __future__ import annotations
import argparse, copy
from pathlib import Path
import pandas as pd
from marefl.config import ExperimentConfig
from marefl.simulator import MARESimulator

STRATEGIES = [
    "fedavg_drop", "fedavg_wait", "static_replacement",
    "minimal_repair", "mare_centralized", "mare_hierarchical",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="../results/grid.csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    base = ExperimentConfig.from_yaml(args.config)
    seeds = [3, 7, 11] if not args.quick else [3]
    client_ps = [0.0, 0.1, 0.3, 0.5] if not args.quick else [0.1, 0.3]
    edge_ps = [0.0, 0.05, 0.15] if not args.quick else [0.05]
    alphas = [0.1, 0.5, 1.0] if not args.quick else [0.5]
    rows = []
    for strategy in STRATEGIES:
        for seed in seeds:
            for pc in client_ps:
                for pe in edge_ps:
                    for alpha in alphas:
                        cfg = copy.deepcopy(base)
                        cfg.strategy = strategy; cfg.seed = seed
                        cfg.client_failure_prob = pc; cfg.edge_failure_prob = pe
                        cfg.dirichlet_alpha = alpha
                        print("\n###", strategy, seed, pc, pe, alpha)
                        rows.extend(MARESimulator(cfg).run())
                        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
