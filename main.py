from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from marefl.config import ExperimentConfig
from marefl.simulator import MARESimulator


def main():
    ap = argparse.ArgumentParser(description="MARE-FL reproducible simulator")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--strategy", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    if args.strategy: cfg.strategy = args.strategy
    if args.seed is not None: cfg.seed = args.seed
    if args.output: cfg.output_csv = args.output
    sim = MARESimulator(cfg)
    rows = sim.run()
    out = Path(cfg.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    out.with_name(out.stem + "_knowledge.json").write_text(
        json.dumps(sim.knowledge.snapshot(), indent=2), encoding="utf-8"
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
