from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="../results/summary.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    last = df.sort_values("round").groupby(
        ["strategy","dataset","seed","alpha","client_failure_prob","edge_failure_prob"],
        as_index=False
    ).tail(1)
    summary = df.groupby("strategy", as_index=False).agg(
        mean_accuracy=("accuracy","mean"),
        completion_rate=("round_completed","mean"),
        recovery_success_rate=("recovery_success_rate","mean"),
        mttr_s=("mttr_s","mean"),
        mean_bytes=("bytes_total","mean"),
        recovery_bytes=("recovery_bytes","mean"),
        control_overhead_s=("control_overhead_s","mean"),
        mean_round_latency_s=("sim_round_latency_s","mean"),
    )
    final_acc = last.groupby("strategy", as_index=False)["accuracy"].mean().rename(columns={"accuracy":"final_accuracy"})
    summary = summary.merge(final_acc, on="strategy", how="left")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
