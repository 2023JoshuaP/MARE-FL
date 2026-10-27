"""Rebuild the numbers, tables and figures of the MARE-FL paper from the raw grid CSVs.

Usage (from the root of the repository):
    python make_paper_results.py results/grid_fashion_full.csv.gz results/grid_har_full.csv.gz \
        --out results/paper --plots

Inputs are the CSV files written by run_grid.py or run_grid_parallel.py (one row per round),
either plain (.csv) or compressed (.csv.gz).
Final accuracy is the test accuracy of the last round of each run, averaged over the 108 scenarios.
Outputs (in --out):
    table2.csv           Table II plus final accuracy, MTTR, control time and round latency
    paired_tests.csv     wins / ties / losses, median paired difference and Wilcoxon p-value
    by_failure_rate.csv  values per client failure probability (Fig. 2 and Section VI text)
    action_mix.csv       share of each client action per failure probability (Fig. 3)
    state_action.csv     action chosen for each fault state (Section VI-C)
    fig2.png, fig3.png   only with --plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RUN_KEYS = ["strategy", "dataset", "seed", "alpha", "client_failure_prob", "edge_failure_prob"]
SCENARIO = ["dataset", "seed", "alpha", "client_failure_prob", "edge_failure_prob"]
NAMES = {
    "fedavg_drop": "FedAvg-Drop",
    "fedavg_wait": "FedAvg-Wait",
    "static_replacement": "Static Replacement",
    "minimal_repair": "Minimal Repair",
    "mare_centralized": "MARE-Centralized",
    "mare_hierarchical": "MARE-Hierarchical",
    "fixed_rule": "Fixed rule (best replacement)",
    "fixed_rule_random": "Fixed rule (random replacement)",
}
MAIN = ["fedavg_drop", "fedavg_wait", "static_replacement", "minimal_repair", "mare_centralized", "mare_hierarchical"]
PAIRS = [
    ("mare_hierarchical", "minimal_repair"),
    ("mare_hierarchical", "static_replacement"),
    ("mare_centralized", "minimal_repair"),
    ("mare_centralized", "static_replacement"),
    ("mare_centralized", "mare_hierarchical"),
    ("mare_centralized", "fedavg_wait"),
    ("mare_centralized", "fixed_rule"),
    ("mare_centralized", "fixed_rule_random"),
    ("fixed_rule_random", "static_replacement"),
]


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run (50 rounds)."""
    df = df.sort_values("round")
    runs = df.groupby(RUN_KEYS).agg(
        selected=("selected", "sum"),
        accepted=("accepted_updates", "sum"),
        dropped_at_edge=("dropped_edge_updates", "sum"),
        attempts=("recovery_attempts", "sum"),
        successes=("successful_recoveries", "sum"),
        bytes_total=("bytes_total", "sum"),
        control_s=("control_overhead_s", "sum"),
        round_latency_s=("sim_round_latency_s", "mean"),
        round_completion=("round_completed", "mean"),
        mttr_round_mean_s=("mttr_s", "mean"),
        final_accuracy=("accuracy", "last"),
    ).reset_index()
    # Effective updates as reported in the paper: updates (original or recovered) received by the edge.
    runs["effective_edge"] = runs.accepted / runs.selected
    # Stricter version: also removes updates lost when their edge fails and is not recovered.
    runs["effective_global"] = (runs.accepted - runs.dropped_at_edge) / runs.selected
    return runs


def table2(runs: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, s), r in runs.groupby(["dataset", "strategy"]):
        rr = rounds[(rounds.dataset == ds) & (rounds.strategy == s)]
        ok = rr.successful_recoveries > 0
        # mttr_s is time / successes per round, so time = mttr_s * successes.
        mttr_pooled = (rr.mttr_s[ok] * rr.successful_recoveries[ok]).sum() / max(rr.successful_recoveries.sum(), 1)
        rows.append({
            "dataset": ds,
            "method": NAMES.get(s, s),
            "runs": len(r),
            "effective_edge": r.effective_edge.mean(),
            "effective_global": r.effective_global.mean(),
            "recovery_success": r.successes.sum() / r.attempts.sum() if r.attempts.sum() else np.nan,
            "traffic_gb_per_run": r.bytes_total.mean() / 1e9,
            "final_accuracy": r.final_accuracy.mean(),
            "mttr_round_mean_s": r.mttr_round_mean_s.mean(),  # value used in the paper
            "mttr_pooled_s": mttr_pooled,
            "control_s_per_run": r.control_s.mean(),
            "round_latency_s": r.round_latency_s.mean(),
            "round_completion": r.round_completion.mean(),
        })
    order = {v: i for i, v in enumerate(NAMES.values())}
    out = pd.DataFrame(rows)
    return out.sort_values(["dataset", "method"], key=lambda c: c.map(order) if c.name == "method" else c)


def paired_tests(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["effective_edge", "effective_global", "final_accuracy"]:
        piv = runs.pivot_table(index=SCENARIO, columns="strategy", values=metric).reset_index()
        for ds, p in piv.groupby("dataset"):
            for a, b in PAIRS:
                if a not in p or b not in p:
                    continue
                d = (p[a] - p[b]).dropna()
                if d.empty:
                    continue
                try:
                    pval = wilcoxon(p[a], p[b]).pvalue
                except ValueError:  # all differences are zero
                    pval = np.nan
                rows.append({"dataset": ds, "metric": metric, "method": NAMES[a], "versus": NAMES[b],
                             "scenarios": len(d), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                             "losses": int((d < 0).sum()), "median_diff": d.median(), "wilcoxon_p": pval})
    return pd.DataFrame(rows)


def by_failure_rate(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["dataset", "strategy", "client_failure_prob"])
    out = g.agg(effective_edge=("effective_edge", "mean"), effective_global=("effective_global", "mean"),
                successes=("successes", "sum"), attempts=("attempts", "sum"),
                control_s_per_run=("control_s", "mean"), round_latency_s=("round_latency_s", "mean"),
                round_completion=("round_completion", "mean")).reset_index()
    out["recovery_success"] = out.successes / out.attempts.replace(0, np.nan)
    out["method"] = out.strategy.map(NAMES)
    return out.drop(columns=["successes", "attempts"])


def action_tables(rounds: pd.DataFrame):
    mix, state = [], []
    for (ds, s, pc), r in rounds.groupby(["dataset", "strategy", "client_failure_prob"]):
        counts, by_state = {}, {}
        for cell in r.actions:
            for a in json.loads(cell):
                if a["scope"] != "client" or a["action"] == "ACCEPT":
                    continue
                counts[a["action"]] = counts.get(a["action"], 0) + 1
                key = (a["state"], a["action"])
                by_state[key] = by_state.get(key, 0) + 1
        total = sum(counts.values())
        for act, n in counts.items():
            mix.append({"dataset": ds, "method": NAMES.get(s, s), "client_failure_prob": pc,
                        "action": act, "events": n, "share": n / total})
        for (st, act), n in by_state.items():
            tot_st = sum(v for (x, _), v in by_state.items() if x == st)
            state.append({"dataset": ds, "method": NAMES.get(s, s), "client_failure_prob": pc,
                          "state": st, "action": act, "events": n, "share_of_state": n / tot_st})
    return pd.DataFrame(mix), pd.DataFrame(state)


def plots(fr: pd.DataFrame, mix: pd.DataFrame, out: Path, dataset: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = fr[fr.dataset == dataset]
    fig, ax = plt.subplots(figsize=(4, 3))
    for s in MAIN:
        name = NAMES[s]
        r = d[d.strategy == s].sort_values("client_failure_prob")
        if len(r):
            ax.plot(r.client_failure_prob, r.effective_edge, marker="o", label=name)
    ax.set_xlabel("Client failure probability $p_c$")
    ax.set_ylabel("Effective updates")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig2.png", dpi=200)

    m = mix[(mix.dataset == dataset) & (mix.method == "MARE-Centralized")]
    piv = m.pivot_table(index="client_failure_prob", columns="action", values="share", fill_value=0)
    piv = piv.reindex(columns=[c for c in ["WAIT", "RETRY", "REPLACE", "DROP"] if c in piv.columns])
    fig, ax = plt.subplots(figsize=(4, 3))
    piv.plot(kind="bar", stacked=True, ax=ax, width=0.7, rot=0)
    ax.set_xlabel("Client failure probability $p_c$")
    ax.set_ylabel("Share of fault events")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out / "fig3.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="grid CSV files (.csv or .csv.gz)")
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--plot-dataset", default="fashionmnist")
    args = ap.parse_args()

    rounds = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = per_run(rounds)
    counts = runs.groupby(["dataset", "strategy"]).size()
    if counts.nunique() > 1:
        print("WARNING: methods have a different number of runs (a run may have failed):")
        print(counts.to_string())
    t2 = table2(runs, rounds)
    tests = paired_tests(runs)
    fr = by_failure_rate(runs)
    mix, state = action_tables(rounds)

    t2.to_csv(out / "table2.csv", index=False)
    tests.to_csv(out / "paired_tests.csv", index=False)
    fr.to_csv(out / "by_failure_rate.csv", index=False)
    mix.to_csv(out / "action_mix.csv", index=False)
    state.to_csv(out / "state_action.csv", index=False)
    if args.plots:
        plots(fr, mix, out, args.plot_dataset)

    pd.set_option("display.width", 200)
    cols = ["dataset", "method", "runs", "effective_edge", "recovery_success", "traffic_gb_per_run", "final_accuracy"]
    print(t2[cols].round(3).to_string(index=False))
    print()
    print(tests[tests.metric == "effective_edge"].round(4).to_string(index=False))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
