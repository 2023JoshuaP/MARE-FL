from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marefl.config import ExperimentConfig
from marefl.simulator import MARESimulator

strategies = [
    "fedavg_drop", "fedavg_wait", "static_replacement",
    "minimal_repair", "mare_centralized", "mare_hierarchical",
]
base = ExperimentConfig.from_yaml("configs/smoke_digits.yaml")
traces = {}
for s in strategies:
    cfg = ExperimentConfig(**base.to_dict())
    cfg.strategy = s
    rows = MARESimulator(cfg).run()
    traces[s] = [(r["selected_ids"], r["client_statuses"], r["failed_edge_ids"]) for r in rows]
ref = traces[strategies[0]]
for s in strategies[1:]:
    assert traces[s] == ref, f"paired fault trace mismatch: {s}"
print("OK: all strategies received identical selected clients and injected fault traces.")
