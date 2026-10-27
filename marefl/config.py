from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict
import yaml


@dataclass
class ExperimentConfig:
    dataset: str = "digits"
    seed: int = 7
    rounds: int = 10
    num_clients: int = 12
    num_edges: int = 3
    clients_per_round: int = 6
    dirichlet_alpha: float = 0.5
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.03
    hidden_dim: int = 64

    strategy: str = "mare_hierarchical"

    client_failure_prob: float = 0.20
    disconnect_fraction: float = 0.45
    straggler_prob: float = 0.10
    edge_failure_prob: float = 0.08
    bursty_failures: bool = True
    burst_enter_prob: float = 0.08
    burst_exit_prob: float = 0.35
    burst_multiplier: float = 2.5

    deadline_s: float = 1.0
    base_latency_s: float = 0.12
    latency_sigma: float = 0.55
    edge_outage_mean_s: float = 1.50

    knowledge_ewma: float = 0.75
    prior_reliability: float = 0.90
    prior_latency_s: float = 0.15
    retry_success_prob: float = 0.70
    static_wait_s: float = 0.60
    max_wait_s: float = 1.00

    planner_w_time: float = 1.00
    planner_w_comm: float = 0.18
    planner_w_compute: float = 0.35
    planner_w_failure: float = 1.10
    planner_w_drop: float = 1.60

    local_control_latency_s: float = 0.015
    cloud_control_latency_s: float = 0.090
    reroute_latency_s: float = 0.120

    min_updates_for_partial: int = 2
    device: str = "cpu"
    output_csv: str = "results/run.csv"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        valid = {k: v for k, v in raw.items() if k in cls.__annotations__}
        return cls(**valid)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
