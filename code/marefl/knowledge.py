from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class EntityStats:
    reliability: float
    latency_s: float
    recovery_success: float = 0.5
    observations: int = 0
    failures: int = 0
    stragglers: int = 0
    recoveries: int = 0
    successful_recoveries: int = 0


class KnowledgeBase:
    """Shared MAPE-K knowledge using online EWMA summaries only.

    No entropy, model divergence, or sample-value scoring is stored here.
    """
    def __init__(self, num_clients: int, num_edges: int, cfg):
        self.cfg = cfg
        self.clients: Dict[int, EntityStats] = {
            i: EntityStats(cfg.prior_reliability, cfg.prior_latency_s) for i in range(num_clients)
        }
        self.edges: Dict[int, EntityStats] = {
            e: EntityStats(cfg.prior_reliability, cfg.prior_latency_s) for e in range(num_edges)
        }
        self.action_counts: Dict[str, int] = {}
        self.action_success: Dict[str, int] = {}

    def _ewma(self, old: float, obs: float) -> float:
        a = self.cfg.knowledge_ewma
        return a * old + (1.0 - a) * obs

    def observe_client(self, i: int, status: str, latency_s: float):
        s = self.clients[i]
        s.observations += 1
        ok = 1.0 if status == "ok" else 0.0
        s.reliability = self._ewma(s.reliability, ok)
        s.latency_s = self._ewma(s.latency_s, latency_s)
        if status in {"failed", "disconnected"}:
            s.failures += 1
        if status == "straggler":
            s.stragglers += 1

    def observe_edge(self, e: int, failed: bool, outage_s: float):
        s = self.edges[e]
        s.observations += 1
        s.reliability = self._ewma(s.reliability, 0.0 if failed else 1.0)
        if failed:
            s.failures += 1
            s.latency_s = self._ewma(s.latency_s, outage_s)

    def record_action(self, action: str, success: bool, entity_kind: str | None = None, entity_id: int | None = None):
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        if success:
            self.action_success[action] = self.action_success.get(action, 0) + 1
        if entity_kind == "client" and entity_id is not None:
            s = self.clients[entity_id]
        elif entity_kind == "edge" and entity_id is not None:
            s = self.edges[entity_id]
        else:
            return
        s.recoveries += 1
        s.successful_recoveries += int(success)
        s.recovery_success = self._ewma(s.recovery_success, 1.0 if success else 0.0)

    def best_replacement(self, candidates: list[int]) -> int | None:
        if not candidates:
            return None
        return max(candidates, key=lambda i: (
            self.clients[i].reliability,
            -self.clients[i].latency_s,
            self.clients[i].recovery_success,
        ))

    def snapshot(self):
        return {
            "clients": {str(k): asdict(v) for k, v in self.clients.items()},
            "edges": {str(k): asdict(v) for k, v in self.edges.items()},
            "action_counts": dict(self.action_counts),
            "action_success": dict(self.action_success),
        }
