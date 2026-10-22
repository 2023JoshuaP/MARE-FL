from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Telemetry:
    entity_kind: str
    entity_id: int
    raw_state: str
    latency_s: float
    deadline_s: float
    outage_s: float = 0.0


@dataclass
class Analysis:
    state: str
    severity: float
    transient_likelihood: float


@dataclass
class Decision:
    action: str
    estimated_cost: float
    rationale: str


class Monitor:
    def collect_client(self, i: int, status: str, latency_s: float, deadline_s: float) -> Telemetry:
        return Telemetry("client", i, status, latency_s, deadline_s)

    def collect_edge(self, e: int, failed: bool, outage_s: float, deadline_s: float) -> Telemetry:
        return Telemetry("edge", e, "failed" if failed else "ok", 0.0, deadline_s, outage_s)


class Analyzer:
    def analyze(self, t: Telemetry, reliability: float) -> Analysis:
        if t.entity_kind == "client":
            if t.raw_state == "ok":
                return Analysis("Healthy", 0.0, 1.0)
            if t.raw_state == "straggler":
                sev = min(1.0, t.latency_s / max(t.deadline_s, 1e-9))
                return Analysis("Straggler", sev, reliability)
            if t.raw_state == "disconnected":
                return Analysis("Disconnected", 0.75, max(0.2, reliability))
            return Analysis("Failed", 1.0, 0.0)
        if t.raw_state == "ok":
            return Analysis("Healthy", 0.0, 1.0)
        transient = max(0.05, min(0.95, reliability))
        return Analysis("Unavailable", 1.0, transient)


class Planner:
    """Cost-based planner for MARE-FL.

    The planner works exclusively with operational state: expected latency,
    historical reliability, recovery success, and resource cost.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def _cost(self, *, time_s=0.0, comm_mb=0.0, compute_units=0.0, failure_risk=0.0, drop_units=0.0):
        c = self.cfg
        return (
            c.planner_w_time * time_s
            + c.planner_w_comm * comm_mb
            + c.planner_w_compute * compute_units
            + c.planner_w_failure * failure_risk
            + c.planner_w_drop * drop_units
        )

    def client_decision(self, analysis: Analysis, stats, model_mb: float, has_replacement: bool) -> Decision:
        if analysis.state == "Healthy":
            return Decision("ACCEPT", 0.0, "healthy update")

        choices = []
        if analysis.state == "Straggler":
            wait_s = min(self.cfg.max_wait_s, max(0.0, stats.latency_s))
            choices.append(Decision("WAIT", self._cost(time_s=wait_s, failure_risk=1-stats.reliability), "wait for late update"))
        if analysis.state == "Disconnected":
            choices.append(Decision("RETRY", self._cost(
                time_s=self.cfg.base_latency_s, comm_mb=model_mb,
                failure_risk=(1-stats.recovery_success) * 0.6), "retry transient client link"))
        if has_replacement:
            choices.append(Decision("REPLACE", self._cost(
                time_s=stats.latency_s, comm_mb=2*model_mb, compute_units=1.0,
                failure_risk=1-stats.reliability), "dispatch a replacement client"))
        choices.append(Decision("DROP", self._cost(drop_units=1.0), "continue without this contribution"))
        return min(choices, key=lambda d: d.estimated_cost)

    def edge_decision(self, analysis: Analysis, stats, model_mb: float, has_alternate_edge: bool, received_updates: int) -> Decision:
        if analysis.state == "Healthy":
            return Decision("ACCEPT", 0.0, "edge available")
        choices = []
        wait_s = min(self.cfg.max_wait_s, max(0.0, stats.latency_s))
        choices.append(Decision("WAIT_EDGE", self._cost(time_s=wait_s, failure_risk=1-stats.reliability), "wait for edge recovery"))
        if has_alternate_edge and received_updates > 0:
            choices.append(Decision("REROUTE", self._cost(
                time_s=self.cfg.reroute_latency_s,
                comm_mb=model_mb * received_updates,
                failure_risk=0.15), "retransmit cached updates to an alternate edge"))
        if received_updates >= self.cfg.min_updates_for_partial:
            choices.append(Decision("PARTIAL_AGG", self._cost(drop_units=0.6), "continue with surviving edges"))
        else:
            choices.append(Decision("DROP_EDGE", self._cost(drop_units=1.2), "drop unavailable edge contribution"))
        return min(choices, key=lambda d: d.estimated_cost)


class MAPEKController:
    def __init__(self, cfg, knowledge, level: str):
        self.cfg = cfg
        self.knowledge = knowledge
        self.level = level
        self.monitor = Monitor()
        self.analyzer = Analyzer()
        self.planner = Planner(cfg)

    @property
    def control_latency_s(self):
        return self.cfg.local_control_latency_s if self.level == "edge" else self.cfg.cloud_control_latency_s

    def decide_client(self, i: int, status: str, latency_s: float, model_mb: float, has_replacement: bool):
        t = self.monitor.collect_client(i, status, latency_s, self.cfg.deadline_s)
        stats = self.knowledge.clients[i]
        a = self.analyzer.analyze(t, stats.reliability)
        d = self.planner.client_decision(a, stats, model_mb, has_replacement)
        return t, a, d

    def decide_edge(self, e: int, failed: bool, outage_s: float, model_mb: float, has_alternate_edge: bool, received_updates: int):
        t = self.monitor.collect_edge(e, failed, outage_s, self.cfg.deadline_s)
        stats = self.knowledge.edges[e]
        a = self.analyzer.analyze(t, stats.reliability)
        d = self.planner.edge_decision(a, stats, model_mb, has_alternate_edge, received_updates)
        return t, a, d
