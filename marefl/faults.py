from __future__ import annotations
import numpy as np


class FaultProcess:
    """Strategy-independent fault generator.

    It is intentionally driven by a scheduling RNG separate from all recovery
    decisions, so paired methods see identical selected clients and fault traces.
    """
    def __init__(self, num_edges: int, cfg, rng: np.random.Generator):
        self.num_edges = num_edges
        self.cfg = cfg
        self.rng = rng
        self.burst = np.zeros(num_edges, dtype=bool)

    def _advance_burst(self):
        if not self.cfg.bursty_failures:
            self.burst[:] = False
            return
        for e in range(self.num_edges):
            if self.burst[e]:
                if self.rng.random() < self.cfg.burst_exit_prob:
                    self.burst[e] = False
            elif self.rng.random() < self.cfg.burst_enter_prob:
                self.burst[e] = True

    def round_faults(self, selected: list[int], edges: np.ndarray):
        self._advance_burst()
        client_status = {}
        for i in selected:
            mult = self.cfg.burst_multiplier if self.burst[edges[i]] else 1.0
            p_fail = min(0.90, self.cfg.client_failure_prob * mult)
            p_strag = min(0.90, self.cfg.straggler_prob * mult)
            u = self.rng.random()
            if u < p_fail:
                # Separate transient disconnects from hard crashes without changing
                # the overall client-failure probability.
                status = "disconnected" if self.rng.random() < self.cfg.disconnect_fraction else "failed"
            elif u < p_fail + p_strag:
                status = "straggler"
            else:
                status = "ok"
            latency = float(self.rng.lognormal(
                mean=np.log(max(self.cfg.base_latency_s, 1e-3)), sigma=self.cfg.latency_sigma
            ))
            if latency > self.cfg.deadline_s and status == "ok":
                status = "straggler"
            client_status[i] = {"status": status, "latency_s": latency}

        edge_status = {}
        for e in range(self.num_edges):
            mult = self.cfg.burst_multiplier if self.burst[e] else 1.0
            p = min(0.90, self.cfg.edge_failure_prob * mult)
            failed = bool(self.rng.random() < p)
            outage = float(self.rng.exponential(self.cfg.edge_outage_mean_s)) if failed else 0.0
            edge_status[e] = {"failed": failed, "outage_s": outage}
        return client_status, edge_status, self.burst.copy()
