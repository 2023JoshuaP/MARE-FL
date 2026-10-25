from __future__ import annotations
import json, random, time
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import load_dataset, dirichlet_partition
from .models import build_model
from .compression import state_delta, add_delta, state_nbytes, weighted_average_deltas
from .faults import FaultProcess
from .knowledge import KnowledgeBase
from .mapek import MAPEKController
from .baselines import minimal_repair_candidate
from .metrics import evaluate


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_local(global_state, model_factory, subset, cfg, train_seed: int):
    model = model_factory().to(cfg.device)
    model.load_state_dict(global_state, strict=True)
    model.train()
    gen = torch.Generator().manual_seed(int(train_seed))
    loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward(); opt.step()
    wall = time.perf_counter() - t0
    new_state = clone_state(model)
    return state_delta(new_state, global_state), len(subset), wall


class MARESimulator:
    def __init__(self, cfg):
        self.cfg = cfg
        seed_all(cfg.seed)
        # Scheduling/fault RNG is isolated from recovery decisions.
        self.rng_sched = np.random.default_rng(cfg.seed)
        self.rng_recovery = np.random.default_rng(cfg.seed + 17171)
        self.trainset, self.testset, in_dim, classes, kind = load_dataset(cfg.dataset)
        self.client_sets, self.hists = dirichlet_partition(
            self.trainset, cfg.num_clients, cfg.dirichlet_alpha, cfg.seed
        )
        self.edges = np.asarray([i % cfg.num_edges for i in range(cfg.num_clients)], dtype=int)
        self.model_factory = lambda: build_model(kind, in_dim, cfg.hidden_dim, classes)
        self.model = self.model_factory().to(cfg.device)
        self.faults = FaultProcess(cfg.num_edges, cfg, self.rng_sched)
        self.knowledge = KnowledgeBase(cfg.num_clients, cfg.num_edges, cfg)
        self.edge_controllers = {
            e: MAPEKController(cfg, self.knowledge, level="edge") for e in range(cfg.num_edges)
        }
        self.cloud_controller = MAPEKController(cfg, self.knowledge, level="cloud")
        self.rows = []

    def _select_clients(self):
        k = min(self.cfg.clients_per_round, self.cfg.num_clients)
        return self.rng_sched.choice(self.cfg.num_clients, size=k, replace=False).astype(int).tolist()

    def _replacement_pool(self, primary, selected, used, same_edge_only=False):
        pool = [i for i in range(self.cfg.num_clients) if i not in selected and i not in used]
        if same_edge_only:
            pool = [i for i in pool if self.edges[i] == self.edges[primary]]
        return pool

    def _choose_replacement(self, primary, selected, used, mode):
        if mode == "hierarchical":
            pool = self._replacement_pool(primary, selected, used, same_edge_only=True)
            choice = self.knowledge.best_replacement(pool)
            if choice is not None:
                return choice
            pool = self._replacement_pool(primary, selected, used, same_edge_only=False)
            return self.knowledge.best_replacement(pool)
        pool = self._replacement_pool(primary, selected, used, same_edge_only=False)
        if mode == "minimal":
            return minimal_repair_candidate(primary, pool, self.hists)
        if mode == "random":
            return int(self.rng_recovery.choice(pool)) if pool else None
        return self.knowledge.best_replacement(pool)

    def _record_recovery(self, action, success, rec_time, rec_bytes, kind, entity_id, acc):
        if action in {"ACCEPT", "DROP", "DROP_EDGE", "PARTIAL_AGG"}:
            return
        acc["attempts"] += 1
        acc["successes"] += int(success)
        acc["time_sum"] += rec_time if success else 0.0
        acc["bytes"] += int(rec_bytes)
        self.knowledge.record_action(action, success, kind, entity_id)

    def _execute_client(self, i, status, latency, full_state, selected, used, model_bytes, rnd, strategy, acc, actions):
        cfg = self.cfg
        model_mb = model_bytes / (1024.0 * 1024.0)
        control_s = 0.0
        extra_down = extra_up = 0
        extra_trainings = 0
        wasted_compute = 0.0

        # Healthy updates do not invoke adaptation.
        if status == "ok":
            d, n, wall = train_local(full_state, self.model_factory, self.client_sets[i], cfg, cfg.seed*100000 + rnd*1000 + i)
            return (d, n, wall, latency, extra_down, model_bytes, extra_trainings, wasted_compute, "ACCEPT", control_s)

        repl = None
        action = "DROP"
        decision_cost = None

        if strategy == "fedavg_drop":
            action = "DROP"
        elif strategy == "fedavg_wait":
            if status == "straggler" and latency <= cfg.deadline_s + cfg.static_wait_s:
                action = "WAIT"
            elif status == "disconnected":
                action = "RETRY"
            else:
                action = "DROP"
        elif strategy == "static_replacement":
            repl = self._choose_replacement(i, selected, used, "random")
            action = "REPLACE" if repl is not None else "DROP"
        elif strategy == "minimal_repair":
            repl = self._choose_replacement(i, selected, used, "minimal")
            action = "REPLACE" if repl is not None else "DROP"
        elif strategy in {"mare_centralized", "mare_hierarchical"}:
            if strategy == "mare_hierarchical":
                controller = self.edge_controllers[int(self.edges[i])]
                repl_mode = "hierarchical"
            else:
                controller = self.cloud_controller
                repl_mode = "knowledge"
            candidate = self._choose_replacement(i, selected, used, repl_mode)
            _, analysis, decision = controller.decide_client(i, status, latency, model_mb, candidate is not None)
            action = decision.action
            decision_cost = decision.estimated_cost
            control_s += controller.control_latency_s
            if action == "REPLACE":
                repl = candidate
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        success = False
        recovery_time = control_s
        recovery_bytes = 0
        result = None

        if action == "WAIT":
            # A wait succeeds only if the realized delay remains within the allowed wait horizon.
            success = latency <= cfg.deadline_s + cfg.max_wait_s
            recovery_time += min(latency, cfg.max_wait_s)
            if success:
                d, n, wall = train_local(full_state, self.model_factory, self.client_sets[i], cfg, cfg.seed*100000 + rnd*1000 + i)
                result = (d, n, wall)
        elif action == "RETRY":
            reliability = self.knowledge.clients[i].reliability
            p = min(0.98, cfg.retry_success_prob * (0.5 + 0.5 * reliability))
            success = bool(self.rng_recovery.random() < p)
            recovery_time += cfg.base_latency_s
            recovery_bytes += model_bytes
            extra_down += model_bytes
            if success:
                d, n, wall = train_local(full_state, self.model_factory, self.client_sets[i], cfg, cfg.seed*100000 + rnd*1000 + i)
                result = (d, n, wall)
                extra_up += model_bytes
                recovery_bytes += model_bytes
        elif action == "REPLACE" and repl is not None:
            used.add(repl)
            rel = self.knowledge.clients[repl].reliability
            success = bool(self.rng_recovery.random() < max(0.05, min(0.99, rel)))
            recovery_time += self.knowledge.clients[repl].latency_s
            extra_down += model_bytes
            recovery_bytes += model_bytes
            extra_trainings += 1
            if success:
                d, n, wall = train_local(full_state, self.model_factory, self.client_sets[repl], cfg, cfg.seed*100000 + rnd*1000 + repl)
                result = (d, n, wall)
                extra_up += model_bytes
                recovery_bytes += model_bytes
            else:
                # Failed dispatch is counted as wasted recovery work without training the model.
                wasted_compute += 0.0
        elif action == "DROP":
            success = False

        actions.append({
            "scope": "client", "id": int(i), "state": status, "action": action,
            "success": bool(success), "cost": decision_cost,
            "replacement": int(repl) if repl is not None else None,
        })
        self._record_recovery(action, success, recovery_time, recovery_bytes, "client", i, acc)

        if result is None:
            return (None, 0, 0.0, recovery_time, extra_down, extra_up, extra_trainings, wasted_compute, action, control_s)
        d, n, wall = result
        return (d, n, wall, recovery_time + latency, extra_down, extra_up, extra_trainings, wasted_compute, action, control_s)

    def run(self):
        cfg = self.cfg
        full_state = clone_state(self.model)
        model_bytes = state_nbytes(full_state)
        model_mb = model_bytes / (1024.0 * 1024.0)

        for rnd in range(1, cfg.rounds + 1):
            selected = self._select_clients()
            statuses, edge_status, burst = self.faults.round_faults(selected, self.edges)

            bytes_down = model_bytes * len(selected)
            bytes_up_client = 0
            bytes_edge_cloud = 0
            local_wall = 0.0
            wasted_compute = 0.0
            extra_trainings = 0
            control_overhead = 0.0
            event_latencies = []
            actions = []
            recovery = {"attempts": 0, "successes": 0, "time_sum": 0.0, "bytes": 0}
            used_replacements = set()
            edge_items: Dict[int, List] = {e: [] for e in range(cfg.num_edges)}

            failed_clients = disconnected = stragglers = 0
            accepted_updates = 0

            for i in selected:
                st = statuses[i]["status"]
                lat = float(statuses[i]["latency_s"])
                failed_clients += int(st == "failed")
                disconnected += int(st == "disconnected")
                stragglers += int(st == "straggler")

                out = self._execute_client(
                    i, st, lat, full_state, selected, used_replacements,
                    model_bytes, rnd, cfg.strategy, recovery, actions
                )
                d, n, wall, evlat, bdown, bup, extra, waste, action, ctrl = out
                bytes_down += bdown; bytes_up_client += bup
                local_wall += wall; wasted_compute += waste; extra_trainings += extra
                control_overhead += ctrl; event_latencies.append(evlat)
                if d is not None:
                    edge_items[int(self.edges[i])].append((d, n, i))
                    accepted_updates += 1

            # Monitor/update operational knowledge only after the decision for this event,
            # preventing the controller from seeing the current fault before planning.
            for i in selected:
                self.knowledge.observe_client(i, statuses[i]["status"], float(statuses[i]["latency_s"]))

            edge_aggs = {}
            edge_weights = {}
            for e, items in edge_items.items():
                if items:
                    edge_aggs[e] = weighted_average_deltas([(d, n) for d, n, _ in items])
                    edge_weights[e] = float(sum(n for _, n, _ in items))

            cloud_items = []
            edge_failures = 0
            rerouted_edges = 0
            dropped_edge_updates = 0

            for e, agg in edge_aggs.items():
                failed = bool(edge_status[e]["failed"])
                outage = float(edge_status[e]["outage_s"])
                edge_failures += int(failed)
                count_updates = len(edge_items[e])

                if not failed:
                    bytes_edge_cloud += model_bytes
                    cloud_items.append((agg, edge_weights[e]))
                    continue

                if cfg.strategy == "fedavg_wait":
                    action = "WAIT_EDGE" if outage <= cfg.static_wait_s else "DROP_EDGE"
                    ctrl = 0.0
                    decision_cost = None
                elif cfg.strategy == "static_replacement":
                    action = "REROUTE" if cfg.num_edges > 1 else "DROP_EDGE"
                    ctrl = 0.0
                    decision_cost = None
                elif cfg.strategy in {"fedavg_drop", "minimal_repair"}:
                    action = "DROP_EDGE"; ctrl = 0.0; decision_cost = None
                else:
                    _, analysis, decision = self.cloud_controller.decide_edge(
                        e, True, outage, model_mb, cfg.num_edges > 1, count_updates
                    )
                    action = decision.action; ctrl = self.cloud_controller.control_latency_s
                    decision_cost = decision.estimated_cost
                    control_overhead += ctrl

                success = False
                rec_time = ctrl
                rec_bytes = 0
                if action == "WAIT_EDGE":
                    success = outage <= cfg.max_wait_s
                    rec_time += min(outage, cfg.max_wait_s)
                    if success:
                        bytes_edge_cloud += model_bytes
                        rec_bytes += model_bytes
                        cloud_items.append((agg, edge_weights[e]))
                    else:
                        dropped_edge_updates += count_updates
                        wasted_compute += sum(0.0 for _ in edge_items[e])
                elif action == "REROUTE":
                    success = cfg.num_edges > 1
                    rec_time += cfg.reroute_latency_s
                    if success:
                        # Cached local updates are retransmitted to another edge.
                        reroute_bytes = model_bytes * count_updates
                        bytes_edge_cloud += model_bytes
                        bytes_up_client += reroute_bytes
                        rec_bytes += reroute_bytes + model_bytes
                        rerouted_edges += 1
                        cloud_items.append((agg, edge_weights[e]))
                    else:
                        dropped_edge_updates += count_updates
                elif action in {"PARTIAL_AGG", "DROP_EDGE"}:
                    dropped_edge_updates += count_updates
                actions.append({
                    "scope": "edge", "id": int(e), "state": "failed", "action": action,
                    "success": bool(success), "cost": decision_cost, "outage_s": outage,
                })
                self._record_recovery(action, success, rec_time, rec_bytes, "edge", e, recovery)

            for e in range(cfg.num_edges):
                self.knowledge.observe_edge(e, bool(edge_status[e]["failed"]), float(edge_status[e]["outage_s"]))

            global_delta = weighted_average_deltas(cloud_items)
            round_completed = global_delta is not None
            if round_completed:
                full_state = add_delta(full_state, global_delta)
                self.model.load_state_dict(full_state, strict=True)

            ev = evaluate(self.model, self.testset, cfg.device)
            total_fault_events = failed_clients + disconnected + stragglers + edge_failures
            mttr = recovery["time_sum"] / max(recovery["successes"], 1)
            total_bytes = bytes_down + bytes_up_client + bytes_edge_cloud
            row = {
                "round": rnd,
                "strategy": cfg.strategy,
                "dataset": cfg.dataset,
                "seed": cfg.seed,
                "alpha": cfg.dirichlet_alpha,
                "client_failure_prob": cfg.client_failure_prob,
                "edge_failure_prob": cfg.edge_failure_prob,
                "accuracy": ev["accuracy"],
                "test_loss": ev["loss"],
                "selected": len(selected),
                "accepted_updates": accepted_updates,
                "failed_clients": failed_clients,
                "disconnected_clients": disconnected,
                "stragglers": stragglers,
                "edge_failures": edge_failures,
                "fault_events": total_fault_events,
                "recovery_attempts": recovery["attempts"],
                "successful_recoveries": recovery["successes"],
                "recovery_success_rate": recovery["successes"] / max(recovery["attempts"], 1),
                "mttr_s": mttr,
                "round_completed": int(round_completed),
                "bytes_down": int(bytes_down),
                "bytes_up_client": int(bytes_up_client),
                "bytes_edge_cloud": int(bytes_edge_cloud),
                "recovery_bytes": int(recovery["bytes"]),
                "bytes_total": int(total_bytes),
                "extra_trainings": extra_trainings,
                "local_wall_s": local_wall,
                "wasted_compute_s": wasted_compute,
                "control_overhead_s": control_overhead,
                "sim_round_latency_s": (max(event_latencies) if event_latencies else 0.0) + control_overhead,
                "rerouted_edges": rerouted_edges,
                "dropped_edge_updates": dropped_edge_updates,
                "burst_edges": int(np.sum(burst)),
                "mean_client_reliability": float(np.mean([s.reliability for s in self.knowledge.clients.values()])),
                "mean_edge_reliability": float(np.mean([s.reliability for s in self.knowledge.edges.values()])),
                "selected_ids": json.dumps(selected),
                "client_statuses": json.dumps({str(i): statuses[i]["status"] for i in selected}, sort_keys=True),
                "failed_edge_ids": json.dumps([int(e) for e, x in edge_status.items() if x["failed"]]),
                "actions": json.dumps(actions, sort_keys=True),
            }
            self.rows.append(row)
            print(
                f"round={rnd:03d} strategy={cfg.strategy:18s} acc={row['accuracy']:.4f} "
                f"faults={total_fault_events} rec={recovery['successes']}/{recovery['attempts']} "
                f"bytes={total_bytes}"
            )
        return self.rows
