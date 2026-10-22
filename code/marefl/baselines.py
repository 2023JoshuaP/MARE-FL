from __future__ import annotations
import numpy as np


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def minimal_repair_candidate(primary: int, candidates: list[int], hists: np.ndarray) -> int | None:
    """A compact approximation of the prior minimal-repair idea for baseline use.

    MARE-FL itself never calls this data-distribution matching routine.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda j: js_divergence(hists[primary], hists[j]))
