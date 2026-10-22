from __future__ import annotations
from typing import Dict, Tuple
import torch


def state_delta(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]):
    return {k: new_state[k].detach().cpu() - old_state[k].detach().cpu() for k in old_state}


def add_delta(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]):
    return {k: base[k].detach().cpu() + delta[k].detach().cpu() for k in base}


def flatten_state(state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, list]:
    meta = []
    parts = []
    for k, v in state.items():
        t = v.detach().cpu().float().contiguous()
        meta.append((k, tuple(v.shape), v.dtype, t.numel()))
        parts.append(t.view(-1))
    return torch.cat(parts) if parts else torch.empty(0), meta


def unflatten_state(vec: torch.Tensor, meta: list) -> Dict[str, torch.Tensor]:
    out = {}
    off = 0
    for k, shape, dtype, n in meta:
        chunk = vec[off:off+n].view(shape)
        out[k] = chunk.to(dtype=dtype)
        off += n
    return out


def state_nbytes(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def topk_checkpoint(delta: Dict[str, torch.Tensor], density: float):
    vec, meta = flatten_state(delta)
    n = int(vec.numel())
    if n == 0:
        return delta, 0
    k = max(1, min(n, int(round(n * density))))
    if k == n:
        return delta, state_nbytes(delta)
    idx = torch.topk(vec.abs(), k=k, largest=True, sorted=False).indices
    sparse = torch.zeros_like(vec)
    sparse[idx] = vec[idx]
    # Approximate payload: float32 value + int32 index per nonzero, plus tiny metadata.
    bytes_est = int(k * 8 + 64 * len(meta))
    return unflatten_state(sparse, meta), bytes_est


def weighted_average_deltas(items):
    """items: list[(delta, weight)]"""
    if not items:
        return None
    total = float(sum(w for _, w in items))
    out = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in items[0][0].items()}
    for d, w in items:
        a = float(w) / max(total, 1e-12)
        for k in out:
            out[k] += d[k].float() * a
    return out
