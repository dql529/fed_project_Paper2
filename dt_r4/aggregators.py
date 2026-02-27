import collections
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import dt_r4.config as C
from dt_r4.utils import weighted_average_aggregation


def exp_clamp(z: float) -> float:
    """Clamp exponent input to avoid overflow."""
    z = max(-20.0, min(20.0, float(z)))
    return math.exp(z)


def mean_aggregation(models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    num = max(1, len(models))
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for i, model in enumerate(models):
            sd = model.state_dict()
            for k, v in sd.items():
                if i == 0:
                    agg_dict[k] = v.detach().clone() / num
                else:
                    agg_dict[k] += v.detach() / num
    return agg_dict


def byzantine_median_aggregation(models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    if not models:
        return {}
    state_dicts = [m.state_dict() for m in models]
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k in state_dicts[0].keys():
            tensors = [sd[k].detach() for sd in state_dicts]
            device = tensors[0].device
            stacked = torch.stack([t.cpu() for t in tensors], dim=0)
            agg_dict[k] = torch.median(stacked, dim=0).values.to(device)
    return agg_dict


def median_aggregation(models: List[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    if not models:
        return {}
    state_dicts = [m.state_dict() for m in models]
    ref = state_dicts[0]
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k in ref.keys():
            stacked = torch.stack(
                [sd[k].detach().to("cpu", dtype=torch.float32) for sd in state_dicts],
                dim=0,
            )
            med = torch.median(stacked, dim=0).values
            agg_dict[k] = med.to(dtype=ref[k].dtype, device=ref[k].device)
    return agg_dict


def trimmed_mean_aggregation(models: List[torch.nn.Module], trim_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
    if not models:
        return {}
    state_dicts = [m.state_dict() for m in models]
    ref = state_dicts[0]
    m = len(state_dicts)
    trim = int(m * trim_ratio)
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k in ref.keys():
            stacked = torch.stack(
                [sd[k].detach().to("cpu", dtype=torch.float32) for sd in state_dicts],
                dim=0,
            )
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if m > 2 * trim:
                trimmed = sorted_vals[trim : m - trim]
            else:
                trimmed = sorted_vals
            mean_val = trimmed.mean(dim=0)
            agg_dict[k] = mean_val.to(dtype=ref[k].dtype, device=ref[k].device)
    return agg_dict


def _state_dict_to_cpu_float(sd: Dict[str, torch.Tensor]) -> "collections.OrderedDict[str, torch.Tensor]":
    return collections.OrderedDict(
        (k, v.detach().to("cpu", dtype=torch.float32)) for k, v in sd.items()
    )


def _flatten_state_dict(sd_cpu_float: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in sd_cpu_float.values()])


def _pairwise_distances(sd_cpu_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    vecs = [_flatten_state_dict(sd) for sd in sd_cpu_list]
    mat = torch.stack(vecs, dim=0)
    sq = (mat * mat).sum(dim=1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * (mat @ mat.t())
    d2.clamp_(min=0)
    return torch.sqrt(d2 + 1e-12)


def krum_aggregation(models: List[torch.nn.Module], f: int = 0) -> Dict[str, torch.Tensor]:
    if not models:
        return {}
    states_raw = [m.state_dict() for m in models]
    n = len(states_raw)
    if n <= 2 * f + 2:
        return median_aggregation(models)

    states_cpu = [_state_dict_to_cpu_float(sd) for sd in states_raw]
    dists = _pairwise_distances(states_cpu).numpy()

    nb_in_score = max(1, n - f - 2)
    scores = []
    for i in range(n):
        s = np.sort(dists[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(s)
    winner = int(np.argmin(scores))

    ref_sd = states_raw[winner]
    return {k: v.detach().clone() for k, v in ref_sd.items()}


def multi_krum_aggregation(models: List[torch.nn.Module], f: int = 0, m: Optional[int] = None) -> Dict[str, torch.Tensor]:
    if not models:
        return {}
    states_raw = [m.state_dict() for m in models]
    n = len(states_raw)
    if n <= 2 * f + 2:
        return median_aggregation(models)

    if m is None:
        m = max(1, n - f - 2)
    m = max(1, min(m, n))

    states_cpu = [_state_dict_to_cpu_float(sd) for sd in states_raw]
    dists = _pairwise_distances(states_cpu).numpy()

    nb_in_score = max(1, n - f - 2)
    scores = []
    for i in range(n):
        s = np.sort(dists[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(s)
    winners = np.argsort(scores)[:m].tolist()

    ref = states_raw[0]
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k in ref.keys():
            stacked = torch.stack([states_cpu[i][k] for i in winners], dim=0)
            mean_val = stacked.mean(dim=0)
            agg_dict[k] = mean_val.to(dtype=ref[k].dtype, device=ref[k].device)
    return agg_dict


def bulyan_aggregation(models: List[torch.nn.Module], f: int = 0) -> Dict[str, torch.Tensor]:
    if not models:
        return {}
    states_raw = [m.state_dict() for m in models]
    n = len(states_raw)
    if n <= 4 * f + 3:
        trim_ratio = min(0.1, f / max(1, n))
        return trimmed_mean_aggregation(models, trim_ratio=trim_ratio)

    states_cpu = [_state_dict_to_cpu_float(sd) for sd in states_raw]
    dists = _pairwise_distances(states_cpu).numpy()

    m = max(1, n - 2 * f)
    nb_in_score = max(1, n - f - 2)
    scores = []
    for i in range(n):
        s = np.sort(dists[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(s)
    winners = np.argsort(scores)[:m].tolist()

    ref = states_raw[0]
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k in ref.keys():
            stacked = torch.stack([states_cpu[i][k] for i in winners], dim=0)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if m > 2 * f:
                trimmed = sorted_vals[f : m - f]
            else:
                trimmed = sorted_vals
            mean_val = trimmed.mean(dim=0)
            agg_dict[k] = mean_val.to(dtype=ref[k].dtype, device=ref[k].device)
    return agg_dict


def _compute_weighted_topk_state(
    models: List[torch.nn.Module],
    reputations: Dict[int, float],
    top_k_ratio: float,
    rep_alpha: float,
    mix_uniform: float,
    median_gate_lambda: float,
    need_median: bool,
    weight_min: Optional[float] = None,
    weight_max: Optional[float] = None,
) -> Dict[str, object]:
    if not models:
        return {
            "models_sel": [],
            "weights": np.array([], dtype=np.float64),
            "med_sd_cpu": None,
            "meta": {},
        }

    n = len(models)
    top_k_ratio = float(top_k_ratio)
    rep_alpha = max(0.0, float(rep_alpha))
    mix_uniform = max(0.0, min(1.0, float(mix_uniform)))
    median_gate_lambda = max(0.0, float(median_gate_lambda))
    weight_min = None if weight_min is None else max(0.0, float(weight_min))
    weight_max = None if weight_max is None else max(0.0, float(weight_max))
    if (weight_min is not None) and (weight_max is not None):
        weight_max = max(weight_max, weight_min)

    ranked_all = sorted(
        ((m, float(reputations.get(m.drone_id, 0.0))) for m in models),
        key=lambda x: x[1],
        reverse=True,
    )
    rank_map = {m.drone_id: i + 1 for i, (m, _) in enumerate(ranked_all)}

    min_k = max(1, n - int(getattr(C, "MALICIOUS_NODES", 0)))
    k = max(min_k, int(math.ceil(n * top_k_ratio)))
    k = min(k, n)

    selected = ranked_all[:k]
    models_sel = [m for m, _ in selected]
    rep = np.array([max(r, 1e-12) for _, r in selected], dtype=np.float64)

    # R4-based硬门控：信誉低于阈值的直接置零
    tau_gate = float(getattr(C, "R4_AGG_TAU_GATE", 0.0))
    if tau_gate > 0.0:
        rep = np.where(rep < tau_gate, 0.0, rep)

    if (rep.sum() <= 0.0) or (not np.isfinite(rep).all()):
        w = np.ones(k, dtype=np.float64) / k
    else:
        rep_pow = rep**rep_alpha
        if (rep_pow.sum() <= 0.0) or (not np.isfinite(rep_pow).all()):
            w = np.ones(k, dtype=np.float64) / k
        else:
            w = rep_pow / rep_pow.sum()

    if mix_uniform > 0:
        w = (1.0 - mix_uniform) * w + mix_uniform * (np.ones(k) / k)
        w = w / w.sum()

    med_sd_cpu = None
    dists: List[float] = []
    gate = None
    if (median_gate_lambda > 0) or need_median:
        med_sd_cpu = {}
        ref_sd = models_sel[0].state_dict()
        for k_name in ref_sd.keys():
            stacked = torch.stack(
                [m.state_dict()[k_name].detach().cpu().float() for m in models_sel],
                dim=0,
            )
            med_sd_cpu[k_name] = torch.median(stacked, dim=0).values

    if (median_gate_lambda > 0) and (med_sd_cpu is not None):
        for m in models_sel:
            sd = m.state_dict()
            s = 0.0
            for k_name, med in med_sd_cpu.items():
                diff = sd[k_name].detach().cpu().float() - med
                s += float((diff * diff).sum().item())
            dists.append(math.sqrt(s + 1e-12))
        dist_scale = float(np.median(dists)) + 1e-12
        gate = np.exp(-median_gate_lambda * (np.array(dists) / dist_scale))
        w = w * gate
        w = w / (w.sum() + 1e-12)

    # Clamp individual weights to avoid near-hard exclusion, then renormalize
    if (weight_min is not None) or (weight_max is not None):
        lo = 0.0 if weight_min is None else weight_min
        hi = np.inf if weight_max is None else weight_max
        w = np.clip(w, lo, hi)
        s = w.sum()
        if (s <= 0.0) or (not np.isfinite(s)):
            w = np.ones_like(w) / len(w)
        else:
            w = w / s

    meta: Dict[int, Dict[str, float]] = {}
    selected_ids = {m.drone_id for m in models_sel}
    for m, rep_val in ranked_all:
        nid = m.drone_id
        entry = {
            "rep": float(rep_val),
            "rank": int(rank_map[nid]),
            "selected": nid in selected_ids,
            "weight": 0.0,
            "gate": float("nan"),
            "dist": float("nan"),
        }
        meta[nid] = entry

    for idx, m in enumerate(models_sel):
        nid = m.drone_id
        meta[nid]["weight"] = float(w[idx])
        if gate is not None:
            meta[nid]["gate"] = float(gate[idx])
            meta[nid]["dist"] = float(dists[idx])

    return {
        "models_sel": models_sel,
        "weights": w,
        "med_sd_cpu": med_sd_cpu,
        "meta": meta,
    }


def weighted_topk_median_aggregation(
    models: List[torch.nn.Module],
    reputations: Dict[int, float],
    top_k_ratio: float,
    rep_alpha: float,
    mix_uniform: float,
    median_blend: float,
    median_gate_lambda: float,
    weight_min: Optional[float] = None,
    weight_max: Optional[float] = None,
    state: Optional[Dict[str, object]] = None,
) -> Dict[str, torch.Tensor]:
    if not models:
        return {}

    median_blend = max(0.0, min(1.0, float(median_blend)))
    if state is None:
        state = _compute_weighted_topk_state(
            models,
            reputations,
            top_k_ratio=top_k_ratio,
            rep_alpha=rep_alpha,
            mix_uniform=mix_uniform,
            median_gate_lambda=median_gate_lambda,
            need_median=median_blend > 0,
            weight_min=weight_min,
            weight_max=weight_max,
        )

    models_sel = state["models_sel"]
    w = state["weights"]
    med_sd_cpu = state["med_sd_cpu"]

    if not models_sel:
        return {}

    w_t = torch.tensor(w, dtype=torch.float32)

    ref_sd = models_sel[0].state_dict()
    agg_dict: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k_name in ref_sd.keys():
            stacked = torch.stack(
                [m.state_dict()[k_name].detach() for m in models_sel], dim=0
            )
            stacked_f = stacked.float()
            weights = w_t.to(device=stacked_f.device).view(
                -1, *([1] * (stacked_f.dim() - 1))
            )
            mean_val = (stacked_f * weights).sum(dim=0)
            if median_blend > 0:
                med_val = med_sd_cpu[k_name].to(device=stacked_f.device)
                out = (1.0 - median_blend) * mean_val + median_blend * med_val
            else:
                out = mean_val
            agg_dict[k_name] = out.to(dtype=stacked.dtype, device=stacked.device)
    return agg_dict


def normalize_reputations(reputations: Dict[int, float], num_nodes: int) -> Dict[int, float]:
    total = float(sum(reputations.values()))
    if total <= 0:
        return {i: 1.0 / num_nodes for i in range(num_nodes)}
    return {i: float(reputations.get(i, 0.0)) / total for i in range(num_nodes)}


__all__ = [
    "exp_clamp",
    "mean_aggregation",
    "byzantine_median_aggregation",
    "median_aggregation",
    "trimmed_mean_aggregation",
    "krum_aggregation",
    "multi_krum_aggregation",
    "bulyan_aggregation",
    "weighted_topk_median_aggregation",
    "normalize_reputations",
    "_compute_weighted_topk_state",
]
