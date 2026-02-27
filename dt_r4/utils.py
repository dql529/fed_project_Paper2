"""
dt_r4/utils.py

通用小工具：聚合、相关、ramp 等。
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import R4_WARMUP_ROUNDS, R4_RAMP_ROUNDS, R4_WARMUP_ALPHA


def weighted_average_aggregation(models, reputations: Dict[int, float]):
    total_rep = sum(reputations.values()) or 1.0
    agg_dict = {}
    for i, model in enumerate(models):
        w = reputations.get(model.drone_id, 0.0) / total_rep
        sd = model.state_dict()
        for k, v in sd.items():
            if i == 0:
                agg_dict[k] = v.clone() * w
            else:
                agg_dict[k] += v * w
    return agg_dict


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def exponential_decay(x: float, a: float = 0.8) -> float:
    return float(np.exp(-a * x))


def corr_safe(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ramp_alpha(target_alpha: float, current_round: int) -> float:
    """
    R4 alpha warmup+ramp:
      - target_alpha <= 0: 完全禁用R4 => alpha=0
      - 否则：前 R4_WARMUP_ROUNDS 用 base，再线性上升到 target.
    """
    target_alpha = float(target_alpha)
    if target_alpha <= 0:
        return 0.0
    if current_round <= R4_WARMUP_ROUNDS:
        return float(R4_WARMUP_ALPHA)
    if R4_RAMP_ROUNDS <= 0:
        return target_alpha

    delta = current_round - R4_WARMUP_ROUNDS
    if delta >= R4_RAMP_ROUNDS:
        return target_alpha

    ratio = delta / R4_RAMP_ROUNDS
    return float(R4_WARMUP_ALPHA + (target_alpha - R4_WARMUP_ALPHA) * ratio)


def alpha_key(r4_target: float, r4_used: float, sig: int = 6) -> str:
    """A short, stable string key for plotting/merging.

    Example: r4_target=4, r4_used=1 -> "4@1".
    We use a limited number of significant digits to avoid float artifacts.
    """

    fmt = f"{{:.{sig}g}}@{{:.{sig}g}}"
    return fmt.format(float(r4_target), float(r4_used))
