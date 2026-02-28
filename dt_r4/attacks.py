"""Attack module for node-level attack objectives.

This module intentionally keeps attack-specific logic out of training and
aggregator control flow to simplify auditability and future extension.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _empty_attack_payload() -> Dict[str, float]:
    return {
        "attack_enabled": 0.0,
        "mimic_loss": float("nan"),
        "poison_loss": float("nan"),
        "final_kl_to_teacher": float("nan"),
    }


def apply_attack(
    local_model: torch.nn.Module,
    global_model: Optional[torch.nn.Module],
    data,
    ref_set: Optional[torch.Tensor],
    teacher: Optional[torch.Tensor],
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """Apply a specialized attack helper.

    Parameters
    ----------
    local_model:
        Current local model.
    global_model:
        Reserved for future non-mimic attacks.
    data:
        (x, y) tensor tuple for local training data. Used for poison-loss proxy.
    ref_set:
        Reference set for semantic mimic attack.
    teacher:
        Teacher probabilities on ref_set.
    config:
        Dict including:
        - attack_mode: str
        - adaptive_mimic_lambda: float

    Returns
    -------
    dict with:
        - attack_enabled (0/1)
        - mimic_loss
        - poison_loss
        - final_kl_to_teacher
    """
    del global_model
    cfg = config or {}
    attack_mode = str(cfg.get("attack_mode", "none")).strip().lower()

    if attack_mode != "adaptive_mimic":
        return _empty_attack_payload()

    lam = float(cfg.get("adaptive_mimic_lambda", 0.0))
    if lam <= 0:
        return _empty_attack_payload()

    if ref_set is None or teacher is None or ref_set.numel() == 0:
        return _empty_attack_payload()

    x_ref = ref_set
    q = teacher
    if x_ref.shape[0] != q.shape[0]:
        return _empty_attack_payload()

    ref_logits = local_model(x_ref)
    p = F.softmax(ref_logits, dim=1)
    eps = 1e-12
    mimic_loss = (q * (torch.log(q.clamp_min(eps)) - torch.log(p.clamp_min(eps)))).sum(
        dim=1
    ).mean()

    poison_loss = float("nan")
    x_local, y_local = None, None
    if isinstance(data, tuple) and len(data) >= 2:
        x_local, y_local = data[0], data[1]
    if x_local is not None and y_local is not None and x_local.numel() > 0:
        logits = local_model(x_local)
        poison = F.cross_entropy(logits, y_local)
        poison_loss = float(poison.item())

    return {
        "attack_enabled": 1.0,
        "mimic_loss": float(mimic_loss.item()),
        "poison_loss": float(poison_loss),
        "final_kl_to_teacher": float(mimic_loss.item()),
    }


__all__ = ["apply_attack"]

