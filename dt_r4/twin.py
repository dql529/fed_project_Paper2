"""
dt_r4/twin.py

孪生失配（Phase 0）与孪生输出（Phase 2 mask/KL）相关函数。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import dt_r4.config as C
from .config import (
    TWIN_DRIFT_MAX,
    TWIN_LOGIT_BIAS_MAX,
    ENFORCE_TWIN_MONOTONIC,
    TWIN_MIN_ACC_DROP,
    TWIN_CALIB_SCALE,
    TWIN_CALIB_MAX_ITERS,
    TWIN_MISMATCH_SPECS,
)
from .runtime import device
from .models import TeacherNet


@dataclass
class TwinMismatchContext:
    name: str
    desc: str
    drift: float  # relative to ref_std
    drop_dims: List[int]  # fixed dims
    logit_bias: float  # relative to logits.std
    drift_vec: torch.Tensor  # shape [n_feat]
    logit_bias_dir: torch.Tensor  # shape [2], fixed direction
    n_feat: int


@torch.no_grad()
def make_r4_mask_and_weights(probs: torch.Tensor):
    """
    probs: [N,2] softmax(probabilities) used to define "confident-normal" region.
    返回：
      mask: [N] bool
      weights: [N] float (conf * mask)
      conf: [N] float
      pred: [N] long
    """
    conf, pred = probs.max(dim=1)

    mask = torch.ones_like(conf, dtype=torch.bool)
    if C.R4_USE_ONLY_CONFIDENT:
        mask &= conf >= C.R4_CONF_THRESH
    if C.R4_ONLY_NORMAL:
        mask &= pred == C.NORMAL_CLASS_INDEX

    weights = conf * mask.float()
    # 额外记录权重集中度，便于判断是否“只盯着少数 easy 点”
    weight_sum = float(weights.sum().item())
    weight_top1_frac = (
        float(weights.max().item()) / weight_sum if weight_sum > 1e-12 else float("nan")
    )
    # 可选调试输出
    if getattr(C, "DEBUG_R4_MASK_LOG", False):
        mask_frac = float(mask.float().mean().item())
        print(
            f"[R4 mask] frac={mask_frac:.3f}, top1_frac={weight_top1_frac:.3f}, "
            f"conf_thresh={getattr(C, 'R4_CONF_THRESH', 0.0):.2f}, only_normal={getattr(C, 'R4_ONLY_NORMAL', True)}"
        )

    # 将集中度塞进 weights 的属性，供上层读取
    weights._r4_top1_frac = weight_top1_frac  # type: ignore[attr-defined]
    return mask, weights, conf, pred, weight_top1_frac


def build_twin_mismatch_context(
    ref_x: torch.Tensor, spec: Dict[str, Any]
) -> TwinMismatchContext:
    """
    完全确定性（不依赖 seed、没有 random.choice），确保跨 seed 方差≈0。
    drift 方向固定：全 +1。
    """
    n_feat = int(ref_x.shape[1])
    ref_std = ref_x.std(dim=0).detach().clamp(min=1e-6)

    drift = float(spec.get("drift", 0.0))
    drift = max(0.0, min(TWIN_DRIFT_MAX, drift))

    # 固定方向：全 +1
    drift_vec = drift * ref_std * torch.ones_like(ref_std)

    drop_dims = [int(d) for d in spec.get("drop_dims", [])]
    drop_dims = [d for d in drop_dims if 0 <= d < n_feat]

    logit_bias = float(spec.get("logit_bias", 0.0))
    logit_bias = max(0.0, min(TWIN_LOGIT_BIAS_MAX, logit_bias))

    # 固定类偏置方向：class0 +, class1 -
    logit_bias_dir = torch.tensor([1.0, -1.0], dtype=torch.float32, device=ref_x.device)

    return TwinMismatchContext(
        name=str(spec.get("name", "")),
        desc=str(spec.get("desc", "")),
        drift=drift,
        drop_dims=drop_dims,
        logit_bias=logit_bias,
        drift_vec=drift_vec.to(ref_x.device),
        logit_bias_dir=logit_bias_dir,
        n_feat=n_feat,
    )


@torch.no_grad()
def get_twin_logits(
    teacher_model: TeacherNet, x: torch.Tensor, ctx: TwinMismatchContext
) -> torch.Tensor:
    """
    用“失配后的孪生”产生 logits（数字孪生输出）。
    注入：固定漂移 / 固定丢维 / 固定类偏置（确定性）
    """
    x_twin = x
    if ctx.drift > 0:
        x_twin = x_twin + ctx.drift_vec
    if len(ctx.drop_dims) > 0:
        x_twin = x_twin.clone()
        x_twin[:, ctx.drop_dims] = 0.0

    logits = teacher_model(x_twin)

    # 固定类偏置：class0 +b, class1 -b，其中 b = logit_bias * logits.std()
    if ctx.logit_bias > 0:
        std = logits.detach().std().clamp(min=1e-6)
        bias = ctx.logit_bias_dir * (ctx.logit_bias * std)
        logits = logits + bias  # broadcast [2]

    return logits


@torch.no_grad()
def evaluate_twin_metrics(
    teacher_model: TeacherNet,
    df: pd.DataFrame,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    ctx: TwinMismatchContext,
):
    # 当前数据集（noise pool）评估
    df_x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32, device=device)
    df_y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long, device=device)

    softmax = nn.Softmax(dim=1)

    # ========== 1) 用当前 ctx 的 twin 来算 twin 本身在 noise/ref 上的性能 ==========
    logits_noise = get_twin_logits(teacher_model, df_x, ctx)
    probs_noise = softmax(logits_noise)
    preds_noise = probs_noise.argmax(dim=1)
    acc_noise = accuracy_score(df_y.cpu(), preds_noise.cpu())
    f1_noise = f1_score(df_y.cpu(), preds_noise.cpu())

    logits_ref = get_twin_logits(teacher_model, ref_x, ctx)
    probs_ref = softmax(logits_ref)
    preds_ref = probs_ref.argmax(dim=1)
    acc_ref = accuracy_score(ref_y.cpu(), preds_ref.cpu())
    f1_ref = f1_score(ref_y.cpu(), preds_ref.cpu())

    max_conf = probs_ref.max(dim=1).values.mean().item()
    entropy = (-probs_ref * (probs_ref + 1e-12).log()).sum(dim=1).mean().item()

    # ========== 2) R4 的 mask 覆盖率：强制用 base twin (L0) 定义 normal prior 区域 ==========
    base_ctx = build_twin_mismatch_context(ref_x, TWIN_MISMATCH_SPECS[0])  # L0
    logits_mask = get_twin_logits(teacher_model, ref_x, base_ctx)
    probs_mask = softmax(logits_mask)

    mask, _, _, _, _ = make_r4_mask_and_weights(probs_mask)
    mask_frac = mask.float().mean().item()
    mask_cnt = int(mask.sum().item())

    return {
        "Twin_Acc_Noise": float(acc_noise),
        "Twin_F1_Noise": float(f1_noise),
        "Twin_Acc_Ref": float(acc_ref),
        "Twin_F1_Ref": float(f1_ref),
        "Twin_Conf_Ref": float(max_conf),
        "Twin_Entropy_Ref": float(entropy),
        "Twin_Dropped_Features": (
            ",".join(map(str, ctx.drop_dims)) if ctx.drop_dims else ""
        ),
        "Twin_Drift": float(ctx.drift),
        "Twin_LogitBias": float(ctx.logit_bias),
        "Twin_R4MaskFrac": float(mask_frac),
        "Twin_R4MaskCount": int(mask_cnt),
    }


def _twin_acc_ref(
    teacher_model: TeacherNet,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    spec: Dict[str, Any],
) -> float:
    ctx = build_twin_mismatch_context(ref_x, spec)
    logits = get_twin_logits(teacher_model, ref_x, ctx)
    preds = logits.argmax(dim=1)
    return float(accuracy_score(ref_y.cpu(), preds.cpu()))


def calibrate_twin_mismatch_specs(
    teacher_model: TeacherNet,
    ref_x: torch.Tensor,
    ref_y: torch.Tensor,
    specs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    自动校准：确保 Twin_Acc_Ref(L0) > Twin_Acc_Ref(L1) > Twin_Acc_Ref(L2)
    若初始 specs 已满足，则不改。
    若不满足：放大 L1/L2 的 drift 与 logit_bias（保持 drop_dims 不变）。
    """
    if not ENFORCE_TWIN_MONOTONIC:
        return specs
    if len(specs) != 3:
        return specs

    s0 = copy.deepcopy(specs[0])
    s1 = copy.deepcopy(specs[1])
    s2 = copy.deepcopy(specs[2])

    acc0 = _twin_acc_ref(teacher_model, ref_x, ref_y, s0)

    # L1
    acc1 = _twin_acc_ref(teacher_model, ref_x, ref_y, s1)
    it = 0
    while acc1 >= acc0 - TWIN_MIN_ACC_DROP and it < TWIN_CALIB_MAX_ITERS:
        s1["drift"] = min(
            TWIN_DRIFT_MAX, float(s1.get("drift", 0.0)) * TWIN_CALIB_SCALE
        )
        s1["logit_bias"] = min(
            TWIN_LOGIT_BIAS_MAX, float(s1.get("logit_bias", 0.0)) * TWIN_CALIB_SCALE
        )
        acc1 = _twin_acc_ref(teacher_model, ref_x, ref_y, s1)
        it += 1

    # L2：确保至少不弱于 L1 的强度，再找 acc2 < acc1
    s2["drift"] = max(float(s2.get("drift", 0.0)), float(s1.get("drift", 0.0)) * 1.05)
    s2["logit_bias"] = max(
        float(s2.get("logit_bias", 0.0)), float(s1.get("logit_bias", 0.0)) * 1.05
    )
    s2["drift"] = min(TWIN_DRIFT_MAX, float(s2["drift"]))
    s2["logit_bias"] = min(TWIN_LOGIT_BIAS_MAX, float(s2["logit_bias"]))

    acc2 = _twin_acc_ref(teacher_model, ref_x, ref_y, s2)
    it = 0
    while acc2 >= acc1 - TWIN_MIN_ACC_DROP and it < TWIN_CALIB_MAX_ITERS:
        s2["drift"] = min(
            TWIN_DRIFT_MAX, float(s2.get("drift", 0.0)) * TWIN_CALIB_SCALE
        )
        s2["logit_bias"] = min(
            TWIN_LOGIT_BIAS_MAX, float(s2.get("logit_bias", 0.0)) * TWIN_CALIB_SCALE
        )
        acc2 = _twin_acc_ref(teacher_model, ref_x, ref_y, s2)
        it += 1

    if not (acc0 > acc1 and acc1 > acc2):
        raise RuntimeError(
            "Twin mismatch monotonicity check failed.\n"
            f"  acc0={acc0:.4f}, acc1={acc1:.4f}, acc2={acc2:.4f}\n"
            "请手动调大 L1/L2 的 drift / logit_bias 或调整 drop_dims，使 Twin_Acc_Ref 严格单调递减。"
        )

    print("\n=== Twin mismatch calibrated (deterministic) ===")
    print(
        f"L0: drift={s0['drift']:.4f}, drop={s0.get('drop_dims', [])}, logit_bias={s0['logit_bias']:.4f}, Twin_Acc_Ref={acc0:.4f}"
    )
    print(
        f"L1: drift={s1['drift']:.4f}, drop={s1.get('drop_dims', [])}, logit_bias={s1['logit_bias']:.4f}, Twin_Acc_Ref={acc1:.4f}"
    )
    print(
        f"L2: drift={s2['drift']:.4f}, drop={s2.get('drop_dims', [])}, logit_bias={s2['logit_bias']:.4f}, Twin_Acc_Ref={acc2:.4f}"
    )

    return [s0, s1, s2]


@torch.no_grad()
def compute_r4_mask_stats(
    teacher_model: TeacherNet,
    ref_x: torch.Tensor,
    twin_contexts: List[TwinMismatchContext],
) -> Dict[str, Dict[str, float]]:
    """
    给 backfill 用：计算每个 twin 的 mask 覆盖率（按当前 R4_CONF_THRESH / normal-only 策略）
    """
    softmax = nn.Softmax(dim=1)
    stats: Dict[str, Dict[str, float]] = {}
    for ctx in twin_contexts:
        twin_logits_ref = get_twin_logits(teacher_model, ref_x, ctx)
        twin_probs = softmax(twin_logits_ref)

        mask, _, _, _, _ = make_r4_mask_and_weights(twin_probs)

        stats[ctx.name] = {
            "frac": float(mask.float().mean().item()),
            "count": float(mask.sum().item()),
        }
    return stats
