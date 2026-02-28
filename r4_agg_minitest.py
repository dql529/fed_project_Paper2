import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

import dt_r4.config as C
from dt_r4 import aggregators as agg
from dt_r4.data import (
    build_noise_variants_fixed,
    get_teacher_model,
    load_and_clean_csv,
    load_node_splits,
    load_reference_data,
    sample_audit_set,
    sample_reference_subset,
)
from dt_r4.federated import CentralServer, DroneNode
from dt_r4.runtime import device, set_seeds
from dt_r4.twin import build_twin_mismatch_context, get_twin_logits
from dt_r4.utils import weighted_average_aggregation
from dt_r4.config import R2_IGNORE_MALICIOUS


EXPERIMENT_GROUPS = [
    "base",
    "sdt",
    "tau",
    "server_val",
    "krum",
    "mimic",
    "refaudit",
]

REQUIRED_ROUNDS_COLS = [
    "seed",
    "round",
    "attack_mode",
    "dt_level",
    "method",
    "mal_nodes",
    "tau_gate",
    "S_DT",
    "S_DT_ratio",
    "num_valid",
    "valid_ratio",
    "fallback_flag",
    "W_mal",
    "benign_pass_rate",
    "malicious_pass_rate",
    "benign_admitted_weight_mass",
]

REQUIRED_NODES_COLS = [
    "seed",
    "round",
    "node_id",
    "is_malicious",
    "R4",
    "KL_q_p",
    "Rep",
    "passed_gate",
    "pi",
    "R2",
    "R2_source",
]


def _normalize_cli_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _coerce_int_list(values: Sequence) -> List[int]:
    return [int(x) for x in values]


def _coerce_float_list(values: Sequence) -> List[float]:
    return [float(x) for x in values]


def _coerce_int_matrix(values: Sequence) -> List[int]:
    out: List[int] = []
    for x in values:
        if x is None:
            continue
        out.append(int(x))
    return out


def _coerce_float_matrix(values: Sequence) -> List[float]:
    out: List[float] = []
    for x in values:
        if x is None:
            continue
        out.append(float(x))
    return out


def _infer_group(
    *,
    method: str,
    attack: str,
    dt_level: str,
    tau_grid: Sequence[float],
    ref_grid: Sequence[int],
    audit_grid: Sequence[int],
) -> str:
    if method == "server_val":
        return "server_val"
    if method in {"krum", "bulyan", "byzantine_median"}:
        return "krum"
    if attack == "adaptive_mimic":
        return "mimic"
    if len(ref_grid) > 1 or len(audit_grid) > 1:
        return "refaudit"
    if len(tau_grid) > 1:
        return "tau"
    if method == "weighted" and str(dt_level).upper() != "D0":
        return "sdt"
    return "base"


def _tensor_from_df(df):
    x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32, device=device)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long, device=device)
    return x, y


def _macro_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    num_classes = max(
        int(tgt_np.max()) + 1 if tgt_np.size else 0,
        int(pred_np.max()) + 1 if pred_np.size else 0,
    )
    if num_classes <= 1:
        return 1.0 if pred_np.size == 0 else float(np.mean(pred_np == tgt_np))

    f1s = []
    for c in range(num_classes):
        tp = np.sum((pred_np == c) & (tgt_np == c))
        fp = np.sum((pred_np == c) & (tgt_np != c))
        fn = np.sum((pred_np != c) & (tgt_np == c))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
        elif tp == 0 and fp == 0 and fn == 0:
            f1s.append(1.0)
        else:
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))


def eval_model(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        f1 = _macro_f1(pred, y)
    return float(acc), float(f1)


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def _nanmean_or_nan(values: Sequence[float]) -> float:
    arr = np.array([float(x) for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_div(numer: float, denom: float) -> float:
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(numer / denom)


def _parse_int_list_csv(
    value: Optional[str], default: Optional[Sequence[int]] = None
) -> List[Optional[int]]:
    if value is None:
        return list(default) if default is not None else [None]
    text = str(value).strip()
    if text == "":
        return list(default) if default is not None else [None]
    out: List[Optional[int]] = []
    for x in text.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        return [None] if default is None else list(default)
    return out


def _parse_float_list_csv(
    value: Optional[str], default: Optional[Sequence[float]] = None
) -> List[float]:
    if value is None:
        return (
            list(default)
            if default is not None
            else [float(getattr(C, "R4_GATE_TAU", 0.5))]
        )
    text = str(value).strip()
    if text == "":
        return (
            list(default)
            if default is not None
            else [float(getattr(C, "R4_GATE_TAU", 0.5))]
        )
    out: List[float] = []
    for x in text.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        return (
            [float(getattr(C, "R4_GATE_TAU", 0.5))]
            if default is None
            else list(default)
        )
    return out


def _parse_cli_list_ints(value: Optional[str], default: Optional[Sequence[int]]) -> List[int]:
    return [v for v in _parse_int_list_csv(value, default=default) if v is not None]


def _parse_cli_list_floats(
    value: Optional[str], default: Optional[Sequence[float]]
) -> List[float]:
    return list(_parse_float_list_csv(value, default=default))


def _normalize_csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _row_hashes(x: torch.Tensor, y: torch.Tensor, decimals: int = 6) -> set[str]:
    if x is None or x.numel() == 0:
        return set()
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().reshape(-1, 1)
    payload = np.concatenate([x_np, y_np], axis=1)
    payload = np.round(payload.astype(np.float32), decimals=decimals)
    return {
        hashlib.md5(row.tobytes()).hexdigest() for row in payload
    }


def _overlap_size(left: set[str], right: set[str]) -> int:
    if not left or not right:
        return 0
    return int(len(left.intersection(right)))


def _split_leakage_summary(node_data_objects, r4_ref_x, r4_ref_y, audit_x, audit_y):
    ref_hashes = _row_hashes(r4_ref_x, r4_ref_y)
    audit_hashes = _row_hashes(audit_x, audit_y)

    train_hashes: set[str] = set()
    test_hashes: set[str] = set()
    for item in node_data_objects:
        if item is None:
            continue
        train = getattr(item["train"], "x", None)
        train_y = getattr(item["train"], "y", None)
        test = getattr(item["test"], "x", None)
        test_y = getattr(item["test"], "y", None)
        if train is not None and train_y is not None:
            train_hashes |= _row_hashes(train, train_y)
        if test is not None and test_y is not None:
            test_hashes |= _row_hashes(test, test_y)

    split_summary = {
        "ref_rows": int(len(ref_hashes)),
        "audit_rows": int(len(audit_hashes)),
        "local_train_rows": int(len(train_hashes)),
        "local_test_rows": int(len(test_hashes)),
        "inter_ref_train": _overlap_size(ref_hashes, train_hashes),
        "inter_ref_test": _overlap_size(ref_hashes, test_hashes),
        "inter_audit_train": _overlap_size(audit_hashes, train_hashes),
        "inter_audit_test": _overlap_size(audit_hashes, test_hashes),
    }

    return split_summary


def _raise_if_leaked(name: str, summary: Dict[str, int]) -> None:
    bad = {
        k: v
        for k, v in summary.items()
        if k.startswith("inter_") and int(v) > 0
    }
    if bad:
        raise ValueError(f"[{name}] data leakage detected: {bad}")


def _mean_if_finite(values: Sequence[float]) -> float:
    arr = np.array([float(x) for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _flatten_model(model: torch.nn.Module) -> torch.Tensor:
    sd = model.state_dict()
    return torch.cat(
        [v.detach().reshape(-1).to("cpu", dtype=torch.float32) for v in sd.values()]
    )


def _pairwise_dist_matrix(vecs: List[torch.Tensor]) -> torch.Tensor:
    mat = torch.stack(vecs, dim=0)
    sq = (mat * mat).sum(dim=1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * (mat @ mat.t())
    d2.clamp_(min=0)
    return torch.sqrt(d2 + 1e-12)


def _select_krum_indices(models: List[torch.nn.Module], f: int) -> List[int]:
    n = len(models)
    if n <= 2 * f + 2:
        return list(range(n))
    vecs = [_flatten_model(m) for m in models]
    dmat = _pairwise_dist_matrix(vecs).numpy()
    nb_in_score = max(1, n - f - 2)
    scores: List[float] = []
    for i in range(n):
        s = np.sort(dmat[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(float(s))
    return [int(np.argmin(scores))]


def _select_bulyan_indices(models: List[torch.nn.Module], f: int) -> List[int]:
    n = len(models)
    if n <= 2 * f + 2:
        return list(range(n))
    vecs = [_flatten_model(m) for m in models]
    dmat = _pairwise_dist_matrix(vecs).numpy()
    nb_in_score = max(1, n - f - 2)
    scores: List[float] = []
    for i in range(n):
        s = np.sort(dmat[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(float(s))
    m = max(1, n - 2 * f)
    return [int(i) for i in np.argsort(scores)[:m]]


def _geo_aggregation(models: List[torch.nn.Module], mode: str):
    mode = str(mode).lower().strip()
    if mode == "median":
        return agg.median_aggregation(models)
    return agg.mean_aggregation(models)


def _need_r2_data(
    use_reputation: bool, audit_size: Optional[int], audit_size_used: int
):
    if not use_reputation:
        return "none"
    if audit_size is None:
        return "local_test"
    if int(audit_size) <= 0:
        return "none"
    return "server_audit" if audit_size_used > 0 else "none"


def _compute_reference_effective_counts(server: CentralServer, ref_size: int) -> Tuple[int, int, int]:
    # true number of available samples after DT corruption and masking
    n_ref_eff = int(max(0, int(ref_size)))
    if server.r4_weights is None:
        return n_ref_eff, 0, n_ref_eff
    mask = getattr(server, "r4_mask", None)
    if mask is None or not torch.is_tensor(mask):
        n_valid = int(mask.nelement()) if hasattr(mask, "nelement") else n_ref_eff
    else:
        n_valid = int(mask.sum().item())
    n_masked = max(0, n_ref_eff - n_valid)
    return n_ref_eff, n_masked, n_valid


def run_once(
    method: str,
    rounds: int,
    deploy_variant: int,
    scenario: str,
    seed: int = 0,
    tau_gate: Optional[float] = None,
    dt_support_min: Optional[int] = None,
    fallback_mode: Optional[str] = None,
    adaptive_mimic_lambda: float = 0.0,
    ref_size: Optional[int] = None,
    audit_size: Optional[int] = None,
    diag: Optional[List[Dict]] = None,
    round_rows: Optional[List[Dict]] = None,
    node_rows: Optional[List[Dict]] = None,
    meta: Optional[Dict] = None,
    return_extra: bool = True,
) -> Tuple[Dict, Dict]:
    method = str(method).strip().lower()
    set_seeds(seed)

    effective_tau = float(
        tau_gate if tau_gate is not None else getattr(C, "R4_GATE_TAU", 0.5)
    )
    dt_support_min = int(
        dt_support_min
        if dt_support_min is not None
        else getattr(C, "DT_SUPPORT_MIN", 10)
    )
    fallback_mode = str(fallback_mode or getattr(C, "GEO_FALLBACK_AGG", "mean")).lower()

    attack_mode = (meta or {}).get("attack", getattr(C, "MAL_ATTACK_MODE", "none"))
    dt_level = (meta or {}).get("dt_level", getattr(C, "_DT_ACTIVE_LEVEL", "D0"))
    method_detail = str(meta or {}).get("method_detail", "")
    used_attack_mode = str(attack_mode)
    uses_dt = False

    use_reputation = method.startswith("weighted")
    server_val = method == "server_val"

    rep_ablation = method if use_reputation else ""
    if rep_ablation in {"weighted", "weighted_full"}:
        rep_ablation = "weighted_full"
    if use_reputation and rep_ablation not in {
        "weighted_full",
        "weighted_r4only",
        "weighted_r2only",
        "weighted_nogate",
        "server_val",
    }:
        rep_ablation = "weighted_full"

    if method in {"krum", "bulyan"}:
        if C.MALICIOUS_NODES < 1:
            return (
                {
                    "eval_acc": float("nan"),
                    "eval_f1": float("nan"),
                    "final_acc": float("nan"),
                    "final_f1": float("nan"),
                    "w_mal": float("nan"),
                    "skipped": True,
                    "skip_reason": "theoretical condition violated",
                    "tau_gate": effective_tau,
                    "lambda_m": adaptive_mimic_lambda,
                    "ref_size": int(ref_size) if ref_size is not None else -1,
                    "audit_size": int(audit_size) if audit_size is not None else -1,
                },
                {},
            )
        if method == "krum" and C.NUM_NODES < 2 * C.MALICIOUS_NODES + 3:
            return (
                {
                    "eval_acc": float("nan"),
                    "eval_f1": float("nan"),
                    "final_acc": float("nan"),
                    "final_f1": float("nan"),
                    "w_mal": float("nan"),
                    "skipped": True,
                    "skip_reason": "theoretical condition violated",
                    "tau_gate": effective_tau,
                    "lambda_m": adaptive_mimic_lambda,
                    "ref_size": int(ref_size) if ref_size is not None else -1,
                    "audit_size": int(audit_size) if audit_size is not None else -1,
                },
                {},
            )
        if method == "bulyan" and C.NUM_NODES < 4 * C.MALICIOUS_NODES + 3:
            return (
                {
                    "eval_acc": float("nan"),
                    "eval_f1": float("nan"),
                    "final_acc": float("nan"),
                    "final_f1": float("nan"),
                    "w_mal": float("nan"),
                    "skipped": True,
                    "skip_reason": "theoretical condition violated",
                    "tau_gate": effective_tau,
                    "lambda_m": adaptive_mimic_lambda,
                    "ref_size": int(ref_size) if ref_size is not None else -1,
                    "audit_size": int(audit_size) if audit_size is not None else -1,
                },
                {},
            )

    if use_reputation:
        if rep_ablation == "weighted_r4only":
            ablation_config = "R4"
        elif rep_ablation == "weighted_r2only":
            ablation_config = "R2"
        elif rep_ablation == "weighted_nogate":
            ablation_config = "R2,R3,R4"
        else:
            ablation_config = "R2,R3,R4"
        need_r4 = "R4" in {x.strip() for x in ablation_config.split(",") if x.strip()}
    else:
        ablation_config = ""
        need_r4 = False

    if method == "server_val":
        uses_dt = False
        method_detail = "server_val_audit_loss"
    elif method.startswith("weighted") and not (audit_size is None):
        # weighted baseline keeps DT support and fallback on R4 gate
        uses_dt = (need_r4 and method in {"weighted", "weighted_full"}) and str(dt_level).upper() != "D0"
        if not method_detail:
            method_detail = "weighted"
    elif method.startswith("weighted"):
        if not method_detail:
            method_detail = "weighted"

    orig_tau = float(getattr(C, "R4_GATE_TAU", 0.5))
    orig_level = getattr(C, "_DT_ACTIVE_LEVEL", None)
    orig_attack = C.MAL_ATTACK_MODE
    orig_mal_nodes = C.MALICIOUS_NODES

    if use_reputation and rep_ablation == "weighted_nogate":
        C.R4_GATE_TAU = 0.0
    elif method != "server_val":
        C.R4_GATE_TAU = effective_tau

    try:
        C._DT_ACTIVE_LEVEL = dt_level
        C.MAL_ATTACK_MODE = attack_mode

        node_data_objects, _ = load_node_splits(
            C.NUM_NODES,
            C.CSV_PATH,
            seed=seed,
            malicious_nodes=C.MALICIOUS_NODES if C.PRE_SPLIT_POISON else 0,
            attack_mode=attack_mode,
            label_flip_ratio=float(getattr(C, "LABEL_FLIP_RATIO", 0.0)),
            pre_split_poison=bool(getattr(C, "PRE_SPLIT_POISON", False)),
        )

        ref_x_full, ref_y_full = load_reference_data(C.GLOBAL_REF_CSV)
        r4_ref_x, r4_ref_y = sample_reference_subset(
            ref_x_full,
            ref_y_full,
            int(ref_size) if ref_size is not None else None,
            seed=seed + 200,
        )

        if dt_level is not None and (need_r4 or attack_mode == "adaptive_mimic"):
            dt_cfg = C.DT_MISMATCH_LEVELS.get(dt_level, C.DT_MISMATCH_LEVELS["D0"])
            keep_ratio = float(dt_cfg.get("keep_ratio", 1.0))
            noise_std = float(dt_cfg.get("noise_std", 0.0))
            if keep_ratio < 1.0 and r4_ref_x.numel() > 0:
                keep_n = max(1, int(len(r4_ref_x) * keep_ratio))
                g = torch.Generator(device=r4_ref_x.device)
                g.manual_seed(seed + 201)
                idx = torch.randperm(
                    len(r4_ref_x), generator=g, device=r4_ref_x.device
                )[:keep_n]
                r4_ref_x = r4_ref_x[idx]
                r4_ref_y = r4_ref_y[idx]
            if noise_std > 0.0:
                r4_ref_x = r4_ref_x + noise_std * torch.randn_like(r4_ref_x)

        audit_x = torch.empty((0, ref_x_full.shape[1]), device=device)
        audit_y = torch.empty((0,), dtype=torch.long, device=device)
        audit_size_used = 0
        if audit_size is not None:
            a_size = int(audit_size)
            if a_size > 0:
                audit_x, audit_y = sample_audit_set(
                    ref_x_full, ref_y_full, a_size, seed=seed + 300
                )
                audit_size_used = int(audit_x.shape[0])

        split_summary = _split_leakage_summary(
            node_data_objects, r4_ref_x, r4_ref_y, audit_x, audit_y
        )
        split_prefix = (
            f"{meta.get('attack', attack_mode) if isinstance(meta, dict) else attack_mode}-"
            f"{meta.get('dt_level', dt_level) if isinstance(meta, dict) else dt_level}-"
            f"seed{seed}"
        )
        print(
            f"[split-check] {split_prefix} "
            f"ref={split_summary['ref_rows']} "
            f"audit={split_summary['audit_rows']} "
            f"inter(ref,local_train)={split_summary['inter_ref_train']} "
            f"inter(ref,local_test)={split_summary['inter_ref_test']} "
            f"inter(audit,local_train)={split_summary['inter_audit_train']} "
            f"inter(audit,local_test)={split_summary['inter_audit_test']}"
        )
        _raise_if_leaked(split_prefix, split_summary)

        r2_source = _need_r2_data(use_reputation, audit_size, audit_size_used)

        clean_eval_x = torch.cat([d["test"].x for d in node_data_objects], dim=0)
        clean_eval_y = torch.cat([d["test"].y for d in node_data_objects], dim=0)
        train_eval_x = torch.cat([d["train"].x for d in node_data_objects], dim=0)
        train_eval_y = torch.cat([d["train"].y for d in node_data_objects], dim=0)

        if scenario.upper() == "A":
            dep_x, dep_y = clean_eval_x, clean_eval_y
        else:
            noise_variants = build_noise_variants_fixed(C.CSV_PATH, dataset_seed=123)
            deploy_variant = max(0, min(deploy_variant, len(noise_variants) - 1))
            dep_df = load_and_clean_csv(noise_variants[deploy_variant]["path"])
            dep_x, dep_y = _tensor_from_df(dep_df)

        # ---- server ----
        server = CentralServer(
            ablation_config=ablation_config,
            r4_alpha=4.0,
            use_perf_penalty=False,
        )

        teacher_probs_for_mimic = None
        preserve_teacher = False
        if need_r4 or attack_mode == "adaptive_mimic":
            teacher = get_teacher_model()
            twin_ctx = build_twin_mismatch_context(ref_x_full, C.TWIN_MISMATCH_SPECS[0])
            twin_logits_ref = (
                get_twin_logits(teacher, r4_ref_x, twin_ctx)
                if r4_ref_x.numel() > 0
                else torch.empty((0, 2), device=device)
            )
        if (
            attack_mode == "adaptive_mimic"
            and adaptive_mimic_lambda > 0
            and r4_ref_x.numel() > 0
        ):
            with torch.no_grad():
                teacher_logits = teacher(r4_ref_x)
            teacher_probs_for_mimic = F.softmax(teacher_logits / 1.0, dim=1).detach()
            server.set_teacher_reference(teacher_probs=teacher_probs_for_mimic)
            preserve_teacher = True

        if need_r4 and r4_ref_x.numel() > 0:
            server.set_twin_reference(
                twin_logits_for_probs=twin_logits_ref,
                twin_logits_for_mask=twin_logits_ref,
                temperature=1.0,
                preserve_teacher_cache=preserve_teacher,
            )

        nodes: List[DroneNode] = []
        for i in range(C.NUM_NODES):
            node = DroneNode(
                drone_id=i,
                is_malicious=(i < C.MALICIOUS_NODES),
                attack_mode=attack_mode,
                adaptive_mimic_lambda=(
                    adaptive_mimic_lambda if attack_mode == "adaptive_mimic" else 0.0
                ),
                adaptive_mimic_ref_data=(
                    r4_ref_x
                    if attack_mode == "adaptive_mimic" and adaptive_mimic_lambda > 0
                    else None
                ),
                adaptive_mimic_teacher_probs=(
                    teacher_probs_for_mimic
                    if attack_mode == "adaptive_mimic" and adaptive_mimic_lambda > 0
                    else None
                ),
            )
            node.data = node_data_objects[i]["train"]
            node.test_data = node_data_objects[i]["test"]
            server.data_age[node.drone_id] = 0
            server.reputation[node.drone_id] = 1.0
            nodes.append(node)

        last_rep: Dict[int, float] = {}
        last_comp: Dict[int, tuple] = {}
        fallback_flags: List[float] = []
        sdt_series: List[float] = []
        sdt_ratio_series: List[float] = []
        valid_ratio_series: List[float] = []
        num_masked_series: List[int] = []
        num_valid_series: List[int] = []
        benign_pass_rates: List[float] = []
        malicious_pass_rates: List[float] = []
        benign_adm_mass: List[float] = []
        w_mal_rounds: List[float] = []
        mimic_losses: List[float] = []
        poison_losses: List[float] = []
        final_kls: List[float] = []

        for r in range(1, rounds + 1):
            local_models: List[torch.nn.Module] = []
            node_perfs: Dict[int, float] = {}
            node_logits: Dict[int, torch.Tensor] = {}
            node_ref_logits: Dict[int, torch.Tensor] = {}

            for node in nodes:
                node.receive_global_model(server.global_model)
                node.train()
                node.local_model.drone_id = node.drone_id
                local_models.append(node.local_model)
                if node.is_malicious and attack_mode == "adaptive_mimic":
                    payload = node.last_attack_payload or {}
                    mimic_losses.append(float(payload.get("mimic_loss", float("nan"))))
                    poison_losses.append(float(payload.get("poison_loss", float("nan"))))
                    final_kls.append(float(payload.get("final_kl_to_teacher", float("nan"))))

                with torch.no_grad():
                    if need_r4 and r4_ref_x.numel() > 0:
                        node_logits[node.drone_id] = node.local_model(r4_ref_x)
                    if server_val and r4_ref_x.numel() > 0:
                        node_ref_logits[node.drone_id] = node.local_model(r4_ref_x)

                    if use_reputation:
                        if r2_source == "server_audit" and audit_size_used > 0:
                            test_x, test_y = audit_x, audit_y
                        elif r2_source == "none":
                            test_x = torch.empty(
                                (0, node.test_data.x.shape[1]), device=device
                            )
                            test_y = torch.empty((0,), dtype=torch.long, device=device)
                        else:
                            test_x, test_y = node.test_data.x, node.test_data.y
                            if (
                                not getattr(C, "PRE_SPLIT_POISON", False)
                                and node.is_malicious
                                and getattr(C, "POISON_LOCAL_TEST", False)
                                and attack_mode == "label_flip"
                            ):
                                test_y = test_y.clone()
                                flip_ratio = float(getattr(C, "LABEL_FLIP_RATIO", 1.0))
                                if flip_ratio > 0 and test_y.numel() > 0:
                                    mask = torch.rand_like(test_y.float()) < flip_ratio
                                    num_classes = (
                                        int(test_y.max().item()) + 1
                                        if test_y.numel()
                                        else 0
                                    )
                                    if num_classes == 2:
                                        test_y[mask] = 1 - test_y[mask]
                                    elif num_classes > 1:
                                        rand = torch.randint(
                                            0,
                                            num_classes - 1,
                                            size=(mask.sum(),),
                                            device=test_y.device,
                                        )
                                        orig = test_y[mask]
                                        rand = rand + (rand >= orig).long()
                                        test_y[mask] = rand

                        if test_x.numel() == 0:
                            acc = 1.0
                        else:
                            test_logits = node.local_model(test_x)
                            preds = test_logits.argmax(dim=1)
                            acc = (preds == test_y).float().mean().item()
                        if node.is_malicious and R2_IGNORE_MALICIOUS:
                            acc = 1.0
                        node_perfs[node.drone_id] = float(acc)

            scale = float(getattr(C, "MAL_GRAD_SCALE", 1.0))
            scale_map = getattr(C, "MAL_GRAD_SCALE_MAP", None)
            if scale_map and getattr(C, "MAL_ATTACK_MODE", None) in scale_map:
                scale = float(scale_map[getattr(C, "MAL_ATTACK_MODE")])

            if getattr(C, "MAL_ATTACK_MODE", None) and scale != 1.0:
                base_sd = server.global_model.state_dict()
                for idx, node in enumerate(nodes):
                    if not node.is_malicious:
                        continue
                    sd = local_models[idx].state_dict()
                    for k, v in sd.items():
                        delta = v - base_sd[k]
                        sd[k] = base_sd[k] + scale * delta
                    local_models[idx].load_state_dict(sd)

                    if use_reputation and node_perfs.get(idx) is not None:
                        with torch.no_grad():
                            if r2_source == "server_audit" and audit_size_used > 0:
                                test_x, test_y = audit_x, audit_y
                            elif r2_source == "none":
                                test_x = torch.empty(
                                    (0, node.test_data.x.shape[1]), device=device
                                )
                                test_y = torch.empty(
                                    (0,), dtype=torch.long, device=device
                                )
                            else:
                                test_x, test_y = node.test_data.x, node.test_data.y
                            if test_x.numel() == 0:
                                acc = 1.0
                            else:
                                test_logits = local_models[idx](test_x)
                                preds = test_logits.argmax(dim=1)
                                acc = (preds == test_y).float().mean().item()
                            if node.is_malicious and R2_IGNORE_MALICIOUS:
                                acc = 1.0
                            node_perfs[node.drone_id] = float(acc)
                    if need_r4 and r4_ref_x.numel() > 0:
                        node_logits[node.drone_id] = local_models[idx](r4_ref_x)

            dummy_logits = torch.zeros((1, 2), device=device)
            rep_for_val: Dict[int, float] = {}

            if use_reputation:
                server.reputation = {}
                last_rep = {}
                last_comp = {}
                for node in nodes:
                    rep, comps = server.compute_reputation(
                        drone_id=node.drone_id,
                        performance=node_perfs[node.drone_id],
                        data_age=0,
                        local_logits=node_logits.get(node.drone_id, dummy_logits),
                        current_round=r,
                    )
                    server.reputation[node.drone_id] = float(rep)
                    last_rep[node.drone_id] = float(rep)
                    last_comp[node.drone_id] = comps

            elif server_val:
                if node_ref_logits:
                    raw = server.compute_server_validation_weights(
                        node_ref_logits,
                        target_labels=(r4_ref_y if r4_ref_y.numel() > 0 else None),
                    )
                    if raw.dim() == 0:
                        raw = raw.reshape(-1)
                    if raw.numel() != len(nodes):
                        raw = torch.ones(len(nodes), device=device)
                    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    if raw.sum() <= 0:
                        raw = torch.ones(len(nodes), device=device)
                    norm = raw / raw.sum()
                    rep_for_val = {
                        int(nodes[i].drone_id): float(norm[i].item())
                        for i in range(len(nodes))
                    }
                else:
                    rep_for_val = {
                        node.drone_id: 1.0 / float(max(1, len(nodes))) for node in nodes
                    }

                server.reputation = rep_for_val
                last_rep = rep_for_val
                last_comp = {
                    i: (float("nan"), float("nan"), float("nan"))
                    for i in range(len(nodes))
                }

            S_DT = float("nan")
            num_masked = 0
            num_valid = 0
            n_ref_eff = int(r4_ref_x.shape[0]) if r4_ref_x is not None else 0
            S_DT_ratio = float("nan")
            valid_ratio = float("nan")
            fallback_flag = 0
            fallback_method = "none"
            skip_semantic = False
            if (
                need_r4
                and not server_val
                and server.r4_weights is not None
                and server.r4_weights.numel() > 0
            ):
                S_DT = float(server.r4_weights.sum().item())
                n_ref_eff = int(r4_ref_x.shape[0]) if r4_ref_x is not None else 0
                if n_ref_eff <= 0:
                    n_ref_eff = int(server.r4_weights.numel())
                _, num_masked, num_valid = _compute_reference_effective_counts(
                    server, n_ref_eff
                )
                num_valid = int(num_valid)
                S_DT_ratio = _safe_div(S_DT, max(1.0, float(n_ref_eff)))
                valid_ratio = _safe_div(float(num_valid), float(n_ref_eff))
                support_threshold = _safe_div(float(dt_support_min), float(max(1, n_ref_eff)))
                fallback_flag = int(S_DT_ratio < support_threshold)
                skip_semantic = bool(fallback_flag == 1 and S_DT_ratio < support_threshold)
                fallback_method = (
                    str(fallback_mode) if skip_semantic else "none"
                )
            elif server_val:
                fallback_flag = 0
                fallback_method = "none"
                num_valid = int(r4_ref_x.shape[0]) if r4_ref_x is not None else 0
                num_masked = 0
                S_DT_ratio = float("nan")
                valid_ratio = float("nan")

            sdt_series.append(S_DT)
            sdt_ratio_series.append(float(S_DT_ratio))
            valid_ratio_series.append(float(valid_ratio))
            num_masked_series.append(int(num_masked))
            num_valid_series.append(int(num_valid))
            fallback_flags.append(float(fallback_flag))

            benign_ids = [n.drone_id for n in nodes if not n.is_malicious]
            malicious_ids = [n.drone_id for n in nodes if n.is_malicious]

            if use_reputation and need_r4:
                b_pass = (
                    float(
                        np.mean(
                            [
                                not bool(server.last_r4_gate_hit.get(i, False))
                                for i in benign_ids
                            ]
                        )
                    )
                    if benign_ids
                    else float("nan")
                )
                m_pass = (
                    float(
                        np.mean(
                            [
                                not bool(server.last_r4_gate_hit.get(i, False))
                                for i in malicious_ids
                            ]
                        )
                    )
                    if malicious_ids
                    else float("nan")
                )
                total = sum(last_rep.values()) or 1.0
                b_mass = _safe_div(sum(last_rep.get(i, 0.0) for i in benign_ids), total)
            else:
                b_pass = float("nan")
                m_pass = float("nan")
                b_mass = float("nan")

            benign_pass_rates.append(b_pass)
            malicious_pass_rates.append(m_pass)
            benign_adm_mass.append(b_mass)
            if not np.isfinite(S_DT_ratio):
                S_DT_ratio = float("nan")
            if not np.isfinite(valid_ratio):
                valid_ratio = float("nan")

            selected_ids_for_w: List[int] = [n.drone_id for n in nodes]
            if method.startswith("weighted") or method == "server_val":
                if method == "server_val":
                    agg_sd = weighted_average_aggregation(local_models, rep_for_val)
                    w_mal = _safe_div(
                        sum(rep_for_val.get(i, 0.0) for i in malicious_ids),
                        sum(rep_for_val.values()) or 1.0,
                    )
                elif fallback_flag == 1 and need_r4:
                    agg_sd = _geo_aggregation(local_models, fallback_mode)
                    total_rep = sum(last_rep.values()) or 1.0
                    w_mal = _safe_div(
                        sum(last_rep.get(i, 0.0) for i in malicious_ids), total_rep
                    )
                else:
                    agg_sd = weighted_average_aggregation(
                        local_models, server.reputation
                    )
                    total_rep = sum(last_rep.values()) or 1.0
                    w_mal = _safe_div(
                        sum(last_rep.get(i, 0.0) for i in malicious_ids), total_rep
                    )
            elif method == "mean":
                agg_sd = agg.mean_aggregation(local_models)
                w_mal = _safe_div(len(malicious_ids), len(nodes))
            elif method == "median":
                agg_sd = agg.median_aggregation(local_models)
                w_mal = _safe_div(len(malicious_ids), len(nodes))
            elif method == "byzantine_median":
                agg_sd = agg.byzantine_median_aggregation(local_models)
                w_mal = _safe_div(len(malicious_ids), len(nodes))
            elif method == "trimmed_mean":
                agg_sd = agg.trimmed_mean_aggregation(local_models, trim_ratio=0.2)
                w_mal = _safe_div(len(malicious_ids), len(nodes))
            elif method == "krum":
                winners = _select_krum_indices(local_models, C.MALICIOUS_NODES)
                selected_ids_for_w = [nodes[i].drone_id for i in winners]
                agg_sd = agg.krum_aggregation(local_models, f=C.MALICIOUS_NODES)
                w_mal = _safe_div(
                    sum(nodes[i].is_malicious for i in winners), max(1, len(winners))
                )
            elif method == "bulyan":
                winners = _select_bulyan_indices(local_models, C.MALICIOUS_NODES)
                selected_ids_for_w = [nodes[i].drone_id for i in winners]
                agg_sd = agg.bulyan_aggregation(local_models, f=C.MALICIOUS_NODES)
                w_mal = _safe_div(
                    sum(nodes[i].is_malicious for i in winners), max(1, len(winners))
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            w_mal_rounds.append(float(w_mal if np.isfinite(w_mal) else 0.0))
            if round_rows is not None:
                round_rows.append(
                    {
                        "attack": (meta or {}).get("attack", attack_mode),
                        "attack_mode": str((meta or {}).get("attack", attack_mode)),
                        "exp_group": str((meta or {}).get("exp_group", "")),
                        "level": (meta or {}).get("level", ""),
                        "dt_level": str((meta or {}).get("dt_level", dt_level)),
                        "mal_nodes": int(
                            (meta or {}).get("mal_nodes", C.MALICIOUS_NODES)
                        ),
                        "method": (meta or {}).get("method", method),
                        "seed": int((meta or {}).get("seed", seed)),
                        "tau_gate": float(effective_tau),
                        "lambda_m": float(adaptive_mimic_lambda),
                        "ref_size": int(r4_ref_x.shape[0]),
                        "audit_size": int(audit_size),
                        "round": int(r),
                        "w_mal": float(w_mal),
                        "W_mal": float(w_mal),
                        "S_DT": float(S_DT),
                        "S_DT_ratio": float(S_DT_ratio),
                        "fallback_flag": int(fallback_flag),
                        "fallback_method": str(fallback_method),
                        "skip_semantic": int(1 if skip_semantic else 0),
                        "valid_ratio": float(valid_ratio),
                        "num_masked": int(num_masked),
                        "num_valid": int(num_valid),
                        "benign_pass_rate": float(b_pass),
                        "malicious_pass_rate": float(m_pass),
                        "benign_admitted_weight_mass": float(b_mass),
                        "uses_dt": bool(uses_dt),
                        "method_detail": str(method_detail),
                    }
                )

            if diag is not None and method in {
                "byzantine_median",
                "krum",
                "bulyan",
                "trimmed_mean",
            }:
                base_vec = _flatten_model(server.global_model)
                flat_locals = [_flatten_model(m) for m in local_models]
                dmat = _pairwise_dist_matrix(flat_locals)
                l2_to_global = [torch.dist(v, base_vec).item() for v in flat_locals]

                krum_scores = [None] * len(nodes)
                krum_selected = [False] * len(nodes)
                bulyan_selected = [False] * len(nodes)
                trimmed_flag = [False] * len(nodes)

                if method in {"krum", "bulyan"}:
                    n = len(nodes)
                    f = C.MALICIOUS_NODES
                    nb_in_score = max(1, n - f - 2)
                    for i in range(n):
                        s = np.sort(dmat[i][np.arange(n) != i])[:nb_in_score].sum()
                        krum_scores[i] = float(s)
                    if krum_scores:
                        winner = int(
                            np.argmin(
                                [s if s is not None else 1e9 for s in krum_scores]
                            )
                        )
                        krum_selected[winner] = True
                if method == "bulyan":
                    n = len(nodes)
                    f = C.MALICIOUS_NODES
                    m = max(1, n - 2 * f)
                    top = np.argsort(
                        [s if s is not None else 1e9 for s in krum_scores]
                    )[:m]
                    for idx in top:
                        bulyan_selected[int(idx)] = True

                if method == "trimmed_mean":
                    trim = int(len(nodes) * 0.2)
                    order = np.argsort(l2_to_global)
                    low = set(order[:trim])
                    high = set(order[-trim:]) if trim > 0 else set()
                    for i in range(len(nodes)):
                        if i in low or i in high:
                            trimmed_flag[i] = True

                for idx, node in enumerate(nodes):
                    diag.append(
                        {
                            "method": method,
                            "round": r,
                            "node_id": node.drone_id,
                            "is_malicious": int(node.is_malicious),
                            "l2_to_global": l2_to_global[idx],
                            "krum_score": (
                                krum_scores[idx] if krum_scores[idx] is not None else ""
                            ),
                            "krum_selected": int(krum_selected[idx]),
                            "bulyan_selected": int(bulyan_selected[idx]),
                            "trimmed_flag": int(trimmed_flag[idx]),
                        }
                    )

            server.global_model.load_state_dict(agg_sd)

        clean_acc, clean_f1 = eval_model(
            server.global_model, clean_eval_x, clean_eval_y
        )
        deploy_acc, deploy_f1 = eval_model(server.global_model, dep_x, dep_y)
        train_acc, train_f1 = eval_model(
            server.global_model, train_eval_x, train_eval_y
        )
        final_acc, final_f1 = eval_model(server.global_model, ref_x_full, ref_y_full)

        if node_rows is not None:
            for node in nodes:
                nid = int(node.drone_id)
                comps = last_comp.get(nid, (float("nan"), float("nan"), float("nan")))
                n_rep = float(last_rep.get(nid, float("nan")))
                n_kl = float(server.last_r4_kl.get(nid, float("nan")))
                passed_gate = int(not bool(server.last_r4_gate_hit.get(nid, False))) if need_r4 else 1
                pi = float(last_rep.get(nid, float("nan")))
                payload = node.last_attack_payload if hasattr(node, "last_attack_payload") else {}
                node_rows.append(
                    {
                        "attack": (meta or {}).get("attack", attack_mode),
                        "attack_mode": str((meta or {}).get("attack", attack_mode)),
                        "exp_group": str((meta or {}).get("exp_group", "")),
                        "level": (meta or {}).get("level", ""),
                        "dt_level": str((meta or {}).get("dt_level", dt_level)),
                        "mal_nodes": int(
                            (meta or {}).get("mal_nodes", C.MALICIOUS_NODES)
                        ),
                        "method": (meta or {}).get("method", method),
                        "seed": int((meta or {}).get("seed", seed)),
                        "round": int(rounds),
                        "node_id": nid,
                        "is_malicious": int(bool(node.is_malicious)),
                        "rep": n_rep,
                        "Rep": n_rep,
                        "pi": pi,
                        "R2": float(comps[0]),
                        "R3": float(comps[1]),
                        "R4": float(comps[2]),
                        "r4_kl": n_kl,
                        "KL_q_p": n_kl,
                        "r4_gate_hit": int(bool(server.last_r4_gate_hit.get(nid, False))),
                        "passed_gate": int(passed_gate),
                        "tau_gate": float(effective_tau),
                        "lambda_m": float(adaptive_mimic_lambda),
                        "ref_size": int(r4_ref_x.shape[0]),
                        "audit_size": int(audit_size),
                        "audit_size_used": int(audit_size_used),
                        "R2_source": str(r2_source),
                        "method_detail": str(method_detail),
                        "uses_dt": bool(uses_dt),
                        "mimic_attack_enabled": float(payload.get("attack_enabled", 0.0)),
                        "mimic_loss": float(payload.get("mimic_loss", float("nan"))),
                        "poison_loss": float(payload.get("poison_loss", float("nan"))),
                        "final_kl_to_teacher": float(
                            payload.get("final_kl_to_teacher", float("nan"))
                        ),
                    }
                )

        overall = {
            "train_acc": train_acc,
            "train_f1": train_f1,
            "eval_acc": deploy_acc,
            "eval_f1": deploy_f1,
            "final_acc": final_acc,
            "final_f1": final_f1,
            "eval_drop": clean_acc - deploy_acc,
            "w_mal": float(w_mal_rounds[-1]) if w_mal_rounds else float("nan"),
            "S_DT": _nanmean_or_nan(sdt_series),
            "S_DT_ratio": _nanmean_or_nan(sdt_ratio_series),
            "fallback_rate": (
                float(np.mean(fallback_flags)) if fallback_flags else float("nan")
            ),
            "fallback_method": str(fallback_mode) if fallback_flags and any(
                float(x) == 1 for x in fallback_flags
            ) else "none",
            "skip_semantic": int(any(float(x) == 1 for x in fallback_flags))
            if fallback_flags
            else int(0),
            "benign_pass_rate": _nanmean_or_nan(benign_pass_rates),
            "malicious_pass_rate": _nanmean_or_nan(malicious_pass_rates),
            "benign_admitted_weight_mass": _nanmean_or_nan(benign_adm_mass),
            "num_masked": int(np.mean(num_masked_series)) if num_masked_series else 0,
            "num_valid": int(np.mean(num_valid_series)) if num_valid_series else 0,
            "valid_ratio": _nanmean_or_nan(valid_ratio_series),
            "tau_gate": float(effective_tau),
            "lambda_m": float(adaptive_mimic_lambda),
            "ref_size": int(r4_ref_x.shape[0]),
            "audit_size": int(audit_size_used),
            "fallback_flag_final": int(fallback_flags[-1]) if fallback_flags else 0,
            "fallback_flag_first": int(fallback_flags[0]) if fallback_flags else int(0),
            "method_detail": str(method_detail),
            "uses_dt": bool(uses_dt),
            "mimic_loss": _nanmean_or_nan(mimic_losses),
            "poison_loss": _nanmean_or_nan(poison_losses),
            "final_kl_to_teacher": _nanmean_or_nan(final_kls),
            "w_mal_round_mean": (
                float(np.mean(w_mal_rounds)) if w_mal_rounds else float("nan")
            ),
        }

        rep_table = {
            nid: {
                "rep": last_rep[nid],
                "R2": last_comp[nid][0],
                "R3": last_comp[nid][1],
                "R4": last_comp[nid][2],
            }
            for nid in last_rep
        }

        return overall, rep_table

    finally:
        C.R4_GATE_TAU = orig_tau
        if orig_level is not None:
            C._DT_ACTIVE_LEVEL = orig_level
        else:
            setattr(
                C,
                "_DT_ACTIVE_LEVEL",
                getattr(C, "DT_LEVEL", "D0") if hasattr(C, "DT_LEVEL") else "D0",
            )
        C.MAL_ATTACK_MODE = orig_attack
        C.MALICIOUS_NODES = orig_mal_nodes


def _build_mean_std_table(
    df: "pd.DataFrame",
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    count_from_completed: bool = True,
    skip_col: str = "skipped",
) -> "pd.DataFrame":
    import pandas as pd

    if len(df) == 0:
        return pd.DataFrame(
            columns=list(group_cols)
            + [f"{x}_{s}" for x in metric_cols for s in ("m", "s")]
            + ["count"]
        )

    rows = []
    if count_from_completed and skip_col in df.columns:
        completed = df.loc[df[skip_col].astype(bool) == False]
    else:
        completed = df

    for key, g in completed.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["count"] = int(len(g))
        for metric in metric_cols:
            vals = g[metric].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                m = float("nan")
                s = float("nan")
            else:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=0))
            row[f"{metric}_m"] = m
            row[f"{metric}_s"] = s
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(
            columns=list(group_cols)
            + [f"{m}_{s}" for m in metric_cols for s in ("m", "s")]
            + ["count"]
        )
    return out.sort_values(group_cols).reset_index(drop=True)


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _collect_fieldnames(rows: Sequence[Dict], base_fields: Sequence[str]) -> List[str]:
    fields = list(base_fields)
    seen = {k: True for k in base_fields}
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen[key] = True
    return fields


def _write_json(path: str, payload: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _coerce_exp_group_filter(raw: Optional[str]) -> List[str]:
    groups = [x.lower() for x in _normalize_cli_list(raw)]
    if not groups or "auto" in groups:
        return [x for x in EXPERIMENT_GROUPS]
    out = []
    for g in groups:
        if g not in EXPERIMENT_GROUPS:
            raise ValueError(
                f"Unsupported exp_group={g}, must be one of: {', '.join(EXPERIMENT_GROUPS)} or auto"
            )
        if g not in out:
            out.append(g)
    return out


def _coerce_cli_ints(name: str, value: str, default: Sequence[int]) -> List[int]:
    vals = _parse_cli_list_ints(value, default=default)
    try:
        return [int(v) for v in vals if v is not None]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse int list for --{name}: {value}") from exc


def _coerce_cli_floats(name: str, value: str, default: Sequence[float]) -> List[float]:
    vals = _parse_cli_list_floats(value, default=default)
    try:
        return [float(v) for v in vals]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse float list for --{name}: {value}") from exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        type=str,
        default="out",
        help="Root output directory for grouped artifacts",
    )
    ap.add_argument(
        "--exp-group",
        type=str,
        default="auto",
        help="Comma list of experiment groups: auto | base | sdt | tau | server_val | krum | mimic | refaudit",
    )
    ap.add_argument("--methods", type=str, default="weighted,mean,median,trimmed_mean")
    ap.add_argument("--rounds", type=int, default=C.NUM_ROUNDS)
    ap.add_argument(
        "--deploy-variant",
        type=int,
        default=C.EVAL_DEPLOY_VARIANT,
        help="index into NOISE_VARIANTS (0=clean)",
    )
    ap.add_argument(
        "--scenario",
        type=str,
        default=C.EVAL_SCENARIO,
        help="A=train clean/eval clean, B=train clean/eval noise",
    )
    ap.add_argument(
        "--diag-path",
        type=str,
        default=None,
        help="Optional CSV to dump per-round diagnostics (byzantine methods only)",
    )
    ap.add_argument(
        "--attack-modes",
        type=str,
        default="label_flip,stealth_amp,dt_logit_scale",
        help="Comma list of attack modes to sweep",
    )
    ap.add_argument(
        "--mal-nodes",
        type=str,
        default="0,3,5",
        help="Comma list of malicious node counts",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma list of random seeds",
    )
    ap.add_argument(
        "--dt-levels",
        type=str,
        default="D0,D1,D2",
        help="Comma list of DT fidelity levels (D0/D1/D2)",
    )
    ap.add_argument(
        "--tau-gate",
        type=float,
        default=None,
        help="single deprecated alias (overrides tau-grid when provided)",
    )
    ap.add_argument(
        "--tau-grid",
        type=str,
        default=",".join(
            str(x)
            for x in getattr(
                C,
                "TAU_SENSITIVITY_GRID",
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            )
        ),
        help="Comma list of tau values",
    )
    ap.add_argument(
        "--dt-support-min", type=int, default=getattr(C, "DT_SUPPORT_MIN", 10)
    )
    ap.add_argument(
        "--adaptive-mimic-lambdas",
        type=str,
        default=",".join(
            str(x) for x in getattr(C, "ADAPTIVE_MIMIC_LAMBDAS", [0.1, 1, 10])
        ),
        help="Comma list for adaptive-mimic lambda values",
    )
    ap.add_argument(
        "--ref-size-grid",
        type=str,
        default=",".join(
            str(x)
            for x in getattr(
                C,
                "DT_REF_SIZE_CANDIDATES",
                [32, 64, 128, 256, 512],
            )
        ),
        help="Comma list of R4 reference subset sizes",
    )
    ap.add_argument(
        "--audit-size-grid",
        type=str,
        default=",".join(
            str(x)
            for x in getattr(
                C,
                "R2_AUDIT_SIZE_CANDIDATES",
                [0, 32, 64, 128, 256],
            )
        ),
        help="Comma list of server audit set sizes",
    )
    ap.add_argument("--out-runs", type=str, default=None)
    ap.add_argument("--out-summary", type=str, default=None)
    ap.add_argument("--out-rounds", type=str, default=None)
    ap.add_argument("--out-nodes", type=str, default=None)
    ap.add_argument("--out-fallback-summary", type=str, default="fallback_summary.csv")
    ap.add_argument("--out-passrate-summary", type=str, default="passrate_summary.csv")
    ap.add_argument(
        "--out-sensitivity-ref", type=str, default="sensitivity_summary_ref.csv"
    )
    ap.add_argument(
        "--out-sensitivity-audit", type=str, default="sensitivity_summary_audit.csv"
    )
    args = ap.parse_args()

    exp_groups = _coerce_exp_group_filter(args.exp_group)
    methods = _normalize_cli_list(args.methods)
    attack_modes = _normalize_cli_list(args.attack_modes)
    mal_nodes_list = _coerce_cli_ints("mal-nodes", args.mal_nodes, [0, 1, 2, 3, 5])
    seeds = _coerce_cli_ints("seeds", args.seeds, [0])
    dt_levels = _normalize_cli_list(args.dt_levels) or ["D0"]

    if not methods:
        raise ValueError("No method specified")
    if not attack_modes:
        raise ValueError("No attack mode specified")
    if not mal_nodes_list:
        raise ValueError("No malicious node count specified")
    if not seeds:
        raise ValueError("No seed specified")

    if args.tau_gate is not None:
        tau_grid = [float(args.tau_gate)]
    else:
        tau_grid = _coerce_cli_floats(
            "tau-grid",
            args.tau_grid,
            default=getattr(C, "TAU_SENSITIVITY_GRID", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        )

    adaptive_mimic_lams = _coerce_cli_floats(
        "adaptive-mimic-lambdas",
        args.adaptive_mimic_lambdas,
        default=getattr(C, "ADAPTIVE_MIMIC_LAMBDAS", [0.1, 1, 10]),
    )
    ref_size_grid = _coerce_cli_ints(
        "ref-size-grid",
        args.ref_size_grid,
        default=getattr(C, "DT_REF_SIZE_CANDIDATES", [32, 64, 128, 256, 512]),
    )
    audit_size_grid = _coerce_cli_ints(
        "audit-size-grid",
        args.audit_size_grid,
        default=getattr(C, "R2_AUDIT_SIZE_CANDIDATES", [0, 32, 64, 128, 256]),
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.out_runs is not None or args.out_summary is not None or args.out_rounds is not None or args.out_nodes is not None:
        print(
            "[warn] out-runs/out-summary/out-rounds/out-nodes are ignored in grouped mode; use --out-root"
        )

    import pandas as pd

    print("method, polluted_acc, polluted_f1, clean_acc, clean_f1, W_mal_total")
    print("-------------------------------------------------------------------")

    diag_rows: List[Dict] = []
    runs_rows: List[Dict] = []
    rounds_rows: List[Dict] = []
    nodes_rows: List[Dict] = []

    for attack in attack_modes:
        level_list = ["L1"] if attack == "label_flip" else [""]

        if attack == "label_flip":
            C.LABEL_FLIP_RATIO = 0.5
            C.LABEL_FLIP_LR = 0.06
            C.MAL_GRAD_SCALE_MAP["label_flip"] = 0.5
        else:
            C.LABEL_FLIP_RATIO = float(getattr(C, "LABEL_FLIP_RATIO", 0.5))
            C.LABEL_FLIP_LR = float(getattr(C, "LABEL_FLIP_LR", 0.06))

        for dt_level in dt_levels:
            if dt_level not in getattr(C, "DT_MISMATCH_LEVELS", {}):
                print(f"[warn] unknown dt level={dt_level}, skip")
                continue

            for mal_n in mal_nodes_list:
                if mal_n < 0 or mal_n > C.NUM_NODES:
                    print(f"[warn] invalid mal_nodes={mal_n}, skip")
                    continue

                C.MALICIOUS_NODES = mal_n

                for m in methods:
                    exp_group = _infer_group(
                        method=m,
                        attack=attack,
                        dt_level=dt_level,
                        tau_grid=tau_grid,
                        ref_grid=ref_size_grid,
                        audit_grid=audit_size_grid,
                    )
                    if exp_group not in exp_groups:
                        continue

                    if m == "server_val":
                        lambda_list = [0.0]
                    elif attack == "adaptive_mimic":
                        lambda_list = list(adaptive_mimic_lams)
                    else:
                        lambda_list = [0.0]

                    for tau_gate in tau_grid:
                        for lam_m in lambda_list:
                            for ref_size in ref_size_grid:
                                for audit_size in audit_size_grid:
                                    metrics: List[Dict] = []
                                    for level in level_list:
                                        # keep per-level context in meta
                                        for seed in seeds:
                                            overall, _ = run_once(
                                                method=m,
                                                rounds=args.rounds,
                                                deploy_variant=args.deploy_variant,
                                                scenario=args.scenario,
                                                seed=seed,
                                                tau_gate=float(tau_gate),
                                                dt_support_min=args.dt_support_min,
                                                fallback_mode=getattr(
                                                    C, "GEO_FALLBACK_AGG", "mean"
                                                ),
                                                adaptive_mimic_lambda=float(lam_m),
                                                ref_size=int(ref_size),
                                                audit_size=int(audit_size),
                                                diag=(
                                                    diag_rows
                                                    if m
                                                    in {
                                                        "byzantine_median",
                                                        "krum",
                                                        "bulyan",
                                                        "trimmed_mean",
                                                    }
                                                    else None
                                                ),
                                                round_rows=rounds_rows,
                                                node_rows=nodes_rows,
                                                meta={
                                                    "attack": attack,
                                                    "level": level,
                                                    "dt_level": dt_level,
                                                    "mal_nodes": mal_n,
                                                    "method": m,
                                                    "seed": seed,
                                                    "exp_group": exp_group,
                                                },
                                            )
                                            metrics.append(overall)

                                            runs_rows.append(
                                                {
                                                    "attack_mode": str(attack),
                                                    "attack": attack,
                                                    "level": level,
                                                    "dt_level": dt_level,
                                                    "mal_nodes": int(mal_n),
                                                    "method": m,
                                                    "seed": int(seed),
                                                    "tau_gate": float(tau_gate),
                                                    "lambda_m": float(lam_m),
                                                    "ref_size": int(ref_size),
                                                    "audit_size": int(audit_size),
                                                    "skipped": bool(
                                                        overall.get("skipped", False)
                                                    ),
                                                    "skip_reason": str(
                                                        overall.get("skip_reason", "")
                                                    ),
                                                    "polluted_acc": float(
                                                        overall["eval_acc"]
                                                    ),
                                                    "polluted_f1": float(
                                                        overall["eval_f1"]
                                                    ),
                                                    "clean_acc": float(
                                                        overall["final_acc"]
                                                    ),
                                                    "clean_f1": float(
                                                        overall["final_f1"]
                                                    ),
                                                    "w_mal": float(
                                                        overall.get(
                                                            "w_mal", float("nan")
                                                        )
                                                    ),
                                                    "w_mal_round_mean": float(
                                                        overall.get(
                                                            "w_mal_round_mean",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "S_DT": float(
                                                        overall.get(
                                                            "S_DT", float("nan")
                                                        )
                                                    ),
                                                    "fallback_rate": float(
                                                        overall.get(
                                                            "fallback_rate",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "fallback_flag_first": float(
                                                        overall.get(
                                                            "fallback_flag_first",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "fallback_flag_final": float(
                                                        overall.get(
                                                            "fallback_flag_final",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "fallback_method": str(
                                                        overall.get(
                                                            "fallback_method", "none"
                                                        )
                                                    ),
                                                    "skip_semantic": int(
                                                        1
                                                        if bool(
                                                            overall.get(
                                                                "skip_semantic", 0
                                                            )
                                                        )
                                                        else 0
                                                    ),
                                                    "S_DT_ratio": float(
                                                        overall.get(
                                                            "S_DT_ratio", float("nan")
                                                        )
                                                    ),
                                                    "valid_ratio": float(
                                                        overall.get(
                                                            "valid_ratio", float("nan")
                                                        )
                                                    ),
                                                    "num_masked": float(
                                                        overall.get(
                                                            "num_masked", float("nan")
                                                        )
                                                    ),
                                                    "num_valid": float(
                                                        overall.get(
                                                            "num_valid", float("nan")
                                                        )
                                                    ),
                                                    "benign_pass_rate": float(
                                                        overall.get(
                                                            "benign_pass_rate",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "malicious_pass_rate": float(
                                                        overall.get(
                                                            "malicious_pass_rate",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "benign_admitted_weight_mass": float(
                                                        overall.get(
                                                            "benign_admitted_weight_mass",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "uses_dt": bool(overall.get("uses_dt", False)),
                                                    "method_detail": str(
                                                        overall.get("method_detail", "")
                                                    ),
                                                    "mimic_loss": float(
                                                        overall.get(
                                                            "mimic_loss", float("nan")
                                                        )
                                                    ),
                                                    "poison_loss": float(
                                                        overall.get(
                                                            "poison_loss", float("nan")
                                                        )
                                                    ),
                                                    "final_kl_to_teacher": float(
                                                        overall.get(
                                                            "final_kl_to_teacher",
                                                            float("nan"),
                                                        )
                                                    ),
                                                    "exp_group": str(exp_group),
                                                }
                                            )

                                            if overall.get("skipped", False):
                                                print(
                                                    f"[skip] {attack} dt={dt_level} f={mal_n} method={m} "
                                                    f"tau={tau_gate:g} lam={lam_m:g} ref={ref_size} audit={audit_size}: "
                                                    f"{overall.get('skip_reason', 'skip')}"
                                                )

                                    if not metrics:
                                        continue
                                    good = [
                                        x
                                        for x in metrics
                                        if not bool(x.get("skipped", False))
                                    ]
                                    if not good:
                                        continue

                                    # ---- lightweight per-sweep summary ----
                                    def _mean_std_vec(xs, key):
                                        arr = np.array(
                                            [
                                                float(x[key])
                                                for x in good
                                                if x.get(key) is not None
                                                and np.isfinite(x[key])
                                            ],
                                            dtype=float,
                                        )
                                        if arr.size == 0:
                                            return float("nan"), float("nan")
                                        return float(np.mean(arr)), float(np.std(arr))

                                    pa_m, pa_s = _mean_std_vec(
                                        good, "eval_f1"
                                    )  # keep old display style
                                    pf_m, pf_s = _mean_std_vec(good, "polluted_f1")
                                    ca_m, ca_s = _mean_std_vec(good, "clean_f1")
                                    cf_m, cf_s = _mean_std_vec(good, "clean_f1")
                                    w_m, w_s = _mean_std_vec(good, "w_mal")

                                    print(
                                        f"{m:<15}{pa_m:>12.4f}{pf_m:>12.4f}"
                                        f"{ca_m:>12.4f}{cf_m:>12.4f}"
                                        f"{w_m:>10.4f}"
                                    )

    # ---- grouped write per-experiment outputs ----
    runs_base_cols = [
        "attack_mode",
        "attack",
        "level",
        "dt_level",
        "mal_nodes",
        "method",
        "seed",
        "tau_gate",
        "lambda_m",
        "ref_size",
        "audit_size",
        "skipped",
        "skip_reason",
        "polluted_acc",
        "polluted_f1",
        "clean_acc",
        "clean_f1",
        "w_mal",
        "w_mal_round_mean",
        "S_DT",
        "S_DT_ratio",
        "valid_ratio",
        "fallback_rate",
        "fallback_flag_first",
        "fallback_flag_final",
        "fallback_method",
        "skip_semantic",
        "num_masked",
        "num_valid",
        "benign_pass_rate",
        "malicious_pass_rate",
        "benign_admitted_weight_mass",
        "uses_dt",
        "method_detail",
        "mimic_loss",
        "poison_loss",
        "final_kl_to_teacher",
        "exp_group",
    ]
    rounds_base_cols = [
        "attack_mode",
        "attack",
        "level",
        "dt_level",
        "mal_nodes",
        "method",
        "seed",
        "round",
        "tau_gate",
        "lambda_m",
        "ref_size",
        "audit_size",
        "S_DT",
        "S_DT_ratio",
        "num_masked",
        "num_valid",
        "valid_ratio",
        "fallback_flag",
        "fallback_method",
        "skip_semantic",
        "W_mal",
        "benign_pass_rate",
        "malicious_pass_rate",
        "benign_admitted_weight_mass",
        "uses_dt",
        "method_detail",
    ]
    nodes_base_cols = [
        "attack_mode",
        "attack",
        "level",
        "dt_level",
        "mal_nodes",
        "method",
        "seed",
        "round",
        "node_id",
        "is_malicious",
        "R4",
        "KL_q_p",
        "Rep",
        "passed_gate",
        "pi",
        "R2",
        "R2_source",
        "method_detail",
        "uses_dt",
        "mimic_attack_enabled",
        "mimic_loss",
        "poison_loss",
        "final_kl_to_teacher",
        "exp_group",
    ]
    summary_group_cols = [
        "attack_mode",
        "attack",
        "level",
        "dt_level",
        "mal_nodes",
        "method",
        "tau_gate",
        "lambda_m",
        "ref_size",
        "audit_size",
    ]
    summary_metric_cols = [
        "polluted_acc",
        "polluted_f1",
        "clean_acc",
        "clean_f1",
        "w_mal",
        "w_mal_round_mean",
        "S_DT",
        "S_DT_ratio",
        "valid_ratio",
        "num_masked",
        "num_valid",
        "fallback_rate",
        "benign_pass_rate",
        "malicious_pass_rate",
        "benign_admitted_weight_mass",
    ]

    for exp_group in exp_groups:
        g_runs = [r for r in runs_rows if str(r.get("exp_group", "")) == exp_group]
        if len(g_runs) == 0 and exp_group not in {r.get("exp_group") for r in runs_rows}:
            # keep artifact skeleton for explicitly selected empty groups
            g_runs = []
        g_rounds = [r for r in rounds_rows if str(r.get("exp_group", "")) == exp_group]
        g_nodes = [r for r in nodes_rows if str(r.get("exp_group", "")) == exp_group]

        group_dir = out_root / exp_group
        group_dir.mkdir(parents=True, exist_ok=True)

        _write_json(
            str(group_dir / "config.json"),
            {
                "group": exp_group,
                "command": " ".join(sys.argv),
                "selected_attack_modes": attack_modes,
                "selected_methods": methods,
                "selected_dt_levels": dt_levels,
                "selected_mal_nodes": mal_nodes_list,
                "selected_seeds": seeds,
                "selected_tau_grid": tau_grid,
                "selected_ref_size_grid": ref_size_grid,
                "selected_audit_size_grid": audit_size_grid,
                "selected_adaptive_mimic_lambdas": adaptive_mimic_lams,
                "dt_support_min": int(args.dt_support_min),
                "rounds": int(args.rounds),
                "deploy_variant": int(args.deploy_variant),
                "scenario": str(args.scenario),
                "fallback_mode": str(getattr(C, "GEO_FALLBACK_AGG", "mean")),
                "group_runs": len(g_runs),
                "schema": {
                    "required_runs": runs_base_cols,
                    "required_rounds": rounds_base_cols,
                    "required_nodes": nodes_base_cols,
                },
            },
        )

        _write_csv(str(group_dir / "runs.csv"), g_runs, _collect_fieldnames(g_runs, runs_base_cols))
        _write_csv(
            str(group_dir / "rounds.csv"),
            g_rounds,
            _collect_fieldnames(g_rounds, rounds_base_cols),
        )
        _write_csv(
            str(group_dir / "nodes.csv"),
            g_nodes,
            _collect_fieldnames(g_nodes, nodes_base_cols),
        )

        runs_df = pd.DataFrame(g_runs)
        if "skipped" in runs_df.columns:
            runs_df["skipped"] = runs_df["skipped"].fillna(False).astype(bool)
        else:
            runs_df["skipped"] = False

        summary_df = _build_mean_std_table(runs_df, summary_group_cols, summary_metric_cols)
        _write_csv(
            str(group_dir / "summary.csv"),
            summary_df.to_dict(orient="records"),
            _collect_fieldnames(
                summary_df.to_dict(orient="records"),
                [
                    "attack_mode",
                    "attack",
                    "level",
                    "dt_level",
                    "mal_nodes",
                    "method",
                    "tau_gate",
                    "lambda_m",
                    "ref_size",
                    "audit_size",
                ]
                + [f"{metric}_m" for metric in summary_metric_cols]
                + [f"{metric}_s" for metric in summary_metric_cols]
                + ["count"],
            ),
        )

        fallback_summary_df = _build_mean_std_table(
            runs_df,
            [
                "attack_mode",
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "tau_gate",
                "lambda_m",
                "ref_size",
                "audit_size",
            ],
            [
                "fallback_rate",
                "benign_pass_rate",
                "malicious_pass_rate",
                "benign_admitted_weight_mass",
            ],
        )
        _write_csv(
            str(group_dir / args.out_fallback_summary),
            fallback_summary_df.to_dict(orient="records"),
            _collect_fieldnames(
                fallback_summary_df.to_dict(orient="records"),
                [
                    "attack_mode",
                    "attack",
                    "level",
                    "dt_level",
                    "mal_nodes",
                    "method",
                    "tau_gate",
                    "lambda_m",
                    "ref_size",
                    "audit_size",
                    "fallback_rate_m",
                    "fallback_rate_s",
                    "benign_pass_rate_m",
                    "benign_pass_rate_s",
                    "malicious_pass_rate_m",
                    "malicious_pass_rate_s",
                    "benign_admitted_weight_mass_m",
                    "benign_admitted_weight_mass_s",
                    "count",
                ],
            ),
        )

        passrate_summary_df = _build_mean_std_table(
            runs_df,
            [
                "attack_mode",
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "tau_gate",
                "lambda_m",
                "ref_size",
                "audit_size",
            ],
            ["benign_pass_rate", "malicious_pass_rate"],
        )
        _write_csv(
            str(group_dir / args.out_passrate_summary),
            passrate_summary_df.to_dict(orient="records"),
            _collect_fieldnames(
                passrate_summary_df.to_dict(orient="records"),
                [
                    "attack_mode",
                    "attack",
                    "level",
                    "dt_level",
                    "mal_nodes",
                    "method",
                    "tau_gate",
                    "lambda_m",
                    "ref_size",
                    "audit_size",
                    "benign_pass_rate_m",
                    "benign_pass_rate_s",
                    "malicious_pass_rate_m",
                    "malicious_pass_rate_s",
                    "count",
                ],
            ),
        )

        # sensitivity in ref size / audit size
        sens_ref_df = _build_mean_std_table(
            runs_df,
            [
                "attack_mode",
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "tau_gate",
                "lambda_m",
                "audit_size",
                "ref_size",
            ],
            ["clean_f1", "w_mal", "clean_acc", "polluted_f1", "polluted_acc"],
        )
        _write_csv(
            str(group_dir / args.out_sensitivity_ref),
            sens_ref_df.to_dict(orient="records"),
            _collect_fieldnames(
                sens_ref_df.to_dict(orient="records"),
                [
                    "attack_mode",
                    "attack",
                    "level",
                    "dt_level",
                    "mal_nodes",
                    "method",
                    "tau_gate",
                    "lambda_m",
                    "audit_size",
                    "ref_size",
                    "clean_f1_m",
                    "clean_f1_s",
                    "w_mal_m",
                    "w_mal_s",
                    "clean_acc_m",
                    "clean_acc_s",
                    "polluted_f1_m",
                    "polluted_f1_s",
                    "polluted_acc_m",
                    "polluted_acc_s",
                    "count",
                ],
            ),
        )

        sens_audit_df = _build_mean_std_table(
            runs_df,
            [
                "attack_mode",
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "tau_gate",
                "lambda_m",
                "ref_size",
                "audit_size",
            ],
            ["clean_f1", "w_mal", "clean_acc", "polluted_f1", "polluted_acc"],
        )
        _write_csv(
            str(group_dir / args.out_sensitivity_audit),
            sens_audit_df.to_dict(orient="records"),
            _collect_fieldnames(
                sens_audit_df.to_dict(orient="records"),
                [
                    "attack_mode",
                    "attack",
                    "level",
                    "dt_level",
                    "mal_nodes",
                    "method",
                    "tau_gate",
                    "lambda_m",
                    "ref_size",
                    "audit_size",
                    "clean_f1_m",
                    "clean_f1_s",
                    "w_mal_m",
                    "w_mal_s",
                    "clean_acc_m",
                    "clean_acc_s",
                    "polluted_f1_m",
                    "polluted_f1_s",
                    "polluted_acc_m",
                    "polluted_acc_s",
                    "count",
                ],
            ),
        )

    if len(rounds_rows) > 0 or len(nodes_rows) > 0:
        # compatibility with legacy plotting scripts
        legacy_summary = pd.DataFrame(_build_mean_std_table(
            pd.DataFrame(runs_rows),
            summary_group_cols,
            ["polluted_acc", "polluted_f1", "clean_f1", "clean_acc", "w_mal", "w_mal_round_mean"],
        ).to_dict(orient="records"))
        if len(legacy_summary) > 0:
            legacy_summary["polluted_acc"] = legacy_summary.get("polluted_acc_m", np.nan)
            legacy_summary["polluted_f1"] = legacy_summary.get("polluted_f1_m", np.nan)
            legacy_summary["clean_acc"] = legacy_summary.get("clean_acc_m", np.nan)
            legacy_summary["clean_f1"] = legacy_summary.get("clean_f1_m", np.nan)
            legacy_summary["w_mal"] = legacy_summary.get("w_mal_m", np.nan)
            legacy_cols = [
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "polluted_acc_m",
                "polluted_acc_s",
                "polluted_f1_m",
                "polluted_f1_s",
                "clean_acc_m",
                "clean_acc_s",
                "clean_f1_m",
                "clean_f1_s",
                "w_mal_m",
                "w_mal_s",
                "count",
            ]
            legacy_cols = [c for c in legacy_cols if c in legacy_summary.columns]
            _write_csv(
                str(out_root / "summary_legacy.csv"),
                legacy_summary.to_dict(orient="records"),
                legacy_cols,
            )

    # ---- diagnostics
    if args.diag_path and diag_rows:
        with open(args.diag_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "method",
                "round",
                "node_id",
                "is_malicious",
                "l2_to_global",
                "krum_score",
                "krum_selected",
                "bulyan_selected",
                "trimmed_flag",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in diag_rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
