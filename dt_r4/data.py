"""
dt_r4/data.py

æ°æ®å è½½ãæ¸æ´ãåªå£°æé ãæ seed ååèç¹æ°æ®ç­ã?"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .config import (
    NOISE_DATA_DIR,
    NOISE_VARIANTS,
    TARGET_NOISESET_NAMES,
    TEACHER_CKPT,
)
from .runtime import device, set_seeds
from .models import TeacherNet


# =========================
# 3) ç®åæ°æ®å®¹å?# =========================
class SimpleData:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y


# =========================
# 5) æ°æ®å è½½ + åªå£°çæ
# =========================
def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    return df


def flip_labels_for_noise(labels: pd.Series, flip_ratio: float) -> pd.Series:
    labels = labels.copy()
    if flip_ratio <= 0 or len(labels) == 0:
        return labels
    num_flip = int(len(labels) * flip_ratio)
    if num_flip <= 0:
        return labels

    idx = np.random.choice(labels.index, num_flip, replace=False)
    unique_labels = labels.unique().tolist()

    for i in idx:
        current = labels.at[i]
        candidates = [c for c in unique_labels if c != current]
        if candidates:
            labels.at[i] = random.choice(candidates)

    return labels


def apply_noise_to_df(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    noisy = df.copy()
    feat_cols = noisy.columns[:-1]

    feature_noise_frac = spec.get("feature_noise_frac", 0.0)
    feature_dropout_ratio = spec.get("feature_dropout_ratio", 0.0)
    label_flip_ratio = spec.get("label_flip_ratio", 0.0)
    bias_std_scale = spec.get("bias_std_scale", 0.0)
    biased_feature_ratio = spec.get("biased_feature_ratio", 0.0)
    bias_mode = str(spec.get("bias_mode", "")).strip().lower()
    bias_sign_override = str(spec.get("bias_sign_override", "")).strip().lower()

    if bias_std_scale > 0 and biased_feature_ratio > 0 and bias_mode == "fixed_signed":
        std = noisy[feat_cols].std().replace(0, 1e-6)
        feat_names = list(feat_cols)
        num_biased = max(1, int(np.ceil(len(feat_names) * float(biased_feature_ratio))))
        chosen = np.random.choice(
            feat_names, size=min(num_biased, len(feat_names)), replace=False
        )
        if bias_sign_override == "positive":
            signs = np.ones(len(chosen), dtype=float)
        elif bias_sign_override == "negative":
            signs = -np.ones(len(chosen), dtype=float)
        else:
            signs = np.random.choice([-1.0, 1.0], size=len(chosen))
        offsets = signs * float(bias_std_scale) * std.loc[list(chosen)].values
        noisy.loc[:, list(chosen)] = noisy.loc[:, list(chosen)] + offsets

    if feature_noise_frac > 0:
        std = noisy[feat_cols].std().replace(0, 1e-6)
        noise = np.random.normal(
            0.0, feature_noise_frac * std.values, size=noisy[feat_cols].shape
        )
        noisy[feat_cols] = noisy[feat_cols] + noise

    if feature_dropout_ratio > 0:
        mask = np.random.rand(*noisy[feat_cols].shape) < feature_dropout_ratio
        noisy[feat_cols] = noisy[feat_cols].mask(mask, 0.0)

    if label_flip_ratio > 0:
        noisy.iloc[:, -1] = flip_labels_for_noise(noisy.iloc[:, -1], label_flip_ratio)

    noisy.iloc[:, -1] = noisy.iloc[:, -1].astype(int)
    return noisy


def _resolve_noise_spec_name(
    token: str | None,
    alias_map: Optional[Dict[str, str]] = None,
) -> str:
    raw = str(token or "").strip()
    if raw == "":
        return "clean"
    lowered = raw.lower()
    if lowered in {"clean", "none"}:
        return "clean"
    if alias_map is not None and lowered in alias_map:
        return str(alias_map[lowered]).strip() or "clean"
    return raw


def _lookup_noise_spec(spec_name: str) -> dict:
    name = str(spec_name).strip()
    if name.lower() == "clean":
        return {
            "name": "clean",
            "feature_noise_frac": 0.0,
            "label_flip_ratio": 0.0,
            "feature_dropout_ratio": 0.0,
        }
    for spec in NOISE_VARIANTS:
        if str(spec.get("name", "")).strip() == name:
            return spec
    raise ValueError(
        f"Unknown benign noise spec '{spec_name}'. "
        f"Expected one of: {', '.join(str(s.get('name', '')) for s in NOISE_VARIANTS)}"
    )


def _benign_noise_role(spec_name: str, token: str | None = None) -> str:
    name = str(spec_name or "").strip().lower()
    raw = str(token or "").strip().lower()
    if name in {"", "clean", "none"} or raw in {"", "clean", "none"}:
        return "benign_clean"
    if raw == "light" or name in {"feat_noise_20", "drift_light"}:
        return "benign_light"
    if raw in {"heavy", "heavyp", "heavyn"} or name in {
        "feat_noise_50_dropout_50",
        "drift_heavy",
        "drift_heavy_v2",
        "drift_heavy_pos_v2",
        "drift_heavy_neg_v2",
    }:
        return "benign_heavy"
    if raw in {"medium", "mediump", "mediumn"} or name in {
        "feat_noise_25_dropout_20",
        "feat_noise_35_dropout_30",
        "drift_medium",
        "drift_medium_v2",
        "drift_medium_pos_v2",
        "drift_medium_neg_v2",
    }:
        return "benign_medium"
    return "benign_clean"


def build_noise_variants_fixed(base_csv_path: str, dataset_seed: int = 123):
    """
    åªçæ?è¿å NOISE_VARIANTS ä¸­å£°æççæ¬ï¼å½åå³4ä¸ªï¼ã?    dataset_seed ç¨äºâåºå®åªå£°æ°æ®éâï¼é¿åä¸åseedä¸åªå£°CSVä¸åå¯¼è´æ¯è¾ä¸å¹²åã?    """
    set_seeds(dataset_seed)

    base_df = load_and_clean_csv(base_csv_path)
    os.makedirs(NOISE_DATA_DIR, exist_ok=True)
    variants = []

    for spec in NOISE_VARIANTS:
        is_clean = (
            spec.get("feature_noise_frac", 0.0) == 0.0
            and spec.get("label_flip_ratio", 0.0) == 0.0
            and spec.get("feature_dropout_ratio", 0.0) == 0.0
        )
        if is_clean:
            path = base_csv_path
            desc = "clean"
        else:
            stem = Path(base_csv_path).stem
            path = os.path.join(NOISE_DATA_DIR, f"{stem}_{spec['name']}.csv")

            if not os.path.exists(path):
                noisy_df = apply_noise_to_df(base_df, spec)
                noisy_df.to_csv(path, index=False)

            desc = (
                f"feat_noise={spec.get('feature_noise_frac', 0.0)}, "
                f"label_flip={spec.get('label_flip_ratio', 0.0)}, "
                f"feat_dropout={spec.get('feature_dropout_ratio', 0.0)}"
            )

        variants.append({"name": spec["name"], "path": path, "desc": desc})

    variants = [v for v in variants if v["name"] in TARGET_NOISESET_NAMES]
    variants.sort(key=lambda x: TARGET_NOISESET_NAMES.index(x["name"]))
    return variants


def load_reference_data(csv_path: str):
    ref_df = load_and_clean_csv(csv_path)
    ref_x = torch.tensor(ref_df.iloc[:, :-1].values, dtype=torch.float32, device=device)
    ref_y = torch.tensor(ref_df.iloc[:, -1].values, dtype=torch.long, device=device)
    return ref_x, ref_y


def sample_reference_subset(
    ref_x: torch.Tensor, ref_y: torch.Tensor, n: int | None, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a reference subset for DT sensitivity.
    If n is None/non-positive or larger than available, returns full tensors.
    """
    if n is None or int(n) <= 0:
        return ref_x, ref_y

    n_int = int(n)
    n_total = int(ref_x.shape[0])
    if n_int >= n_total:
        return ref_x, ref_y
    if n_int <= 0:
        return ref_x[:0], ref_y[:0]

    g = torch.Generator(device=ref_x.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g, device=ref_x.device)
    idx = perm[:n_int]
    return ref_x[idx], ref_y[idx]


def sample_audit_set(
    ref_x: torch.Tensor, ref_y: torch.Tensor, n: int | None, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a labeled audit set used by R2. n=0 returns empty tensors (neutral R2).
    """
    if n is None or int(n) <= 0:
        return ref_x[:0], ref_y[:0]
    return sample_reference_subset(ref_x, ref_y, n, seed)


def get_teacher_model():
    model = TeacherNet(num_output_features=2).to(device)
    if not os.path.exists(TEACHER_CKPT):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {TEACHER_CKPT}. Please train it first."
        )
    model.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
    model.eval()
    return model


def load_node_splits(
    num_nodes: int,
    csv_path: str,
    seed: int,
    malicious_nodes: int = 0,
    attack_mode: str | None = None,
    label_flip_ratio: float = 0.0,
    pre_split_poison: bool = False,
    benign_noise_profile: Optional[Sequence[str]] = None,
    benign_noise_alias_map: Optional[Dict[str, str]] = None,
    benign_deploy_noise_profile: Optional[Sequence[str]] = None,
    benign_deploy_noise_alias_map: Optional[Dict[str, str]] = None,
):
    """
    æ¯ä¸ª seed ä¼éæ°ååèç¹æ°æ®æ± ï¼ç¨äºéå¤å®éªï¼ã?
    å¦æ pre_split_poison ä¸?True ä¸èç¹å±äºæ¶æèç¹ï¼i < malicious_nodesï¼å¹¶ä¸?    attack_mode == "label_flip"ï¼ååå¯¹è¯¥èç¹çå®æ´å­éåæ ç­¾ç¿»è½¬ï¼åæå?    train / testï¼è¿æ ·è®­ç»åæµè¯éé½ä¼è¢«åæ ·çæ»å»æ±¡æã?    """
    df = load_and_clean_csv(csv_path)

    indices = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_size = len(df) // num_nodes

    node_data_objects = []
    benign_profile = [str(x).strip() for x in (benign_noise_profile or [])]
    benign_deploy_profile = [
        str(x).strip() for x in (benign_deploy_noise_profile or [])
    ]
    benign_idx = 0
    for i in range(num_nodes):
        start = i * split_size
        end = None if i == num_nodes - 1 else (i + 1) * split_size
        node_idx = indices[start:end]
        node_df = df.iloc[node_idx]
        is_malicious = i < malicious_nodes

        if is_malicious:
            noise_token = "malicious"
            noise_spec_name = str(attack_mode or "malicious")
            noise_role = "malicious"
        else:
            noise_token = (
                benign_profile[benign_idx] if benign_idx < len(benign_profile) else "clean"
            )
            noise_spec_name = _resolve_noise_spec_name(
                noise_token,
                alias_map=benign_noise_alias_map,
            )
            noise_role = _benign_noise_role(noise_spec_name, noise_token)
            deploy_noise_token = (
                benign_deploy_profile[benign_idx]
                if benign_idx < len(benign_deploy_profile)
                else "clean"
            )
            deploy_noise_spec_name = _resolve_noise_spec_name(
                deploy_noise_token,
                alias_map=benign_deploy_noise_alias_map,
            )
            deploy_noise_role = _benign_noise_role(
                deploy_noise_spec_name, deploy_noise_token
            )
            benign_idx += 1
        if is_malicious:
            deploy_noise_token = "malicious"
            deploy_noise_spec_name = str(attack_mode or "malicious")
            deploy_noise_role = "malicious"

        if (
            pre_split_poison
            and is_malicious
            and attack_mode == "label_flip"
            and label_flip_ratio > 0
        ):
            node_df = node_df.copy()
            node_df.iloc[:, -1] = flip_labels_for_noise(
                node_df.iloc[:, -1], flip_ratio=label_flip_ratio
            )

        train_node_df, test_node_df = train_test_split(
            node_df, test_size=0.2, random_state=seed
        )

        if (not is_malicious) and noise_spec_name.lower() not in {"", "clean", "none"}:
            train_node_df = apply_noise_to_df(
                train_node_df,
                _lookup_noise_spec(noise_spec_name),
            )
        if (
            (not is_malicious)
            and deploy_noise_spec_name.lower() not in {"", "clean", "none"}
        ):
            test_node_df = apply_noise_to_df(
                test_node_df,
                _lookup_noise_spec(deploy_noise_spec_name),
            )

        train_x = torch.tensor(
            train_node_df.iloc[:, :-1].values, dtype=torch.float32, device=device
        )
        train_y = torch.tensor(
            train_node_df.iloc[:, -1].values, dtype=torch.long, device=device
        )
        test_x = torch.tensor(
            test_node_df.iloc[:, :-1].values, dtype=torch.float32, device=device
        )
        test_y = torch.tensor(
            test_node_df.iloc[:, -1].values, dtype=torch.long, device=device
        )

        node_data_objects.append(
            {
                "train": SimpleData(train_x, train_y),
                "test": SimpleData(test_x, test_y),
                "noise_role": noise_role,
                "noise_spec": noise_spec_name,
                "noise_token": noise_token,
                "deploy_noise_role": deploy_noise_role,
                "deploy_noise_spec": deploy_noise_spec_name,
                "deploy_noise_token": deploy_noise_token,
            }
        )

    return node_data_objects, df

