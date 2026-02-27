"""
dt_r4/data.py

数据加载、清洗、噪声构造、按 seed 划分节点数据等。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
# 3) 简单数据容器
# =========================
class SimpleData:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y


# =========================
# 5) 数据加载 + 噪声生成
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


def build_noise_variants_fixed(base_csv_path: str, dataset_seed: int = 123):
    """
    只生成/返回 NOISE_VARIANTS 中声明的版本（当前即4个）。
    dataset_seed 用于“固定噪声数据集”，避免不同seed下噪声CSV不同导致比较不干净。
    """
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
):
    """
    每个 seed 会重新划分节点数据池（用于重复实验）。

    如果 pre_split_poison 为 True 且节点属于恶意节点（i < malicious_nodes）并且
    attack_mode == "label_flip"，则先对该节点的完整子集做标签翻转，再拆分
    train / test；这样训练和测试集都会被同样的攻击污染。
    """
    df = load_and_clean_csv(csv_path)

    indices = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_size = len(df) // num_nodes

    node_data_objects = []
    for i in range(num_nodes):
        start = i * split_size
        end = None if i == num_nodes - 1 else (i + 1) * split_size
        node_idx = indices[start:end]
        node_df = df.iloc[node_idx]

        if (
            pre_split_poison
            and i < malicious_nodes
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
            {"train": SimpleData(train_x, train_y), "test": SimpleData(test_x, test_y)}
        )

    return node_data_objects, df
