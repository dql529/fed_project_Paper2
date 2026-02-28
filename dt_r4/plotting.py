from __future__ import annotations

"""Plotting utilities for r4 experiment outputs.

This module intentionally keeps a small, dependency-light API used by:
- r4_agg_minitest.py postprocessing scripts
- plot_from_csv.py
- paper_figs.py
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def load_summary_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_rounds_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_nodes_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


DT_LEVELS = {"D0", "D1", "D2"}
SUMMARY_REQUIRED_COLUMNS = {
    "seed",
    "attack",
    "method",
    "dt_level",
    "mal_nodes",
}
ROUNDS_REQUIRED_COLUMNS = {
    "seed",
    "round",
    "attack",
    "method",
    "dt_level",
    "mal_nodes",
}
NODES_REQUIRED_COLUMNS = {
    "seed",
    "node_id",
    "is_malicious",
    "R4",
    "KL_q_p",
    "Rep",
    "passed_gate",
    "pi",
    "R2",
    "R2_source",
    "attack",
    "dt_level",
    "mal_nodes",
    "method",
}


def parse_csv_list(value: str | None) -> Optional[List[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    out = [x.strip() for x in text.split(",") if x.strip()]
    return out if out else None


def _ensure_columns(df: pd.DataFrame, name: str, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _to_str_set(values: Iterable) -> set[str]:
    out: set[str] = set()
    for v in values:
        sv = str(v).strip()
        if sv:
            out.add(sv)
    return out


def _validate_filter_membership(
    df: pd.DataFrame,
    col: str,
    values: Optional[Sequence],
    name: str,
    label: str,
) -> None:
    if values is None:
        return
    if df.empty:
        raise ValueError(f"Cannot validate requested {label}: {name} data is empty")
    if col not in df.columns:
        raise ValueError(f"{name} missing required column '{col}' for {label} filter")
    available = _to_str_set(df[col].dropna().tolist())
    wanted = _to_str_set(values)
    if not wanted:
        return
    missing = sorted(x for x in wanted if x not in available)
    if missing:
        raise ValueError(
            f"Requested {label} not present in {name}: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(available))}"
        )


def _validate_int_membership(
    df: pd.DataFrame,
    col: str,
    values: Optional[Sequence[int]],
    name: str,
    label: str,
) -> None:
    if values is None:
        return
    if df.empty:
        raise ValueError(f"Cannot validate requested {label}: {name} data is empty")
    if col not in df.columns:
        raise ValueError(f"{name} missing required column '{col}' for {label} filter")
    raw = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    available = set(int(x) for x in raw)
    wanted = [int(v) for v in values]
    missing = [str(v) for v in wanted if v not in available]
    if missing:
        raise ValueError(
            f"Requested {label} not present in {name}: {', '.join(missing)}. "
            f"Available: {', '.join(str(x) for x in sorted(available))}"
        )


def validate_plot_inputs(
    *,
    summary_df: Optional[pd.DataFrame] = None,
    rounds_df: Optional[pd.DataFrame] = None,
    nodes_df: Optional[pd.DataFrame] = None,
    runs_df: Optional[pd.DataFrame] = None,
    attacks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    dt_levels: Optional[Sequence[str]] = None,
) -> None:
    if summary_df is not None and not summary_df.empty:
        _ensure_columns(summary_df, "summary.csv", SUMMARY_REQUIRED_COLUMNS)
        _validate_filter_membership(summary_df, "attack", attacks, "summary.csv", "attacks")
        _validate_filter_membership(summary_df, "method", methods, "summary.csv", "methods")
        _validate_int_membership(summary_df, "mal_nodes", mal_nodes, "summary.csv", "mal_nodes")
        if dt_levels is not None:
            _validate_filter_membership(summary_df, "dt_level", dt_levels, "summary.csv", "dt levels")
        bad_dt = {
            str(x).strip()
            for x in summary_df.get("dt_level", pd.Series(dtype=object)).dropna().tolist()
            if str(x).strip() and str(x).strip() not in DT_LEVELS
        }
        if bad_dt:
            raise ValueError(f"Invalid dt levels found in summary.csv: {', '.join(sorted(bad_dt))}")

    if rounds_df is not None and not rounds_df.empty:
        _ensure_columns(rounds_df, "rounds.csv", ROUNDS_REQUIRED_COLUMNS)

    if nodes_df is not None and not nodes_df.empty:
        _ensure_columns(nodes_df, "nodes.csv", NODES_REQUIRED_COLUMNS)

    if runs_df is not None and not runs_df.empty:
        _validate_filter_membership(runs_df, "attack", attacks, "runs.csv", "attacks")
        _validate_filter_membership(runs_df, "method", methods, "runs.csv", "methods")
        _validate_int_membership(runs_df, "mal_nodes", mal_nodes, "runs.csv", "mal_nodes")
        if dt_levels is not None:
            _validate_filter_membership(runs_df, "dt_level", dt_levels, "runs.csv", "dt levels")


def _parse_csv_value_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    out = [str(v).strip() for v in values if str(v).strip()]
    return out if out else None


def _mean_ci95(series: pd.Series) -> Tuple[float, float, float]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=0))
    n = int(arr.size)
    ci = float(1.96 * s / np.sqrt(max(1, n))) if n > 0 else float("nan")
    return m, ci, n


def _metric_columns(df: pd.DataFrame, metric: str) -> Tuple[str, str, str]:
    mean_col = f"{metric}_m"
    std_col = f"{metric}_s"
    if mean_col in df.columns:
        return mean_col, std_col, "grouped"
    if metric in df.columns:
        return metric, None, "raw"
    if metric == "benign_fp" and "benign_pass_rate" in df.columns:
        return "benign_pass_rate", None, "raw"
    return None, None, "missing"


def _metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    mean_col, _, mode = _metric_columns(df, metric)
    if mode == "missing":
        return pd.Series(dtype=float)
    if mode == "grouped":
        return pd.to_numeric(df[mean_col], errors="coerce")
    return pd.to_numeric(df[mean_col], errors="coerce")


def _choose_metric_column(df: pd.DataFrame, *names: str) -> str | None:
    for col in names:
        if col in df.columns:
            return col
    return None


def _apply_filters(
    df: pd.DataFrame,
    *,
    attacks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    level: Optional[str] = None,
    tau: Optional[float] = None,
    tau_sweep: Optional[Sequence[float]] = None,
    lam_m: Optional[float] = None,
    ref_size: Optional[int] = None,
    audit_size: Optional[int] = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    if attacks is not None:
        out = out[out["attack"].isin(list(attacks))]
    if methods is not None and "method" in out.columns:
        out = out[out["method"].isin(list(methods))]
    if mal_nodes is not None and "mal_nodes" in out.columns:
        out = out[out["mal_nodes"].isin(list(mal_nodes))]
    if dt_levels is not None and "dt_level" in out.columns:
        out = out[out["dt_level"].isin(list(dt_levels))]
    if level is not None and "level" in out.columns:
        out = out[out["level"] == level]
    if tau is not None and "tau_gate" in out.columns:
        out = out[np.isclose(pd.to_numeric(out["tau_gate"], errors="coerce"), float(tau))]
    if tau_sweep is not None and "tau_gate" in out.columns and len(list(tau_sweep)) > 0:
        out = out[pd.to_numeric(out["tau_gate"], errors="coerce").isin(list(map(float, tau_sweep)))]
    if lam_m is not None and "lambda_m" in out.columns:
        out = out[np.isclose(pd.to_numeric(out["lambda_m"], errors="coerce"), float(lam_m))]
    if ref_size is not None and "ref_size" in out.columns:
        out = out[pd.to_numeric(out["ref_size"], errors="coerce") == int(ref_size)]
    if audit_size is not None and "audit_size" in out.columns:
        out = out[pd.to_numeric(out["audit_size"], errors="coerce") == int(audit_size)]

    return out


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _to_float_series(arr: Iterable) -> np.ndarray:
    return np.asarray([float(x) for x in arr], dtype=float)


def _group_ci95_from_group(g: pd.DataFrame, metric: str) -> Tuple[float, float]:
    vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")
    n = len(vals)
    m = float(np.mean(vals))
    ci = float(1.96 * np.std(vals, ddof=0) / np.sqrt(max(1, n)))
    return m, ci


def _ci_fill(ax, x, y, yerr, color, alpha=0.15):
    x_arr = np.asarray(x, dtype=float)
    lo = y - yerr
    hi = y + yerr
    valid = np.isfinite(x_arr) & np.isfinite(y) & np.isfinite(yerr)
    if not np.any(valid):
        return
    ax.fill_between(x_arr[valid], lo[valid], hi[valid], color=color, alpha=alpha)


def _count_from_group(sg: pd.DataFrame, fallback: int) -> int:
    if "count" in sg.columns:
        counts = pd.to_numeric(sg["count"], errors="coerce").dropna()
        if len(counts) > 0:
            return int(counts.iloc[0])
    return int(fallback)


def plot_clean_holdout_vs_f(
    summary_df: pd.DataFrame,
    *,
    attacks: Sequence[str],
    methods: Sequence[str],
    dt_level: str,
    mal_nodes: Optional[Sequence[int]] = None,
    out_path: str | Path,
    metric: str = "clean_f1",
    label_flip_level: str = "L1",
) -> Path:
    _ensure_parent(out_path)
    df = summary_df.copy()
    if df.empty:
        return Path(out_path)
    df = df[df["dt_level"].astype(str) == str(dt_level)]

    if mal_nodes is None:
        if "mal_nodes" in df.columns:
            mal_nodes = sorted([int(x) for x in pd.unique(df["mal_nodes"]) if pd.notna(x)])
        else:
            mal_nodes = [0, 3, 5]

    mean_col, _, mode = _metric_columns(df, metric)
    if mode == "missing":
        raise ValueError(f"Metric '{metric}' not found in summary")

    n_rows = len(attacks)
    fig, axes = plt.subplots(1, n_rows, figsize=(5.2 * n_rows, 4), sharey=True)
    if n_rows == 1:
        axes = [axes]

    for ax, attack in zip(axes, attacks):
        for method in methods:
            sub = df[(df["attack"] == attack) & (df["method"] == method)]
            if attack == "label_flip":
                sub = sub[sub.get("level", "") == label_flip_level]

            y = []
            yerr = []
            x = []
            for f in mal_nodes:
                row = sub[sub["mal_nodes"] == f]
                if row.empty:
                    continue
                # if summary has grouped mean/std, prefer grouped CI
                if f"{metric}_m" in row.columns:
                    m = float(pd.to_numeric(row[f"{metric}_m"].iloc[0], errors="coerce"))
                    s = float(pd.to_numeric(row[f"{metric}_s"].iloc[0], errors="coerce") if f"{metric}_s" in row.columns else np.nan)
                    n = int(row.get("count", pd.Series([1])).iloc[0]) if "count" in row.columns else 0
                    ci = 1.96 * s / np.sqrt(max(1, n)) if np.isfinite(s) and n > 0 else 0.0
                else:
                    m, ci = _mean_ci95(row[mean_col])
                if np.isfinite(m):
                    x.append(int(f))
                    y.append(m)
                    yerr.append(ci if np.isfinite(ci) else 0.0)

            if not x:
                continue
            c = ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3, label=str(method))
            _ci_fill(ax, x, np.asarray(y), np.asarray(yerr), color=c[0].get_color())

        ax.set_xlabel("malicious clients")
        ax.set_title(f"{attack}")
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0.0, top=1.0)

    axes[0].set_ylabel(metric.replace("_", " "))
    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def plot_wmal_vs_round(
    rounds_df: pd.DataFrame,
    *,
    dt_level: str,
    mal_nodes: int,
    attacks: Sequence[str],
    out_path: str | Path,
    method: str = "weighted",
    label_flip_level: str = "L1",
) -> Path:
    _ensure_parent(out_path)
    df = rounds_df.copy()
    if df.empty:
        return Path(out_path)

    df = df[df["dt_level"].astype(str) == str(dt_level)]
    df = df[df["mal_nodes"].astype(int) == int(mal_nodes)]
    df = df[df["method"] == method]

    fig, ax = plt.subplots(figsize=(7, 4))
    for attack in attacks:
        sub = df[df["attack"] == attack]
        if attack == "label_flip" and "level" in sub.columns:
            sub = sub[sub["level"] == label_flip_level]
        if sub.empty:
            continue
        by_round = sub.groupby("round")["w_mal"].agg(["mean", "std", "count"])
        r = by_round.index.to_numpy()
        m = by_round["mean"].to_numpy(dtype=float)
        c = by_round["count"].to_numpy(dtype=int)
        s = by_round["std"].to_numpy(dtype=float)
        ci = 1.96 * np.divide(s, np.sqrt(c), out=np.zeros_like(s), where=c > 0)

        line = ax.plot(r, m, marker="o", linewidth=1.4, label=attack)
        _ci_fill(ax, r, m, ci, color=line[0].get_color())

    ax.set_title(f"W_mal vs round (f={mal_nodes}, dt={dt_level}, {method})")
    ax.set_xlabel("Round")
    ax.set_ylabel("W_mal")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def plot_r4_distribution(
    nodes_df: pd.DataFrame,
    *,
    attack: str,
    dt_level: str,
    mal_nodes: int,
    out_path: str | Path,
    method: str = "weighted",
    label_flip_level: str = "L1",
    r4_col: str = "R4",
) -> Path:
    _ensure_parent(out_path)
    df = nodes_df.copy()
    if df.empty:
        return Path(out_path)
    df = df[df["attack"] == attack]
    df = df[df["dt_level"].astype(str) == str(dt_level)]
    df = df[df["mal_nodes"].astype(int) == int(mal_nodes)]
    df = df[df["method"].astype(str) == str(method)]
    if attack == "label_flip" and "level" in df.columns:
        df = df[df["level"] == label_flip_level]

    if r4_col not in df.columns:
        raise ValueError(f"Column '{r4_col}' not in nodes csv")

    benign = pd.to_numeric(df.loc[df["is_malicious"] == 0, r4_col], errors="coerce").dropna().to_numpy()
    malicious = pd.to_numeric(df.loc[df["is_malicious"] == 1, r4_col], errors="coerce").dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [benign, malicious]
    labels = ["benign", "malicious"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#4c72b0", "#dd8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_title(f"R4 distribution @ dt={dt_level}, f={mal_nodes}, {attack}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("R4")
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def plot_ablation_multiattack(
    summary_df: pd.DataFrame,
    *,
    dt_level: str,
    attacks: Sequence[str],
    out_path: str | Path,
    mal_nodes: Sequence[int],
    metric: str = "polluted_f1",
) -> Path:
    _ensure_parent(out_path)
    df = summary_df.copy()
    if df.empty:
        return Path(out_path)

    df = df[df["dt_level"].astype(str) == str(dt_level)]
    fig, axes = plt.subplots(1, len(attacks), figsize=(6 * len(attacks), 4), sharey=True)
    if len(attacks) == 1:
        axes = [axes]

    for ax, attack in zip(axes, attacks):
        sub = df[df["attack"] == attack]
        methods = [x for x in pd.unique(sub["method"]) if pd.notna(x)]
        if not len(methods):
            continue

        for f in mal_nodes:
            sf = sub[sub["mal_nodes"] == int(f)]
            means = []
            cis = []
            x = []
            for m in methods:
                row = sf[sf["method"] == m]
                if row.empty:
                    continue
                if f"{metric}_m" in row.columns:
                    mean = float(pd.to_numeric(row[f"{metric}_m"].iloc[0], errors="coerce"))
                    std = float(pd.to_numeric(row[f"{metric}_s"].iloc[0], errors="coerce") if f"{metric}_s" in row.columns else np.nan)
                    count = int(row.get("count", pd.Series([1])).iloc[0]) if "count" in row.columns else 0
                    ci = 1.96 * std / np.sqrt(max(1, count)) if np.isfinite(std) and count > 0 else 0.0
                else:
                    mean, ci = _mean_ci95(row[metric]) if metric in row.columns else (float("nan"), float("nan"))
                if np.isfinite(mean):
                    x.append(str(m))
                    means.append(mean)
                    cis.append(ci if np.isfinite(ci) else 0.0)
            if not x:
                continue
            line = ax.errorbar(
                np.arange(len(x)),
                means,
                yerr=cis,
                marker="o",
                capsize=3,
                label=f"f={f}",
                linewidth=1.3,
            )
            _ci_fill(ax, np.arange(len(x)), np.asarray(means), np.asarray(cis), color=line[0].get_color())
            ax.set_xticks(np.arange(len(x)))
            ax.set_xticklabels(x, rotation=20)

        ax.set_title(f"{attack}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Method")
        ax.grid(alpha=0.2)
        if attack == attacks[0]:
            ax.set_ylabel(metric.replace("_", " "))
        ax.legend()

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)

def plot_sdt_vs_round(
    rounds_df: pd.DataFrame,
    *,
    dt_level: str,
    mal_nodes: int = 5,
    attacks: Optional[Sequence[str]] = None,
    out_path: str | Path = "plot_sdt_vs_round.png",
    methods: Optional[Sequence[str]] = None,
    label_flip_level: str = "L1",
) -> Path:
    _ensure_parent(out_path)
    df = rounds_df.copy()
    if df.empty:
        return Path(out_path)
    if "S_DT" not in df.columns:
        return Path(out_path)

    df = df[df["dt_level"].astype(str) == str(dt_level)]
    df = df[df["mal_nodes"].astype(int) == int(mal_nodes)]
    if methods is not None:
        df = df[df["method"].isin(list(methods))]

    if attacks is None:
        attacks = sorted(pd.unique(df["attack"]).tolist())

    fig, ax = plt.subplots(figsize=(7, 4))
    for attack in attacks:
        sub = df[df["attack"] == attack].copy()
        if attack == "label_flip" and "level" in sub.columns:
            sub = sub[sub["level"] == label_flip_level]
        if sub.empty:
            continue

        by_round = sub.groupby("round")["S_DT"].agg(["mean", "std", "count"])
        r = by_round.index.to_numpy()
        m = by_round["mean"].to_numpy(dtype=float)
        s = by_round["std"].to_numpy(dtype=float)
        c = by_round["count"].to_numpy(dtype=int)
        ci = 1.96 * np.divide(s, np.sqrt(c), out=np.zeros_like(s), where=c > 0)

        line = ax.plot(r, m, marker="o", linewidth=1.4, label=attack)
        _ci_fill(ax, r, m, ci, color=line[0].get_color())

    ax.set_xlabel("Round")
    ax.set_ylabel("S_DT")
    ax.set_title(f"S_DT vs round (f={mal_nodes}, dt={dt_level})")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def plot_fallback_prob_table(
    df: pd.DataFrame,
    *,
    attacks: Optional[Sequence[str]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    out_path: str | Path = "fallback_prob_table.csv",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if "fallback_rate" in df.columns:
        g = _apply_filters(df, attacks=attacks, methods=methods, mal_nodes=mal_nodes, dt_levels=dt_levels)
        metric = "fallback_rate"
    elif "fallback_rate_m" in df.columns:
        g = _apply_filters(df, attacks=attacks, methods=methods, mal_nodes=mal_nodes, dt_levels=dt_levels)
        metric = "fallback_rate_m"
    elif "fallback_flag" in df.columns:
        g = _apply_filters(df, attacks=attacks, methods=methods, mal_nodes=mal_nodes, dt_levels=dt_levels)
        metric = "fallback_flag"
    else:
        return pd.DataFrame()

    if g.empty:
        return pd.DataFrame()

    group_cols = ["attack", "dt_level", "mal_nodes"]
    if "method" in g.columns and g["method"].nunique() > 1:
        group_cols.append("method")
    if "lambda_m" in g.columns and g["lambda_m"].nunique() > 1:
        group_cols.append("lambda_m")
    if "ref_size" in g.columns and g["ref_size"].nunique() > 1:
        group_cols.append("ref_size")
    if "audit_size" in g.columns and g["audit_size"].nunique() > 1:
        group_cols.append("audit_size")
    if "tau_gate" in g.columns and g["tau_gate"].nunique() > 1:
        group_cols.append("tau_gate")

    rows = []
    for key, sub in g.groupby(group_cols, dropna=False):
        probs = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        probs = probs[np.isfinite(probs)]
        if probs.size == 0:
            continue
        row = {c: v for c, v in zip(group_cols, key if isinstance(key, tuple) else (key,))}
        m = float(np.mean(probs))
        s = float(np.std(probs, ddof=0))
        n = int(len(probs))
        row["fallback_prob"] = m
        row["fallback_prob_std"] = s
        row["fallback_prob_ci95"] = 1.96 * s / np.sqrt(max(1, n))
        row["count"] = n
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    if str(Path(out_path).suffix).lower() == ".csv":
        out.to_csv(out_path, index=False)
    else:
        # fallback: dump a readable table to txt/csv-like output
        out.to_csv(out_path, index=False)
    return out


def plot_cleanf1_vs_tau(
    summary_df: pd.DataFrame,
    *,
    out_path: str | Path,
    attacks: Optional[Sequence[str]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    label_flip_level: str = "L1",
    metric: str = "clean_f1",
) -> Path:
    _ensure_parent(out_path)
    if summary_df.empty:
        return Path(out_path)

    df = _apply_filters(
        summary_df,
        attacks=attacks,
        methods=methods,
        dt_levels=dt_levels,
        mal_nodes=mal_nodes,
    )
    if df.empty:
        return Path(out_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    if attacks is None:
        attacks = sorted(pd.unique(df["attack"]).tolist())

    mean_col = f"{metric}_m" if f"{metric}_m" in df.columns else metric

    for attack in attacks:
        sub = df[df["attack"] == attack]
        if attack == "label_flip":
            sub = sub[sub.get("level", "") == label_flip_level]
        by_tau = sub.groupby("tau_gate")[mean_col].agg(["mean", "std", "count"])
        if by_tau.empty:
            continue
        x = pd.to_numeric(by_tau.index, errors="coerce").to_numpy(dtype=float)
        y = by_tau["mean"].to_numpy(dtype=float)
        c = by_tau["count"].to_numpy(dtype=int)
        s = by_tau["std"].to_numpy(dtype=float)
        ci = 1.96 * np.divide(s, np.sqrt(c), out=np.zeros_like(s), where=c > 0)
        line = ax.plot(x, y, marker="o", linewidth=1.4, label=attack)
        _ci_fill(ax, x, y, ci, color=line[0].get_color())

    ax.set_title("clean F1 vs tau_gate")
    ax.set_xlabel("tau_gate")
    ax.set_ylabel("clean F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def plot_wmal_vs_tau(
    summary_df: pd.DataFrame,
    *,
    out_path: str | Path,
    attacks: Optional[Sequence[str]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    label_flip_level: str = "L1",
) -> Path:
    return plot_cleanf1_vs_tau(
        summary_df,
        out_path=out_path,
        attacks=attacks,
        dt_levels=dt_levels,
        mal_nodes=mal_nodes,
        methods=methods,
        label_flip_level=label_flip_level,
        metric="w_mal",
    )


def plot_fp_benign_vs_tau(
    summary_df: pd.DataFrame,
    *,
    out_path: str | Path,
    attacks: Optional[Sequence[str]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    label_flip_level: str = "L1",
) -> Path:
    _ensure_parent(out_path)
    if summary_df.empty:
        return Path(out_path)

    df = _apply_filters(
        summary_df,
        attacks=attacks,
        methods=methods,
        dt_levels=dt_levels,
        mal_nodes=mal_nodes,
    )
    metric = _choose_metric_column(df, "benign_pass_rate_m", "benign_pass_rate")
    if df.empty or metric is None:
        return Path(out_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    if attacks is None:
        attacks = sorted(pd.unique(df["attack"]).tolist())

    for attack in attacks:
        sub = df[df["attack"] == attack]
        if attack == "label_flip":
            sub = sub[sub.get("level", "") == label_flip_level]
        by_tau = sub.groupby("tau_gate")[metric].agg(["mean", "std", "count"])
        if by_tau.empty:
            continue
        x = pd.to_numeric(by_tau.index, errors="coerce").to_numpy(dtype=float)
        fp = 1.0 - by_tau["mean"].to_numpy(dtype=float)
        s = by_tau["std"].to_numpy(dtype=float)
        c = by_tau["count"].to_numpy(dtype=int)
        ci = 1.96 * np.divide(s, np.sqrt(c), out=np.zeros_like(s), where=c > 0)
        line = ax.plot(x, fp, marker="o", linewidth=1.4, label=f"{attack}")
        _ci_fill(ax, x, fp, ci, color=line[0].get_color())

    ax.set_title("benign false positive rate vs tau_gate")
    ax.set_xlabel("tau_gate")
    ax.set_ylabel("1 - benign_pass_rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)


def plot_refsize_sensitivity(
    summary_df: pd.DataFrame,
    *,
    out_dir: str | Path,
    prefix: str = "refsize",
    attacks: Optional[Sequence[str]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    fixed_audit_size: Optional[int] = None,
    fixed_lambda: Optional[float] = None,
    fixed_tau: Optional[float] = None,
    label_flip_level: str = "L1",
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return {}

    df = _apply_filters(
        summary_df,
        attacks=attacks,
        methods=methods,
        dt_levels=dt_levels,
        mal_nodes=mal_nodes,
        tau=fixed_tau,
        lam_m=fixed_lambda,
        audit_size=fixed_audit_size,
    )
    if df.empty:
        return {}

    if attacks is None:
        attacks = sorted(pd.unique(df["attack"]).tolist())

    out: Dict[str, Path] = {}
    for metric in ["clean_f1", "w_mal"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for attack in attacks:
            sub_attack = df[df["attack"] == attack]
            if attack == "label_flip":
                sub_attack = sub_attack[sub_attack.get("level", "") == label_flip_level]
            by_size = sub_attack.groupby("ref_size")
            xs = []
            ys = []
            ys_ci = []
            for size, sg in by_size:
                mean_col = f"{metric}_m" if f"{metric}_m" in sg.columns else metric
                std_col = f"{metric}_s" if f"{metric}_s" in sg.columns else None
                if mean_col not in sg.columns:
                    continue
                m = float(pd.to_numeric(sg[mean_col], errors="coerce").dropna().mean())
                n = _count_from_group(sg, 1)
                s = float(pd.to_numeric(sg[std_col], errors="coerce").dropna().iloc[0]) if std_col is not None else 0.0
                ci = 1.96 * s / np.sqrt(max(1, n)) if np.isfinite(s) and n > 0 else 0.0
                xs.append(int(size))
                ys.append(m)
                ys_ci.append(ci)

            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys, dtype=float)[order]
            ys_ci = np.asarray(ys_ci, dtype=float)[order]
            line = ax.plot(xs, ys, marker="o", linewidth=1.4, label=attack)
            _ci_fill(ax, xs, ys, ys_ci, color=line[0].get_color())

        ax.set_xlabel("|X_ref|")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_',' ')} vs ref size")
        ax.grid(alpha=0.2)
        ax.legend()
        path = out_dir / f"{prefix}_{metric}_vs_refsize.png"
        fig.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close(fig)
        out[f"{metric}_vs_refsize"] = path

    return out


def plot_auditsize_sensitivity(
    summary_df: pd.DataFrame,
    *,
    out_dir: str | Path,
    prefix: str = "auditsize",
    attacks: Optional[Sequence[str]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    methods: Optional[Sequence[str]] = None,
    fixed_ref_size: Optional[int] = None,
    fixed_lambda: Optional[float] = None,
    fixed_tau: Optional[float] = None,
    label_flip_level: str = "L1",
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        return {}

    df = _apply_filters(
        summary_df,
        attacks=attacks,
        methods=methods,
        dt_levels=dt_levels,
        mal_nodes=mal_nodes,
        tau=fixed_tau,
        lam_m=fixed_lambda,
        ref_size=fixed_ref_size,
    )
    if df.empty:
        return {}

    if attacks is None:
        attacks = sorted(pd.unique(df["attack"]).tolist())

    out: Dict[str, Path] = {}
    for metric in ["clean_f1", "w_mal"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for attack in attacks:
            sub_attack = df[df["attack"] == attack]
            if attack == "label_flip":
                sub_attack = sub_attack[sub_attack.get("level", "") == label_flip_level]
            by_size = sub_attack.groupby("audit_size")
            xs = []
            ys = []
            ys_ci = []
            for size, sg in by_size:
                mean_col = f"{metric}_m" if f"{metric}_m" in sg.columns else metric
                std_col = f"{metric}_s" if f"{metric}_s" in sg.columns else None
                if mean_col not in sg.columns:
                    continue
                m = float(pd.to_numeric(sg[mean_col], errors="coerce").dropna().mean())
                n = _count_from_group(sg, 1)
                s = float(pd.to_numeric(sg[std_col], errors="coerce").dropna().iloc[0]) if std_col is not None else 0.0
                ci = 1.96 * s / np.sqrt(max(1, n)) if np.isfinite(s) and n > 0 else 0.0
                xs.append(int(size))
                ys.append(m)
                ys_ci.append(ci)

            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys, dtype=float)[order]
            ys_ci = np.asarray(ys_ci, dtype=float)[order]
            line = ax.plot(xs, ys, marker="o", linewidth=1.4, label=attack)
            _ci_fill(ax, xs, ys, ys_ci, color=line[0].get_color())

        ax.set_xlabel("|D_ref^l|")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_',' ')} vs audit size")
        ax.grid(alpha=0.2)
        ax.legend()
        path = out_dir / f"{prefix}_{metric}_vs_auditsize.png"
        fig.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close(fig)
        out[f"{metric}_vs_auditsize"] = path

    return out


def plot_passrate_vs_round(
    rounds_df: pd.DataFrame,
    *,
    dt_level: str,
    mal_nodes: int,
    attacks: Optional[Sequence[str]] = None,
    out_path: str | Path,
    method: str = "weighted",
    metric: str = "benign_pass_rate",
    label_flip_level: str = "L1",
) -> Path:
    _ensure_parent(out_path)
    if rounds_df.empty:
        return Path(out_path)
    if "benign_pass_rate" not in rounds_df.columns or "malicious_pass_rate" not in rounds_df.columns:
        return Path(out_path)

    df = rounds_df.copy()
    df = df[df["dt_level"].astype(str) == str(dt_level)]
    df = df[df["mal_nodes"].astype(int) == int(mal_nodes)]
    df = df[df["method"] == method]

    if attacks is None:
        attacks = sorted(pd.unique(df["attack"]).tolist())

    fig, ax = plt.subplots(figsize=(7, 4))
    for attack in attacks:
        sub = df[df["attack"] == attack]
        if attack == "label_flip" and "level" in sub.columns:
            sub = sub[sub["level"] == label_flip_level]

        if metric == "benign_pass_rate":
            bcol = "benign_pass_rate"
            mcol = "malicious_pass_rate"
            for idx, (col, label, marker) in enumerate(
                [(bcol, f"{attack} benign", "o"), (mcol, f"{attack} malicious", "s")]
            ):
                by_round = sub.groupby("round")[col].agg(["mean", "std", "count"])
                r = by_round.index.to_numpy()
                y = by_round["mean"].to_numpy(dtype=float)
                c = by_round["count"].to_numpy(dtype=int)
                s = by_round["std"].to_numpy(dtype=float)
                ci = 1.96 * np.divide(s, np.sqrt(c), out=np.zeros_like(s), where=c > 0)
                line = ax.plot(r, y, marker=marker, linewidth=1.2, label=label)
                _ci_fill(ax, r, y, ci, color=line[0].get_color())
        else:
            by_round = sub.groupby("round")[metric].agg(["mean", "std", "count"])
            r = by_round.index.to_numpy()
            y = by_round["mean"].to_numpy(dtype=float)
            c = by_round["count"].to_numpy(dtype=int)
            s = by_round["std"].to_numpy(dtype=float)
            ci = 1.96 * np.divide(s, np.sqrt(c), out=np.zeros_like(s), where=c > 0)
            line = ax.plot(r, y, marker="o", linewidth=1.2, label=attack)
            _ci_fill(ax, r, y, ci, color=line[0].get_color())

    ax.set_title(f"Pass-rate vs round (f={mal_nodes}, dt={dt_level}, {method})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return Path(out_path)

def plot_adaptive_mimic_vs_lambda(
    runs_df: pd.DataFrame,
    *,
    nodes_df: Optional[pd.DataFrame] = None,
    dt_level: str = "D1",
    out_dir: str | Path = ".",
    mal_nodes: Optional[Sequence[int]] = None,
    method: str = "weighted",
    fixed_ref_size: Optional[int] = None,
    fixed_audit_size: Optional[int] = None,
    label_flip_level: str = "L1",
    out_prefix: str = "adaptive_mimic",
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if runs_df.empty:
        return {}

    df = runs_df[(runs_df["attack"] == "adaptive_mimic")]
    if df.empty:
        return {}
    if mal_nodes is not None and "mal_nodes" in df.columns:
        df = df[df["mal_nodes"].isin(list(mal_nodes))]
    if dt_level is not None:
        df = df[df["dt_level"].astype(str) == str(dt_level)]
    if method is not None and "method" in df.columns:
        df = df[df["method"] == method]
    if fixed_ref_size is not None:
        df = df[pd.to_numeric(df["ref_size"], errors="coerce") == int(fixed_ref_size)]
    if fixed_audit_size is not None:
        df = df[pd.to_numeric(df["audit_size"], errors="coerce") == int(fixed_audit_size)]
    if "level" in df.columns:
        df = df[df["level"] == label_flip_level]

    if df.empty:
        return {}

    out: Dict[str, Path] = {}

    # clean F1 vs lambda
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    by_lm = df.groupby("lambda_m")
    xs = []
    ys = []
    ys_ci = []
    for lm, sg in by_lm:
        if "clean_f1_m" in sg.columns:
            m = float(pd.to_numeric(sg["clean_f1_m"], errors="coerce").dropna().mean())
            s = float(pd.to_numeric(sg["clean_f1_s"], errors="coerce").dropna().iloc[0]) if "clean_f1_s" in sg.columns else 0.0
            n = _count_from_group(sg, len(sg))
        elif "clean_f1" in sg.columns:
            m, s = _mean_ci95(sg["clean_f1"])
            n = int(len(sg))
        else:
            continue
        ci = 1.96 * s / np.sqrt(max(1, n)) if np.isfinite(s) and n > 0 else 0.0
        xs.append(float(lm))
        ys.append(m)
        ys_ci.append(ci)

    if xs:
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys, dtype=float)[order]
        ys_ci = np.asarray(ys_ci, dtype=float)[order]
        line = ax1.plot(xs, ys, marker="o", linewidth=1.4)
        _ci_fill(ax1, xs, ys, ys_ci, color=line[0].get_color())
        ax1.set_xlabel("lambda_m")
        ax1.set_ylabel("clean F1")
        ax1.set_ylim(0, 1)
        ax1.set_title("Adaptive mimic: clean F1 vs lambda")
        ax1.grid(alpha=0.2)
        p1 = out_dir / f"{out_prefix}_cleanf1_vs_lambda.png"
        fig1.tight_layout()
        plt.savefig(p1, dpi=200)
        plt.close(fig1)
        out["cleanf1"] = p1

    # W_mal vs lambda
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    xs = []
    ys = []
    ys_ci = []
    by_lm = df.groupby("lambda_m")
    for lm, sg in by_lm:
        if "w_mal_m" in sg.columns:
            m = float(pd.to_numeric(sg["w_mal_m"], errors="coerce").dropna().mean())
            s = float(pd.to_numeric(sg["w_mal_s"], errors="coerce").dropna().iloc[0]) if "w_mal_s" in sg.columns else 0.0
            n = _count_from_group(sg, len(sg))
        elif "w_mal" in sg.columns:
            m, s = _mean_ci95(sg["w_mal"])
            n = int(len(sg))
        else:
            continue
        ci = 1.96 * s / np.sqrt(max(1, n)) if np.isfinite(s) and n > 0 else 0.0
        xs.append(float(lm))
        ys.append(m)
        ys_ci.append(ci)

    if xs:
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys, dtype=float)[order]
        ys_ci = np.asarray(ys_ci, dtype=float)[order]
        line = ax2.plot(xs, ys, marker="o", linewidth=1.4)
        _ci_fill(ax2, xs, ys, ys_ci, color=line[0].get_color())
        ax2.set_xlabel("lambda_m")
        ax2.set_ylabel("W_mal")
        ax2.set_ylim(0, 1)
        ax2.set_title("Adaptive mimic: W_mal vs lambda")
        ax2.grid(alpha=0.2)
        p2 = out_dir / f"{out_prefix}_wmal_vs_lambda.png"
        fig2.tight_layout()
        plt.savefig(p2, dpi=200)
        plt.close(fig2)
        out["wmal"] = p2

    # R4 distribution vs lambda (benign vs malicious)
    if nodes_df is not None and not nodes_df.empty:
        ndf = nodes_df.copy()
        ndf = ndf[ndf["attack"] == "adaptive_mimic"]
        ndf = ndf[ndf["dt_level"].astype(str) == str(dt_level)]
        if method is not None and "method" in ndf.columns:
            ndf = ndf[ndf["method"] == method]
        if mal_nodes is not None and "mal_nodes" in ndf.columns:
            ndf = ndf[ndf["mal_nodes"].isin(list(mal_nodes))]
        if fixed_ref_size is not None and "ref_size" in ndf.columns:
            ndf = ndf[pd.to_numeric(ndf["ref_size"], errors="coerce") == int(fixed_ref_size)]
        if fixed_audit_size is not None and "audit_size" in ndf.columns:
            ndf = ndf[pd.to_numeric(ndf["audit_size"], errors="coerce") == int(fixed_audit_size)]
        if not ndf.empty and "R4" in ndf.columns:
            if "level" in ndf.columns:
                ndf = ndf[ndf["level"] == label_flip_level]

            by_lambda = ndf.groupby("lambda_m")
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            first_label = True
            for group_name, sg in by_lambda:
                if sg.empty:
                    continue
                b = pd.to_numeric(sg.loc[sg["is_malicious"] == 0, "R4"], errors="coerce").dropna()
                m = pd.to_numeric(sg.loc[sg["is_malicious"] == 1, "R4"], errors="coerce").dropna()
                if b.empty and m.empty:
                    continue
                ax3.scatter(
                    [float(group_name)] * len(b),
                    b,
                    alpha=0.55,
                    marker="o",
                    label="benign" if first_label else None,
                )
                ax3.scatter(
                    [float(group_name)] * len(m),
                    m,
                    alpha=0.55,
                    marker="s",
                    label="malicious" if first_label else None,
                )
                first_label = False

            if len(ax3.collections) > 0:
                ax3.set_xlabel("lambda_m")
                ax3.set_ylabel("R4")
                ax3.set_ylim(0, 1)
                ax3.set_title("Adaptive mimic: R4 distribution vs lambda")
                ax3.grid(alpha=0.2)
                p3 = out_dir / f"{out_prefix}_r4_vs_lambda.png"
                ax3.legend()
                fig3.tight_layout()
                plt.savefig(p3, dpi=200)
                plt.close(fig3)
                out["r4"] = p3

    return out


def make_plots_from_csv(
    *,
    runs_csv: str = "runs.csv",
    summary_csv: str = "summary.csv",
    out_dir: str = ".",
    attacks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    label_flip_level: str = "L1",
    num_nodes: int = 10,
    metric: str = "polluted_f1",
    tau: Optional[float] = None,
    tau_sweep: Optional[Sequence[float]] = None,
    lam_m: Optional[float] = None,
    ref_size: Optional[int] = None,
    audit_size: Optional[int] = None,
) -> Dict[str, Path]:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv) if summary_csv else pd.DataFrame()
    runs_df = pd.read_csv(runs_csv) if runs_csv else pd.DataFrame()
    validate_plot_inputs(
        summary_df=summary_df,
        runs_df=runs_df,
        attacks=attacks,
        methods=methods,
        mal_nodes=mal_nodes,
        dt_levels=dt_levels,
    )
    outputs: Dict[str, Path] = {}
    dt_levels_for_plot = ["D1"]

    if not summary_df.empty:
        # optional filtering
        sum_df = _apply_filters(
            summary_df,
            attacks=attacks,
            methods=methods,
            mal_nodes=mal_nodes,
            dt_levels=dt_levels,
            tau=tau,
            tau_sweep=tau_sweep,
            lam_m=lam_m,
            ref_size=ref_size,
            audit_size=audit_size,
        )

        if mal_nodes is None:
            if "mal_nodes" in sum_df.columns:
                mal_nodes_for_plot = sorted(int(x) for x in pd.unique(sum_df["mal_nodes"]) if pd.notna(x))
            else:
                mal_nodes_for_plot = [3, 5, 0]
        else:
            mal_nodes_for_plot = list(mal_nodes)

        if dt_levels is None:
            if "dt_level" in sum_df.columns:
                dt_levels_for_plot = [str(x) for x in pd.unique(sum_df["dt_level"]) if str(x).strip()]
        else:
            dt_levels_for_plot = list(dt_levels)

        # FigA-like: clean holdout vs f
        for dt in dt_levels_for_plot:
            path = out_dir_path / f"plotA_{metric}_dt{dt}.png"
            out = plot_clean_holdout_vs_f(
                sum_df,
                attacks=attacks or ["label_flip", "stealth_amp", "dt_logit_scale"],
                methods=methods or ["weighted", "mean", "median", "trimmed_mean"],
                dt_level=dt,
                mal_nodes=mal_nodes_for_plot,
                out_path=path,
                metric="clean_f1" if metric in {"clean_f1", "clean_acc", "polluted_f1", "polluted_acc", "w_mal"} else metric,
                label_flip_level=label_flip_level,
            )
            outputs[f"clean_holdout_vs_f_{dt}"] = out

        # FigB-like: Wmal vs round for largest f
        target_f = max(mal_nodes_for_plot) if mal_nodes_for_plot else 5
        if not runs_df.empty:
            rounds_df = runs_df
        else:
            rounds_df = pd.DataFrame()
        if "round" not in rounds_df.columns and "round" in summary_df.columns:
            # best effort only
            pass
        for dt in dt_levels_for_plot:
            path = out_dir_path / f"plotB_wmal_vs_round_dt{dt}.png"
            try:
                out = plot_wmal_vs_round(
                    rounds_df,
                    dt_level=dt,
                    mal_nodes=target_f,
                    attacks=attacks or ["label_flip", "dt_logit_scale"],
                    out_path=path,
                    method="weighted",
                    label_flip_level=label_flip_level,
                )
                outputs[f"wmal_vs_round_{dt}"] = out
            except Exception:
                pass

        # sensitivity plots
        if (tau_sweep is not None and len(tau_sweep) > 1) or ("tau_gate" in sum_df.columns and sum_df["tau_gate"].nunique() > 1):
            path = out_dir_path / "plot_cleanf1_vs_tau.png"
            outputs["cleanf1_vs_tau"] = plot_cleanf1_vs_tau(
                sum_df,
                out_path=path,
                attacks=attacks,
                dt_levels=dt_levels_for_plot,
                mal_nodes=mal_nodes_for_plot,
                methods=methods,
                label_flip_level=label_flip_level,
                metric="clean_f1",
            )
            path = out_dir_path / "plot_wmal_vs_tau.png"
            outputs["wmal_vs_tau"] = plot_wmal_vs_tau(
                sum_df,
                out_path=path,
                attacks=attacks,
                dt_levels=dt_levels_for_plot,
                mal_nodes=mal_nodes_for_plot,
                methods=methods,
                label_flip_level=label_flip_level,
            )
            if "benign_pass_rate_m" in sum_df.columns:
                path = out_dir_path / "plot_fp_benign_vs_tau.png"
                outputs["fp_benign_vs_tau"] = plot_fp_benign_vs_tau(
                    sum_df,
                    out_path=path,
                    attacks=attacks,
                    dt_levels=dt_levels_for_plot,
                    mal_nodes=mal_nodes_for_plot,
                    methods=methods,
                    label_flip_level=label_flip_level,
                )

        # fallback table
        if "fallback_rate" in runs_df.columns or "fallback_rate_m" in summary_df.columns:
            p = out_dir_path / "fallback_prob_table.csv"
            table = plot_fallback_prob_table(
                runs_df if not runs_df.empty else summary_df,
                attacks=attacks,
                dt_levels=dt_levels_for_plot,
                mal_nodes=mal_nodes_for_plot,
                methods=methods,
            )
            if p and not table.empty:
                table.to_csv(p, index=False)
                outputs["fallback_prob_table"] = p

    # optional run-level pass-rate and sensitivity plots
    if not runs_df.empty:
        if dt_levels is None and "dt_level" in runs_df.columns:
            dt_levels_for_plot = [str(x) for x in pd.unique(runs_df["dt_level"]) if str(x).strip()]
        for dt in dt_levels_for_plot:
            path = out_dir_path / f"plot_fallback_vs_round_dt{dt}.png"
            if "fallback_flag" in runs_df.columns:
                # deprecated: round traces are in rounds_df, not runs_df. keep graceful skip.
                outputs["fallback_vs_round"] = path

    return outputs


__all__ = [
    "load_nodes_csv",
    "load_rounds_csv",
    "load_summary_csv",
    "parse_csv_list",
    "validate_plot_inputs",
    "plot_ablation_multiattack",
    "plot_clean_holdout_vs_f",
    "plot_cleanf1_vs_tau",
    "plot_fp_benign_vs_tau",
    "plot_wmal_vs_tau",
    "plot_sdt_vs_round",
    "plot_fallback_prob_table",
    "plot_r4_distribution",
    "plot_wmal_vs_round",
    "plot_refsize_sensitivity",
    "plot_auditsize_sensitivity",
    "plot_passrate_vs_round",
    "plot_adaptive_mimic_vs_lambda",
    "make_plots_from_csv",
]
