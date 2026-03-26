from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "CMU Serif", "DejaVu Serif"],
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#404040",
    }
)

ATTACKS = ["label_flip", "stealth_amp", "dt_logit_scale"]
ROLE_ORDER = ["benign_clean", "benign_light", "benign_medium", "benign_heavy", "malicious"]
COLORS = {"weighted": "#1f5a91", "median": "#b65e2e"}


def _choose_candidate(summary_df: pd.DataFrame) -> str:
    sub = summary_df[pd.to_numeric(summary_df["mal_nodes"], errors="coerce") == 3].copy()
    agg = (
        sub.groupby("experiment_id", as_index=False)
        .agg(
            strict_hits=("success_strict", "sum"),
            mean_drop=("median_deploy_drop_abs", "mean"),
            min_margin=("weighted_minus_median_deploy", "min"),
        )
        .sort_values(["strict_hits", "mean_drop", "min_margin"], ascending=[False, False, False])
    )
    if agg.empty:
        raise ValueError("No stress candidates found")
    return str(agg.iloc[0]["experiment_id"])


def _facet_axes(n: int, height: float = 3.8):
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, height), constrained_layout=True)
    if n == 1:
        axes = [axes]
    return fig, axes


def plot_stress_frontier(summary_df: pd.DataFrame, out_path: Path) -> None:
    sub = summary_df[pd.to_numeric(summary_df["mal_nodes"], errors="coerce") == 3].copy()
    fig, axes = _facet_axes(3)
    for ax, attack in zip(axes, ATTACKS):
        sdf = sub[sub["attack"] == attack].copy()
        if sdf.empty:
            ax.axis("off")
            continue
        for _, row in sdf.iterrows():
            size = 90 + 900 * max(0.0, float(row["weighted_w_mal"]))
            color = "#2c7a7b" if str(row["family"]) != "iid" else "#7f8c8d"
            ax.scatter(
                float(row["median_deploy_drop_abs"]),
                float(row["weighted_minus_median_deploy"]),
                s=size,
                alpha=0.82,
                color=color,
                edgecolor="#1f1f1f",
                linewidth=0.7,
            )
            ax.text(
                float(row["median_deploy_drop_abs"]) + 0.002,
                float(row["weighted_minus_median_deploy"]) + 0.001,
                str(row["experiment_id"]),
                fontsize=8,
            )
        ax.axvline(0.10, color="#7a1f1f", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#444444", linestyle=":", linewidth=1.0)
        ax.set_title(attack)
        ax.set_xlabel("Median deploy drop vs base")
        ax.set_ylabel("Weighted - Median (deploy)")
    fig.suptitle("Stress frontier", y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_suppression_phase(summary_df: pd.DataFrame, out_path: Path) -> None:
    sub = summary_df[pd.to_numeric(summary_df["mal_nodes"], errors="coerce") == 3].copy()
    fig, axes = _facet_axes(3)
    for ax, attack in zip(axes, ATTACKS):
        sdf = sub[sub["attack"] == attack].copy()
        if sdf.empty:
            ax.axis("off")
            continue
        ax.scatter(sdf["median_w_mal"], sdf["deploy_f1_median"], color=COLORS["median"], s=110, alpha=0.78, label="median")
        ax.scatter(sdf["weighted_w_mal"], sdf["deploy_f1_weighted"], color=COLORS["weighted"], s=110, alpha=0.78, label="weighted")
        for _, row in sdf.iterrows():
            ax.plot(
                [float(row["median_w_mal"]), float(row["weighted_w_mal"])],
                [float(row["deploy_f1_median"]), float(row["deploy_f1_weighted"])],
                color="#9a9a9a",
                linewidth=0.8,
                alpha=0.55,
            )
            ax.text(float(row["weighted_w_mal"]) + 0.002, float(row["deploy_f1_weighted"]) + 0.001, str(row["experiment_id"]), fontsize=8)
        ax.set_title(attack)
        ax.set_xlabel("Admitted malicious weight mass")
        ax.set_ylabel("Deploy F1")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Suppression-performance phase map", y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dumbbell(summary_df: pd.DataFrame, candidate: str, out_path: Path, title: str) -> None:
    sub = summary_df[
        (summary_df["experiment_id"].astype(str) == str(candidate))
        & (pd.to_numeric(summary_df["mal_nodes"], errors="coerce") == 3)
    ].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    metrics = [
        ("reference_f1_median", "reference_f1_weighted", "reference"),
        ("deploy_f1_median", "deploy_f1_weighted", "deploy"),
    ]
    for r, (mcol, wcol, prefix) in enumerate(metrics):
        for c, attack in enumerate(ATTACKS):
            ax = axes[r, c]
            row = sub[sub["attack"] == attack]
            if row.empty:
                ax.axis("off")
                continue
            row = row.iloc[0]
            bm = float(row[mcol]) + float(row[f"median_{prefix}_drop_abs"])
            bw = float(row[wcol]) + float(row[f"weighted_{prefix}_drop_abs"])
            cm = float(row[mcol])
            cw = float(row[wcol])
            ax.plot([0, 1], [bm, cm], color=COLORS["median"], linewidth=2.0)
            ax.scatter([0, 1], [bm, cm], color=COLORS["median"], s=80)
            ax.plot([0, 1], [bw, cw], color=COLORS["weighted"], linewidth=2.0)
            ax.scatter([0, 1], [bw, cw], color=COLORS["weighted"], s=80)
            ax.set_xticks([0, 1], ["baseline", "stress"])
            ax.set_title(f"{attack} - {prefix}")
            if c == 0:
                ax.set_ylabel(f"{prefix} F1")
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_violin_metric(ax, data: pd.DataFrame, metric: str, attack: str) -> None:
    sub = data[data["attack"] == attack].copy()
    if sub.empty:
        ax.axis("off")
        return
    arrays = []
    labels = []
    valid_arrays = []
    valid_positions = []
    for role in ROLE_ORDER:
        vals = pd.to_numeric(sub.loc[sub["node_noise_role"] == role, metric], errors="coerce").dropna().to_numpy()
        arrays.append(vals)
        labels.append(role.replace("benign_", "b-").replace("malicious", "mal"))
    palette = ["#5b8cc0", "#7ba6d6", "#a2b5d8", "#d1a561", "#c05a4f"]
    for i, arr in enumerate(arrays, start=1):
        finite = arr[np.isfinite(arr)]
        if finite.size >= 2:
            valid_arrays.append(finite)
            valid_positions.append(i)
    if valid_arrays:
        parts = ax.violinplot(
            valid_arrays,
            positions=valid_positions,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body, pos in zip(parts["bodies"], valid_positions):
            color = palette[pos - 1]
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor("#222222")
            body.set_linewidth(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#222222")
            parts["cmedians"].set_linewidth(1.0)
    for i, arr in enumerate(arrays, start=1):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            jitter = np.linspace(-0.08, 0.08, finite.size)
            ax.scatter(np.full(finite.size, i) + jitter, finite, s=12, alpha=0.55, color="#222222")
            ax.hlines(np.median(finite), i - 0.16, i + 0.16, color="#222222", linewidth=1.0)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=25)
    ax.set_title(attack)
    ax.set_ylabel(metric)


def plot_role_raincloud(nodes_df: pd.DataFrame, candidate: str, out_path: Path, metrics: List[str], title: str) -> None:
    sub = nodes_df[
        (nodes_df["experiment_id"].astype(str) == str(candidate))
        & (nodes_df["method"].astype(str) == "weighted")
        & (pd.to_numeric(nodes_df["mal_nodes"], errors="coerce") == 3)
    ].copy()
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 3.4 * len(metrics)), constrained_layout=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for r, metric in enumerate(metrics):
        for c, attack in enumerate(ATTACKS):
            _plot_violin_metric(axes[r, c], sub, metric, attack)
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", type=str, default="artifacts/stress_analysis/stress_summary.csv")
    ap.add_argument("--nodes-csv", type=str, default="artifacts/stress_analysis/stress_nodes_long.csv")
    ap.add_argument("--candidate", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="artifacts/stress_figs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(args.summary_csv)
    nodes_df = pd.read_csv(args.nodes_csv) if Path(args.nodes_csv).exists() else pd.DataFrame()
    candidate = str(args.candidate).strip() or _choose_candidate(summary_df)

    plot_stress_frontier(summary_df, out_dir / "StressFrontier.png")
    plot_suppression_phase(summary_df, out_dir / "SuppressionPhaseMap.png")
    plot_dumbbell(summary_df, candidate, out_dir / "Diagnostic_Dumbbell.png", f"Baseline vs stress: {candidate}")
    if not nodes_df.empty:
        plot_role_raincloud(nodes_df, candidate, out_dir / "Diagnostic_RoleRaincloud.png", ["R4", "Rep", "KL_q_p", "passed_gate"], f"Role separation: {candidate}")
    plot_dumbbell(summary_df, candidate, out_dir / "Paper_MainStressResult.png", f"Main stress result: {candidate}")
    if not nodes_df.empty:
        plot_role_raincloud(nodes_df, candidate, out_dir / "Paper_MechanismFigure.png", ["R4", "Rep"], f"Mechanism figure: {candidate}")
    plot_suppression_phase(summary_df[summary_df["experiment_id"].astype(str) == str(candidate)], out_dir / "Paper_SuppressionOutcome.png")
    print(f"[ok] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
