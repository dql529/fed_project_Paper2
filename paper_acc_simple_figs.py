from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTACKS = ["label_flip", "stealth_amp", "dt_logit_scale"]
ROUND_ATTACKS = ["label_flip", "dt_logit_scale"]
METHODS = ["weighted", "median", "mean", "trimmed_mean"]
F_VALUES = [1, 2, 3, 4, 5]
COLORS = {
    "weighted": "#1f77b4",
    "median": "#d62728",
    "mean": "#2ca02c",
    "trimmed_mean": "#ff7f0e",
    "benign_rep": "#1f77b4",
    "malicious_rep": "#d62728",
    "clean_acc": "#1f77b4",
    "deploy_acc": "#2ca02c",
}


def _resolve_root(raw: Path) -> Path:
    if (raw / "config.json").exists():
        return raw
    direct = [d for d in raw.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if len(direct) == 1:
        return direct[0]
    preferred = [d for d in direct if d.name == "heterobenign"]
    if len(preferred) == 1:
        return preferred[0]
    raise FileNotFoundError(f"Could not resolve experiment root from: {raw}")


def _concat_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect_ablation_csvs(ablation_root: Path, filename: str) -> List[Path]:
    paths: List[Path] = []
    for f_value in F_VALUES:
        root = _resolve_root(ablation_root / f"f{f_value}")
        paths.append(root / filename)
    return paths


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _summary_mean(df: pd.DataFrame, attack: str, method: str, f_value: int, metric: str) -> float:
    sub = df[
        (df["attack"].astype(str) == str(attack))
        & (df["method"].astype(str) == str(method))
        & (pd.to_numeric(df["mal_nodes"], errors="coerce") == int(f_value))
    ]
    if attack == "label_flip" and "level" in sub.columns:
        sub = sub[sub["level"].astype(str) == "L1"]
    if sub.empty:
        return float("nan")
    col = f"{metric}_m" if f"{metric}_m" in sub.columns else metric
    return float(pd.to_numeric(sub.iloc[0][col], errors="coerce"))


def plot_simple_acc_summary(
    summary_df: pd.DataFrame,
    *,
    out_path: Path,
    dt_level: str,
) -> None:
    df = summary_df.copy()
    df = df[df["dt_level"].astype(str) == str(dt_level)]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2), sharex=True, sharey="row")
    metrics = [("clean_acc", "Reference ACC"), ("polluted_acc", "Deploy ACC")]

    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, attack in enumerate(ATTACKS):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                xs: List[int] = []
                ys: List[float] = []
                for f_value in F_VALUES:
                    mean = _summary_mean(df, attack, method, f_value, metric)
                    if np.isfinite(mean):
                        xs.append(int(f_value))
                        ys.append(mean)
                if xs:
                    ax.plot(
                        xs,
                        ys,
                        marker="o",
                        linewidth=1.8,
                        markersize=4.5,
                        color=COLORS.get(method, None),
                        label=method,
                    )
            ax.set_title(attack)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.22)
            if row_idx == 1:
                ax.set_xlabel("malicious clients")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Simple ACC Trends Across f", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _round_series(
    rounds_df: pd.DataFrame,
    *,
    attack: str,
    method: str,
    f_value: int,
    metric: str,
    dt_level: str,
) -> pd.DataFrame:
    sub = rounds_df[
        (rounds_df["attack"].astype(str) == str(attack))
        & (rounds_df["method"].astype(str) == str(method))
        & (pd.to_numeric(rounds_df["mal_nodes"], errors="coerce") == int(f_value))
        & (rounds_df["dt_level"].astype(str) == str(dt_level))
    ].copy()
    if attack == "label_flip" and "level" in sub.columns:
        sub = sub[sub["level"].astype(str) == "L1"]
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    return (
        sub.groupby("round")[metric]
        .mean()
        .reset_index()
        .sort_values("round")
    )


def plot_simple_round_metric(
    rounds_df: pd.DataFrame,
    *,
    out_path: Path,
    attack: str,
    metric: str,
    title: str,
    dt_level: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4), sharex=True, sharey=True)
    axes_arr = axes.reshape(-1)
    for idx, f_value in enumerate(F_VALUES):
        ax = axes_arr[idx]
        for method in METHODS:
            series = _round_series(
                rounds_df,
                attack=attack,
                method=method,
                f_value=f_value,
                metric=metric,
                dt_level=dt_level,
            )
            if series.empty:
                continue
            ax.plot(
                series["round"],
                series[metric],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=COLORS.get(method, None),
                label=method,
            )
        ax.set_title(f"f={f_value}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
        ax.set_xlabel("Round")
        ax.set_ylabel("ACC")
    axes_arr[-1].axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_weighted_rep_dynamics(
    rounds_df: pd.DataFrame,
    *,
    out_path: Path,
    attack: str,
    dt_level: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), sharex=True)
    axes_arr = axes.reshape(-1)
    legend_handles: list = []
    legend_labels: list[str] = []
    for idx, f_value in enumerate(F_VALUES):
        ax = axes_arr[idx]
        clean_series = _round_series(
            rounds_df,
            attack=attack,
            method="weighted",
            f_value=f_value,
            metric="round_clean_acc",
            dt_level=dt_level,
        )
        deploy_series = _round_series(
            rounds_df,
            attack=attack,
            method="weighted",
            f_value=f_value,
            metric="round_polluted_acc",
            dt_level=dt_level,
        )
        benign_rep = _round_series(
            rounds_df,
            attack=attack,
            method="weighted",
            f_value=f_value,
            metric="round_benign_rep",
            dt_level=dt_level,
        )
        malicious_rep = _round_series(
            rounds_df,
            attack=attack,
            method="weighted",
            f_value=f_value,
            metric="round_malicious_rep",
            dt_level=dt_level,
        )
        if not clean_series.empty:
            ax.plot(
                clean_series["round"],
                clean_series["round_clean_acc"],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                color=COLORS["clean_acc"],
                label="clean_acc",
            )
        if not deploy_series.empty:
            ax.plot(
                deploy_series["round"],
                deploy_series["round_polluted_acc"],
                marker="o",
                linewidth=1.6,
                markersize=3.8,
                color=COLORS["deploy_acc"],
                label="deploy_acc",
            )
        ax2 = ax.twinx()
        if not benign_rep.empty:
            ax2.plot(
                benign_rep["round"],
                np.log10(benign_rep["round_benign_rep"] + 1.0),
                marker="s",
                linewidth=1.5,
                markersize=3.5,
                linestyle="--",
                color=COLORS["benign_rep"],
                label="log10 benign_rep",
            )
        if not malicious_rep.empty:
            ax2.plot(
                malicious_rep["round"],
                np.log10(malicious_rep["round_malicious_rep"] + 1.0),
                marker="s",
                linewidth=1.5,
                markersize=3.5,
                linestyle="--",
                color=COLORS["malicious_rep"],
                label="log10 malicious_rep",
            )
        ax.set_title(f"f={f_value}")
        ax.set_xlabel("Round")
        ax.set_ylabel("ACC")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
        ax2.set_ylabel("log10(Rep+1)")
        if idx == 0:
            acc_handles, acc_labels = ax.get_legend_handles_labels()
            rep_handles, rep_labels = ax2.get_legend_handles_labels()
            legend_handles = acc_handles + rep_handles
            legend_labels = acc_labels + rep_labels

    axes_arr[-1].axis("off")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.98),
        )
    fig.suptitle(f"Weighted Accuracy and Reputation Dynamics: {attack}", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-root", type=str, default="artifacts/99_workspace_archive/fair_f_ablation")
    ap.add_argument("--round-root", type=str, default="artifacts/99_workspace_archive/acc_rounds_f_ablation")
    ap.add_argument("--out-dir", type=str, default="artifacts/01_appendix_upload/figures")
    ap.add_argument("--dt-level", type=str, default="D0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    summary_df = _concat_csvs(_collect_ablation_csvs(Path(args.ablation_root), "summary.csv"))
    rounds_df = _concat_csvs(_collect_ablation_csvs(Path(args.round_root), "rounds.csv"))

    plot_simple_acc_summary(
        summary_df,
        out_path=out_dir / "FigB1_simple_acc_summary_vs_f.png",
        dt_level=str(args.dt_level),
    )

    plot_simple_round_metric(
        rounds_df,
        out_path=out_dir / "FigB2_simple_round_clean_acc_label_flip.png",
        attack="label_flip",
        metric="round_clean_acc",
        title="Simple Round Clean ACC: label_flip",
        dt_level=str(args.dt_level),
    )
    plot_simple_round_metric(
        rounds_df,
        out_path=out_dir / "FigB3_simple_round_clean_acc_dt_logit_scale.png",
        attack="dt_logit_scale",
        metric="round_clean_acc",
        title="Simple Round Clean ACC: dt_logit_scale",
        dt_level=str(args.dt_level),
    )
    plot_simple_round_metric(
        rounds_df,
        out_path=out_dir / "FigB4_simple_round_deploy_acc_label_flip.png",
        attack="label_flip",
        metric="round_polluted_acc",
        title="Simple Round Deploy ACC: label_flip",
        dt_level=str(args.dt_level),
    )
    plot_simple_round_metric(
        rounds_df,
        out_path=out_dir / "FigB5_simple_round_deploy_acc_dt_logit_scale.png",
        attack="dt_logit_scale",
        metric="round_polluted_acc",
        title="Simple Round Deploy ACC: dt_logit_scale",
        dt_level=str(args.dt_level),
    )
    plot_weighted_rep_dynamics(
        rounds_df,
        out_path=out_dir / "FigB6_weighted_accuracy_rep_label_flip.png",
        attack="label_flip",
        dt_level=str(args.dt_level),
    )
    plot_weighted_rep_dynamics(
        rounds_df,
        out_path=out_dir / "FigB7_weighted_accuracy_rep_dt_logit_scale.png",
        attack="dt_logit_scale",
        dt_level=str(args.dt_level),
    )

    print(f"[ok] wrote simple ACC figures to {out_dir}")


if __name__ == "__main__":
    main()
