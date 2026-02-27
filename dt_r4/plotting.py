"""
dt_r4/plotting.py

Plot helpers that work purely from CSV artifacts produced by r4_agg_minitest.py:
  - runs.csv   : per (attack, dt_level, mal_nodes, method, seed) raw metrics
  - summary.csv: grouped mean/std/count metrics

This lets you re-generate figures without re-running training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PlotSpec:
    attacks: List[str]
    methods: List[str]
    mal_nodes: List[int]
    dt_levels: List[str]
    label_flip_level: str = "L1"
    num_nodes: int = 10


def _as_list_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return items or None


def _as_list_int_csv(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    items = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    return items or None


def load_runs_csv(path: str | Path):
    import pandas as pd

    df = pd.read_csv(path)
    # Normalize types/empties for stable filtering.
    for col in ("attack", "level", "dt_level", "method"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "mal_nodes" in df.columns:
        df["mal_nodes"] = df["mal_nodes"].astype(int)
    if "seed" in df.columns:
        df["seed"] = df["seed"].astype(int)
    return df


def load_summary_csv(path: str | Path):
    import pandas as pd

    df = pd.read_csv(path)
    for col in ("attack", "level", "dt_level", "method"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "mal_nodes" in df.columns:
        df["mal_nodes"] = df["mal_nodes"].astype(int)
    if "count" in df.columns:
        df["count"] = df["count"].astype(int)
    return df


def summarize_runs(runs_df, ddof: int = 0):
    """
    Build a summary DataFrame identical in schema to summary.csv.

    Note: r4_agg_minitest.py uses numpy.std(ddof=0) by default; we keep that
    default here for consistency.
    """
    import pandas as pd

    group_cols = ["attack", "level", "dt_level", "mal_nodes", "method"]
    metric_cols = ["polluted_acc", "polluted_f1", "clean_acc", "clean_f1", "w_mal"]

    rows = []
    for key, g in runs_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["count"] = int(len(g))

        for base in metric_cols:
            vals = g[base].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                m = float("nan")
                s = float("nan")
            else:
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=ddof))
            row[f"{base}_m"] = m
            row[f"{base}_s"] = s

        rows.append(row)

    out = pd.DataFrame(rows)
    # Match the exact column names used by the project summary.csv.
    out = out.rename(
        columns={
            "polluted_acc_m": "polluted_acc_m",
            "polluted_acc_s": "polluted_acc_s",
            "polluted_f1_m": "polluted_f1_m",
            "polluted_f1_s": "polluted_f1_s",
            "clean_acc_m": "clean_acc_m",
            "clean_acc_s": "clean_acc_s",
            "clean_f1_m": "clean_f1_m",
            "clean_f1_s": "clean_f1_s",
            "w_mal_m": "w_mal_m",
            "w_mal_s": "w_mal_s",
        }
    )
    # Stable sort for readability / deterministic plots.
    out = out.sort_values(group_cols).reset_index(drop=True)
    return out


def _filter_attack_level(df, attack: str, label_flip_level: str):
    if "level" not in df.columns:
        return df
    if attack == "label_flip":
        return df[df["level"] == str(label_flip_level)]
    # Non-label-flip attacks use an empty level key in our CSVs.
    return df[(df["level"] == "") | df["level"].isna()]


def plot_figA_polluted(
    summary_df,
    *,
    spec: PlotSpec,
    dt_level: str,
    out_path: str | Path,
    metric: str = "polluted_f1",
):
    """
    Figure A: y=polluted metric (macro-F1 by default), x=malicious ratio,
    facets by attack, curves by method, error bars from std.
    """
    import matplotlib.pyplot as plt

    metric = str(metric).strip().lower()
    if metric not in {"polluted_f1", "polluted_acc"}:
        raise ValueError(f"Unsupported metric for FigA: {metric}")

    m_col = f"{metric}_m"
    s_col = f"{metric}_s"

    df_dt = summary_df[summary_df["dt_level"] == str(dt_level)]

    fig, axes = plt.subplots(
        1, len(spec.attacks), figsize=(5 * len(spec.attacks), 4), sharey=True
    )
    if len(spec.attacks) == 1:
        axes = [axes]

    x = [mal / float(spec.num_nodes) for mal in spec.mal_nodes]

    for ax, attack in zip(axes, spec.attacks):
        df_a = df_dt[df_dt["attack"] == str(attack)]
        df_a = _filter_attack_level(df_a, attack=str(attack), label_flip_level=spec.label_flip_level)

        for method in spec.methods:
            ys, yerr = [], []
            df_m = df_a[df_a["method"] == str(method)]
            for mal in spec.mal_nodes:
                hit = df_m[df_m["mal_nodes"] == int(mal)]
                if len(hit) == 0:
                    ys.append(np.nan)
                    yerr.append(np.nan)
                else:
                    ys.append(float(hit.iloc[0][m_col]))
                    yerr.append(float(hit.iloc[0][s_col]))
            ax.errorbar(x, ys, yerr=yerr, marker="o", capsize=3, label=method)

        ax.set_title(f"{attack} (dt={dt_level})")
        ax.set_xlabel("malicious ratio")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("polluted macro-F1" if metric == "polluted_f1" else "polluted accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(spec.methods))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path)
    plt.close(fig)


def plot_figB_wmal(
    summary_df,
    *,
    spec: PlotSpec,
    dt_level: str,
    out_path: str | Path,
    method: str = "weighted",
):
    """
    Figure B: y=W_mal_total (malicious weight mass) for the rep-based method.
    """
    import matplotlib.pyplot as plt

    df_dt = summary_df[summary_df["dt_level"] == str(dt_level)]

    fig, axes = plt.subplots(
        1, len(spec.attacks), figsize=(5 * len(spec.attacks), 4), sharey=True
    )
    if len(spec.attacks) == 1:
        axes = [axes]

    x = [mal / float(spec.num_nodes) for mal in spec.mal_nodes]

    for ax, attack in zip(axes, spec.attacks):
        df_a = df_dt[df_dt["attack"] == str(attack)]
        df_a = _filter_attack_level(df_a, attack=str(attack), label_flip_level=spec.label_flip_level)
        df_m = df_a[df_a["method"] == str(method)]

        ys, yerr = [], []
        for mal in spec.mal_nodes:
            hit = df_m[df_m["mal_nodes"] == int(mal)]
            if len(hit) == 0:
                ys.append(np.nan)
                yerr.append(np.nan)
            else:
                ys.append(float(hit.iloc[0]["w_mal_m"]))
                yerr.append(float(hit.iloc[0]["w_mal_s"]))

        ax.errorbar(x, ys, yerr=yerr, marker="o", capsize=3, label=method)
        ax.set_title(f"{attack} (dt={dt_level})")
        ax.set_xlabel("malicious ratio")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("W_mal_total")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path)
    plt.close(fig)


def make_plots_from_csv(
    *,
    runs_csv: str | Path = "runs.csv",
    summary_csv: str | Path = "summary.csv",
    out_dir: str | Path = ".",
    attacks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    mal_nodes: Optional[Sequence[int]] = None,
    dt_levels: Optional[Sequence[str]] = None,
    label_flip_level: str = "L1",
    num_nodes: int = 10,
    metric: str = "polluted_f1",
):
    """
    Load CSVs and generate:
      - plotA_pollutedF1_dt{D0/D1/D2}.png (or pollutedAcc)
      - plotB_wmal_dt{D0/D1/D2}.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_path = Path(runs_csv)
    summary_path = Path(summary_csv)

    if summary_path.exists():
        summary_df = load_summary_csv(summary_path)
    else:
        runs_df = load_runs_csv(runs_path)
        summary_df = summarize_runs(runs_df, ddof=0)
        summary_df.to_csv(summary_path, index=False)

    # Defaults from data if not provided.
    if attacks is None:
        attacks = sorted({str(x) for x in summary_df["attack"].unique() if str(x)})
    if methods is None:
        methods = ["weighted", "mean", "median", "trimmed_mean"]
    if mal_nodes is None:
        mal_nodes = sorted({int(x) for x in summary_df["mal_nodes"].unique()})
    if dt_levels is None:
        dt_levels = sorted({str(x) for x in summary_df["dt_level"].unique() if str(x)})

    spec = PlotSpec(
        attacks=list(attacks),
        methods=list(methods),
        mal_nodes=list(mal_nodes),
        dt_levels=list(dt_levels),
        label_flip_level=str(label_flip_level),
        num_nodes=int(num_nodes),
    )

    for dt in spec.dt_levels:
        metric_tag = "pollutedF1" if metric == "polluted_f1" else "pollutedAcc"
        out_a = out_dir / f"plotA_{metric_tag}_dt{dt}.png"
        plot_figA_polluted(summary_df, spec=spec, dt_level=dt, out_path=out_a, metric=metric)

        out_b = out_dir / f"plotB_wmal_dt{dt}.png"
        plot_figB_wmal(summary_df, spec=spec, dt_level=dt, out_path=out_b)


def plot_clean_holdout_vs_f(
    summary_df,
    *,
    attacks: Sequence[str],
    methods: Sequence[str],
    dt_level: str,
    mal_nodes: Sequence[int],
    out_path: str | Path,
    label_flip_level: str = "L1",
    metric: str = "clean_f1",
):
    """
    Fig.1: Clean holdout performance vs f (malicious count).

    This uses `clean_f1_*` (macro-F1 on GLOBAL_REF_CSV) or `clean_acc_*` from summary.csv.
    """
    import matplotlib.pyplot as plt

    metric = str(metric).strip().lower()
    if metric not in {"clean_f1", "clean_acc"}:
        raise ValueError(f"Unsupported metric for clean holdout plot: {metric}")

    m_col = f"{metric}_m"
    s_col = f"{metric}_s"

    df_dt = summary_df[summary_df["dt_level"] == str(dt_level)]

    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4), sharey=True)
    if len(attacks) == 1:
        axes = [axes]

    x = [int(f) for f in mal_nodes]

    for ax, attack in zip(axes, attacks):
        df_a = df_dt[df_dt["attack"] == str(attack)]
        df_a = _filter_attack_level(df_a, attack=str(attack), label_flip_level=str(label_flip_level))

        for method in methods:
            ys, yerr = [], []
            df_m = df_a[df_a["method"] == str(method)]
            for f in x:
                hit = df_m[df_m["mal_nodes"] == int(f)]
                if len(hit) == 0:
                    ys.append(np.nan)
                    yerr.append(np.nan)
                else:
                    ys.append(float(hit.iloc[0][m_col]))
                    yerr.append(float(hit.iloc[0][s_col]))
            ax.errorbar(x, ys, yerr=yerr, marker="o", capsize=3, label=method)

        ax.set_title(f"{attack} (dt={dt_level})")
        ax.set_xlabel("f (malicious clients)")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Clean holdout macro-F1" if metric == "clean_f1" else "Clean holdout accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path)
    plt.close(fig)


def load_rounds_csv(path: str | Path):
    import pandas as pd

    df = pd.read_csv(path)
    for col in ("attack", "level", "dt_level", "method"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ("mal_nodes", "seed", "round"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def load_nodes_csv(path: str | Path):
    import pandas as pd

    df = pd.read_csv(path)
    for col in ("attack", "level", "dt_level", "method"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ("mal_nodes", "seed", "round", "node_id", "is_malicious"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def plot_wmal_vs_round(
    rounds_df,
    *,
    dt_level: str,
    mal_nodes: int,
    attacks: Sequence[str],
    out_path: str | Path,
    method: str = "weighted",
    label_flip_level: str = "L1",
):
    """
    Fig.2: W_mal_total vs round (mean±std across seeds).
    """
    import matplotlib.pyplot as plt

    df = rounds_df[(rounds_df["dt_level"] == str(dt_level)) & (rounds_df["mal_nodes"] == int(mal_nodes))]
    df = df[df["method"] == str(method)]

    fig, axes = plt.subplots(1, len(attacks), figsize=(5 * len(attacks), 4), sharey=True)
    if len(attacks) == 1:
        axes = [axes]

    for ax, attack in zip(axes, attacks):
        dfa = df[df["attack"] == str(attack)]
        dfa = _filter_attack_level(dfa, attack=str(attack), label_flip_level=str(label_flip_level))

        if len(dfa) == 0:
            ax.set_title(f"{attack} (no data)")
            ax.grid(True, linestyle="--", alpha=0.4)
            continue

        g = dfa.groupby("round")["w_mal"]
        x = g.mean().index.to_numpy(dtype=int)
        y = g.mean().to_numpy(dtype=float)
        ystd = g.std(ddof=0).to_numpy(dtype=float)

        ax.plot(x, y, marker="o", label="mean")
        ax.fill_between(x, y - ystd, y + ystd, alpha=0.2, label="±std")
        ax.set_title(f"{attack} (dt={dt_level}, f={mal_nodes})")
        ax.set_xlabel("round")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("W_mal_total")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_r4_distribution(
    nodes_df,
    *,
    attack: str,
    dt_level: str,
    mal_nodes: int,
    out_path: str | Path,
    method: str = "weighted",
    label_flip_level: str = "L1",
):
    """
    Fig.3: R4 (and divergence) distribution for benign vs malicious.

    Note on "n_b/n_m":
    - Each point is one client score for one seed and one round present in nodes.csv.
      (In our default minitest runs, nodes.csv logs only the last round, so points
      are client-level scores at that final round pooled across seeds.)
    """
    import matplotlib.pyplot as plt
    import dt_r4.config as C

    df = nodes_df[
        (nodes_df["attack"] == str(attack))
        & (nodes_df["dt_level"] == str(dt_level))
        & (nodes_df["mal_nodes"] == int(mal_nodes))
        & (nodes_df["method"] == str(method))
    ]
    df = _filter_attack_level(df, attack=str(attack), label_flip_level=str(label_flip_level))

    benign = df[df["is_malicious"] == 0]
    malicious = df[df["is_malicious"] == 1]

    r4_b = benign["R4"].dropna().to_numpy(dtype=float)
    r4_m = malicious["R4"].dropna().to_numpy(dtype=float)
    kl_b = benign["r4_kl"].dropna().to_numpy(dtype=float)
    kl_m = malicious["r4_kl"].dropna().to_numpy(dtype=float)

    if r4_b.size == 0 or r4_m.size == 0:
        raise ValueError(
            "Fig.3 requires both benign and malicious samples. "
            f"Got n_benign={r4_b.size}, n_malicious={r4_m.size} after filtering "
            f"(attack={attack}, dt={dt_level}, f={mal_nodes}, method={method})."
        )

    # Prefer the tau logged in nodes.csv if present/consistent; otherwise fall back to config.
    tau = float(getattr(C, "R4_GATE_TAU", 0.5))
    if "tau_gate" in df.columns:
        uniq = df["tau_gate"].dropna().unique()
        if len(uniq) == 1:
            tau = float(uniq[0])

    pass_b = float(np.mean(r4_b >= tau)) if r4_b.size else float("nan")
    pass_m = float(np.mean(r4_m >= tau)) if r4_m.size else float("nan")

    seeds = sorted({int(s) for s in df["seed"].unique()}) if "seed" in df.columns else []
    rounds = sorted({int(x) for x in df["round"].unique()}) if "round" in df.columns else []
    round_str = f"round={rounds[0]}" if len(rounds) == 1 else f"rounds={len(rounds)}"
    seed_str = f"seeds={len(seeds)}" if seeds else ""
    unit_str = ", ".join([x for x in [seed_str, round_str] if x])

    def _violin(ax, data, labels, ylabel: str, title: str, *, colors: List[str]):
        pos = list(range(1, len(data) + 1))
        parts = ax.violinplot(
            data, positions=pos, showmeans=False, showmedians=True, showextrema=False
        )
        for i, body in enumerate(parts.get("bodies", [])):
            c = colors[i] if i < len(colors) else "#4C72B0"
            body.set_facecolor(c)
            body.set_edgecolor(c)
            body.set_alpha(0.25)

        # jittered points
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            x = rng.normal(i, 0.06, size=len(vals))
            ax.scatter(x, vals, s=10, alpha=0.35, color=colors[i - 1])

        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _violin(
        axes[0],
        [r4_b, r4_m],
        labels=["benign", "malicious"],
        ylabel="R4 score (higher is better)",
        title=f"R4 separation ({attack}, dt={dt_level}, f={mal_nodes})",
        colors=["#4C72B0", "#DD8452"],
    )
    axes[0].axhline(tau, linestyle="--", color="black", linewidth=1.0, alpha=0.7)
    # Keep the figure clean (paper style): only the threshold line + short label.
    axes[0].text(
        0.98,
        tau,
        rf"$\tau_{{gate}}={tau:.2f}$",
        transform=axes[0].get_yaxis_transform(),
        va="bottom",
        ha="right",
        fontsize=10,
        color="black",
    )

    _violin(
        axes[1],
        [kl_b, kl_m],
        labels=["benign", "malicious"],
        ylabel="KL(p_twin || p_i) (lower is better)",
        title=f"KL(p_twin || p_i) ({attack}, dt={dt_level}, f={mal_nodes})",
        colors=["#4C72B0", "#DD8452"],
    )

    # Tighten y-lims to the informative range (with a small margin), without clipping points.
    def _tight_ylim(ax, vals: np.ndarray, *, pad_frac: float = 0.08):
        if vals.size == 0:
            return
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        span = max(1e-12, hi - lo)
        pad = pad_frac * span
        ax.set_ylim(lo - pad, hi + pad)

    _tight_ylim(axes[0], np.concatenate([r4_b, r4_m]))
    _tight_ylim(axes[1], np.concatenate([kl_b, kl_m]))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    # Write a companion caption file with the "statistical unit" and gate pass rates.
    out_path = Path(out_path)
    cap_path = out_path.with_suffix("")  # drop .png
    cap_path = cap_path.parent / (cap_path.name + "_caption.txt")

    # Decide what "one point" means based on how many rounds are present in the filtered df.
    if len(rounds) == 1:
        unit_desc = (
            f"Each dot is one client-level score at the final round (r={rounds[0]}), "
            f"pooled across {len(seeds)} seeds."
        )
    else:
        unit_desc = (
            "Each dot is one client-level score computed per round, pooled across "
            f"{len(seeds)} seeds and {len(rounds)} rounds."
        )

    caption_plain = (
        f"Fig. 3. Separation of semantic consistency between benign and malicious clients under "
        f"attack={attack} at f={mal_nodes} and DT fidelity {dt_level}. "
        f"Left: R4 scores (higher is better) with semantic gate threshold tau_gate={tau:.2f}. "
        f"Right: KL divergence KL(p_twin || p_i) on the server reference set (lower is better). "
        f"{unit_desc} n_benign={r4_b.size}, n_malicious={r4_m.size}. "
        f"Gate pass rate P(R4>=tau_gate)={pass_b:.2f} (benign) vs {pass_m:.2f} (malicious)."
    )

    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(caption_plain + "\n")


def plot_ablation_dt_logit_scale(
    summary_df,
    *,
    dt_level: str,
    out_path: str | Path,
    mal_nodes: Sequence[int] = (0, 3, 5),
    attack: str = "dt_logit_scale",
    methods: Sequence[str] = ("weighted_full", "weighted_r4only", "weighted_r2only", "weighted_nogate"),
    metric: str = "polluted_f1",
):
    """
    Fig.4: Ablation curves under dt_logit_scale (F1 vs f).
    """
    import matplotlib.pyplot as plt

    metric = str(metric).strip().lower()
    if metric not in {"polluted_f1", "polluted_acc"}:
        raise ValueError(f"Unsupported metric for ablation plot: {metric}")

    m_col = f"{metric}_m"
    s_col = f"{metric}_s"

    df = summary_df[(summary_df["attack"] == str(attack)) & (summary_df["dt_level"] == str(dt_level))]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = [int(f) for f in mal_nodes]
    for m in methods:
        df_m = df[df["method"] == str(m)]
        ys, yerr = [], []
        for f in x:
            hit = df_m[df_m["mal_nodes"] == int(f)]
            if len(hit) == 0:
                ys.append(np.nan)
                yerr.append(np.nan)
            else:
                ys.append(float(hit.iloc[0][m_col]))
                yerr.append(float(hit.iloc[0][s_col]))
        ax.errorbar(x, ys, yerr=yerr, marker="o", capsize=3, label=m)

    ax.set_title(f"Ablation under {attack} (dt={dt_level})")
    ax.set_xlabel("f (malicious clients)")
    ax.set_ylabel("polluted macro-F1" if metric == "polluted_f1" else "polluted accuracy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation_multiattack(
    summary_df,
    *,
    dt_level: str,
    attacks: Sequence[str],
    out_path: str | Path,
    mal_nodes: Sequence[int] = (0, 3, 5),
    methods: Sequence[str] = (
        "weighted_full",
        "weighted_r4only",
        "weighted_r2only",
        "weighted_nogate",
    ),
    metric: str = "polluted_f1",
    label_flip_level: str = "L1",
):
    """
    Fig.4 (extended): ablation curves under multiple attacks.

    Each subplot is one attack; y is polluted macro-F1 (or polluted acc).
    """
    import matplotlib.pyplot as plt

    metric = str(metric).strip().lower()
    if metric not in {"polluted_f1", "polluted_acc"}:
        raise ValueError(f"Unsupported metric for ablation plot: {metric}")

    m_col = f"{metric}_m"
    s_col = f"{metric}_s"

    df_dt = summary_df[summary_df["dt_level"] == str(dt_level)]

    fig, axes = plt.subplots(
        1, len(attacks), figsize=(5 * len(attacks), 4), sharey=True
    )
    if len(attacks) == 1:
        axes = [axes]

    x = [int(f) for f in mal_nodes]

    for ax, attack in zip(axes, attacks):
        df_a = df_dt[df_dt["attack"] == str(attack)]
        df_a = _filter_attack_level(
            df_a, attack=str(attack), label_flip_level=str(label_flip_level)
        )

        for m in methods:
            df_m = df_a[df_a["method"] == str(m)]
            ys, yerr = [], []
            for f in x:
                hit = df_m[df_m["mal_nodes"] == int(f)]
                if len(hit) == 0:
                    ys.append(np.nan)
                    yerr.append(np.nan)
                else:
                    ys.append(float(hit.iloc[0][m_col]))
                    yerr.append(float(hit.iloc[0][s_col]))
            ax.errorbar(x, ys, yerr=yerr, marker="o", capsize=3, label=m)

        ax.set_title(f"{attack} (dt={dt_level})")
        ax.set_xlabel("f (malicious clients)")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("polluted macro-F1" if metric == "polluted_f1" else "polluted accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(methods), 4))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path)
    plt.close(fig)
