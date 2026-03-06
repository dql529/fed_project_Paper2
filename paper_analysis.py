from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MAIN_ATTACKS = ["label_flip", "stealth_amp", "dt_logit_scale"]
MAIN_METHODS = ["mean", "median", "trimmed_mean", "weighted"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _ensure_delta_f1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "delta_f1" not in out.columns and {"clean_f1", "polluted_f1"}.issubset(out.columns):
        out["delta_f1"] = pd.to_numeric(out["clean_f1"], errors="coerce") - pd.to_numeric(
            out["polluted_f1"], errors="coerce"
        )
    return out


def _ensure_rep_config(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "rep_config" not in out.columns:
        out["rep_config"] = ""
    methods = out.get("method", pd.Series("", index=out.index)).astype(str)
    out.loc[methods.isin(["weighted", "weighted_full", "weighted_nogate"]), "rep_config"] = "R2,R4"
    out.loc[methods.eq("weighted_r4only"), "rep_config"] = "R4"
    return out


def _mean_std_table(df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        cols = list(group_cols) + [f"{metric}_{suffix}" for metric in metrics for suffix in ("m", "s")] + ["count"]
        return pd.DataFrame(columns=cols)
    rows: list[dict] = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in metrics:
            arr = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_m"] = float(np.mean(arr)) if arr.size else float("nan")
            row[f"{metric}_s"] = float(np.std(arr, ddof=0)) if arr.size else float("nan")
        row["count"] = int(len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_pi_norm(nodes_df: pd.DataFrame) -> pd.DataFrame:
    if nodes_df.empty:
        return nodes_df
    out = nodes_df.copy()
    if "pi_norm" in out.columns:
        return out
    rep = pd.to_numeric(out.get("Rep", np.nan), errors="coerce")
    group_cols = [c for c in ["attack", "dt_level", "mal_nodes", "method", "seed"] if c in out.columns]
    if "round" in out.columns:
        group_cols.append("round")
    if group_cols:
        denom = rep.groupby([out[c] for c in group_cols]).transform("sum")
    else:
        denom = pd.Series(np.repeat(rep.sum(), len(out)), index=out.index)
    out["pi_norm"] = np.where(pd.to_numeric(denom, errors="coerce") > 0, rep / denom, np.nan)
    return out


def _final_node_slice(nodes_df: pd.DataFrame) -> pd.DataFrame:
    if nodes_df.empty or "round" not in nodes_df.columns:
        return nodes_df
    max_round = nodes_df.groupby(["attack", "dt_level", "mal_nodes", "method", "seed"])["round"].transform("max")
    return nodes_df[nodes_df["round"] == max_round].copy()


def _scenario_mask(df: pd.DataFrame, dt: str) -> pd.Series:
    attack = df["attack"].astype(str)
    mal_nodes = pd.to_numeric(df["mal_nodes"], errors="coerce")
    return (
        df["dt_level"].astype(str).eq(dt)
        & (
            ((attack.isin(MAIN_ATTACKS)) & mal_nodes.isin([0, 5]))
            | ((attack.eq("label_flip")) & mal_nodes.eq(3))
        )
    )


def _ablation_mask(df: pd.DataFrame, dt: str) -> pd.Series:
    attack = df["attack"].astype(str)
    mal_nodes = pd.to_numeric(df["mal_nodes"], errors="coerce")
    return (
        df["dt_level"].astype(str).eq(dt)
        & (
            ((attack.isin(MAIN_ATTACKS)) & mal_nodes.eq(5))
            | ((attack.eq("label_flip")) & mal_nodes.eq(3))
        )
    )


def _merge_ablation_runs(primary_dir: Path, extra_dir: Path | None) -> pd.DataFrame:
    frames = []
    primary = _read_csv(primary_dir / "runs.csv")
    if not primary.empty:
        frames.append(primary)
    if extra_dir is not None:
        extra = _read_csv(extra_dir / "runs.csv")
        if not extra.empty:
            frames.append(extra)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["attack", "dt_level", "mal_nodes", "method", "seed", "tau_gate", "ref_size", "audit_size"],
        keep="last",
    )
    return merged


def _cohens_dz(diff: np.ndarray) -> float:
    if diff.size == 0:
        return float("nan")
    if np.allclose(diff, 0.0):
        return 0.0
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if sd <= 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def _paired_stats(
    runs_df: pd.DataFrame,
    *,
    dt: str,
    main_method: str,
    baseline: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for attack, mal_nodes in [("label_flip", 5), ("stealth_amp", 5), ("dt_logit_scale", 5), ("label_flip", 3)]:
        sub = runs_df[
            runs_df["dt_level"].astype(str).eq(dt)
            & runs_df["attack"].astype(str).eq(attack)
            & (pd.to_numeric(runs_df["mal_nodes"], errors="coerce") == mal_nodes)
            & runs_df["method"].astype(str).isin([main_method, baseline])
        ]
        for metric in metrics:
            wide = sub.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
            if main_method not in wide.columns or baseline not in wide.columns:
                rows.append(
                    {
                        "attack": attack,
                        "mal_nodes": mal_nodes,
                        "metric": metric,
                        "n_pairs": 0,
                        "mean_weighted": float("nan"),
                        "mean_median": float("nan"),
                        "mean_diff": float("nan"),
                        "wilcoxon_stat": float("nan"),
                        "p_value": float("nan"),
                        "cohens_dz": float("nan"),
                        "decision_note": "insufficient_pairs",
                    }
                )
                continue
            wide = wide[[main_method, baseline]].dropna()
            diff = (wide[main_method] - wide[baseline]).to_numpy(dtype=float)
            note = "ok"
            stat = float("nan")
            p_value = float("nan")
            if diff.size < 2:
                note = "insufficient_pairs"
            elif np.allclose(diff, 0.0):
                stat = 0.0
                p_value = 1.0
                note = "all_zero_diffs"
            else:
                res = wilcoxon(diff)
                stat = float(res.statistic)
                p_value = float(res.pvalue)
            rows.append(
                {
                    "attack": attack,
                    "mal_nodes": mal_nodes,
                    "metric": metric,
                    "n_pairs": int(diff.size),
                    "mean_weighted": float(wide[main_method].mean()) if diff.size else float("nan"),
                    "mean_median": float(wide[baseline].mean()) if diff.size else float("nan"),
                    "mean_diff": float(diff.mean()) if diff.size else float("nan"),
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "cohens_dz": _cohens_dz(diff),
                    "decision_note": note,
                }
            )
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, default="artifacts/base")
    ap.add_argument("--ablation-dir", type=str, default="artifacts/appendix_ablation/base")
    ap.add_argument("--ablation-extra-dir", type=str, default="artifacts/appendix_ablation_lf3/base")
    ap.add_argument("--tau-dir", type=str, default="artifacts/appendix_taucheck/tau")
    ap.add_argument("--out-dir", type=str, default="artifacts/paper_tables")
    ap.add_argument("--dt", type=str, default="D0")
    ap.add_argument("--main-method", type=str, default="weighted")
    ap.add_argument("--baseline", type=str, default="median")
    ap.add_argument("--node-attack", type=str, default="label_flip")
    ap.add_argument("--node-f", type=int, default=5)
    args = ap.parse_args()

    base_runs = _ensure_rep_config(_ensure_delta_f1(_read_csv(Path(args.base_dir) / "runs.csv")))
    base_nodes = _ensure_rep_config(_ensure_pi_norm(_final_node_slice(_read_csv(Path(args.base_dir) / "nodes.csv"))))
    ablation_runs = _ensure_rep_config(
        _ensure_delta_f1(
            _merge_ablation_runs(
                Path(args.ablation_dir),
                Path(args.ablation_extra_dir) if args.ablation_extra_dir else None,
            )
        )
    )
    tau_runs = _ensure_rep_config(_ensure_delta_f1(_read_csv(Path(args.tau_dir) / "runs.csv")))
    out_dir = Path(args.out_dir)
    dt = str(args.dt).upper()

    main_runs = base_runs[_scenario_mask(base_runs, dt)].copy()
    main_runs = main_runs[main_runs["method"].astype(str).isin(MAIN_METHODS)]
    table_main = _mean_std_table(
        main_runs,
        ["attack", "dt_level", "mal_nodes", "method"],
        ["clean_f1", "polluted_f1", "delta_f1"],
    ).sort_values(["attack", "mal_nodes", "method"])
    _write(table_main, out_dir / "table_main_performance.csv")

    ablation_sel = ablation_runs[_ablation_mask(ablation_runs, dt)].copy()
    ablation_sel = ablation_sel[
        ablation_sel["method"].astype(str).isin([args.main_method, "weighted_r4only"])
    ]
    table_ablation = _mean_std_table(
        ablation_sel,
        ["attack", "dt_level", "mal_nodes", "method", "rep_config"],
        ["clean_f1", "polluted_f1", "delta_f1", "w_mal"],
    ).sort_values(["attack", "mal_nodes", "method"])
    _write(table_ablation, out_dir / "table_ablation_r4only.csv")

    mech_nodes = base_nodes[
        _ablation_mask(base_nodes, dt)
        & base_nodes["method"].astype(str).eq(args.main_method)
    ].copy()
    table_mechanism = _mean_std_table(
        mech_nodes,
        ["attack", "dt_level", "mal_nodes", "is_malicious"],
        ["R4", "Rep", "pi_norm", "passed_gate"],
    ).sort_values(["attack", "mal_nodes", "is_malicious"])
    _write(table_mechanism, out_dir / "table_mechanism.csv")

    table_stats = _paired_stats(
        base_runs,
        dt=dt,
        main_method=args.main_method,
        baseline=args.baseline,
        metrics=["clean_f1", "polluted_f1", "delta_f1"],
    )
    _write(table_stats, out_dir / "table_stats_weighted_vs_median.csv")

    seed_frames = []
    base_seed = main_runs[main_runs["method"].astype(str).isin([args.main_method, args.baseline])].copy()
    if not base_seed.empty:
        base_seed["source"] = "base_compare"
        seed_frames.append(
            base_seed[
                ["source", "attack", "dt_level", "mal_nodes", "method", "seed", "clean_f1", "polluted_f1", "delta_f1", "w_mal"]
            ]
        )
    if not ablation_sel.empty:
        ab_seed = ablation_sel.copy()
        ab_seed["source"] = "ablation"
        seed_frames.append(
            ab_seed[
                ["source", "attack", "dt_level", "mal_nodes", "method", "seed", "clean_f1", "polluted_f1", "delta_f1", "w_mal"]
            ]
        )
    table_seed = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    _write(table_seed, out_dir / "table_seed_results.csv")

    tau_sel = tau_runs[
        tau_runs["dt_level"].astype(str).eq(dt)
        & tau_runs["attack"].astype(str).eq("label_flip")
        & (pd.to_numeric(tau_runs["mal_nodes"], errors="coerce") == 3)
        & tau_runs["method"].astype(str).eq(args.main_method)
        & pd.to_numeric(tau_runs["tau_gate"], errors="coerce").isin([0.6, 0.7])
    ].copy()
    table_tau = _mean_std_table(
        tau_sel,
        ["attack", "dt_level", "mal_nodes", "method", "tau_gate"],
        ["clean_f1", "polluted_f1", "delta_f1", "benign_pass_rate", "malicious_pass_rate", "w_mal"],
    ).sort_values(["tau_gate"])
    _write(table_tau, out_dir / "table_tau_sanity.csv")

    node_plot = mech_nodes[
        mech_nodes["attack"].astype(str).eq(args.node_attack)
        & (pd.to_numeric(mech_nodes["mal_nodes"], errors="coerce") == int(args.node_f))
    ][
        ["attack", "dt_level", "mal_nodes", "method", "seed", "node_id", "is_malicious", "R4", "Rep", "pi_norm", "passed_gate"]
    ].copy()
    _write(node_plot, out_dir / f"node_plot_data_{args.node_attack}_f{int(args.node_f)}.csv")


if __name__ == "__main__":
    main()
