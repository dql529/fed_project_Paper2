from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ATTACKS = ["label_flip", "stealth_amp", "dt_logit_scale"]
METHODS = ["weighted", "median", "mean", "trimmed_mean"]
STAGES = [
    ("E0_base_original", Path("artifacts/base")),
    ("E1_noise_only", Path("artifacts/evolution_noise_only")),
    ("E2_fair_tuned_v1", Path("artifacts/local_R3A3_S3")),
    ("E3_current_fair_mainline", Path("artifacts/current_mainline_b0s4")),
]


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _get_method_row(table_a: pd.DataFrame, attack: str, mal_nodes: int, method: str) -> pd.Series:
    sub = table_a[
        (table_a["attack"].astype(str) == attack)
        & (pd.to_numeric(table_a["mal_nodes"], errors="coerce") == mal_nodes)
        & (table_a["method"].astype(str) == method)
    ]
    if sub.empty:
        raise KeyError((attack, mal_nodes, method))
    return sub.iloc[0]


def _get_mech_rows(table_b: pd.DataFrame, attack: str, mal_nodes: int) -> Tuple[pd.Series, pd.Series]:
    sub = table_b[
        (table_b["attack"].astype(str) == attack)
        & (pd.to_numeric(table_b["mal_nodes"], errors="coerce") == mal_nodes)
        & (table_b["method"].astype(str) == "weighted")
    ]
    benign = sub[pd.to_numeric(sub["is_malicious"], errors="coerce") == 0]
    malicious = sub[pd.to_numeric(sub["is_malicious"], errors="coerce") == 1]
    if benign.empty or malicious.empty:
        raise KeyError((attack, mal_nodes, "weighted_mech"))
    return benign.iloc[0], malicious.iloc[0]


def _row_delta_f1(row: pd.Series) -> float:
    if "delta_f1_m" in row.index:
        return float(row["delta_f1_m"])
    return float(row["clean_f1_m"]) - float(row["polluted_f1_m"])


def _stage_gap_metrics(root: Path, mal_nodes: int = 3) -> pd.DataFrame:
    table_a = _read_csv(root / "tables/table_A_methods.csv")
    rows: List[Dict] = []
    for attack in ATTACKS:
        w = _get_method_row(table_a, attack, mal_nodes, "weighted")
        m = _get_method_row(table_a, attack, mal_nodes, "median")
        rows.append(
            {
                "attack": attack,
                "gap_clean": float(w["clean_f1_m"]) - float(m["clean_f1_m"]),
                "gap_polluted": float(w["polluted_f1_m"]) - float(m["polluted_f1_m"]),
                "w_mal_gap": float(m["w_mal_m"]) - float(w["w_mal_m"]),
            }
        )
    return pd.DataFrame(rows)


def _range_row(factor: str, comparison: str, values: pd.DataFrame) -> Dict:
    return {
        "factor": factor,
        "comparison": comparison,
        "delta_gap_clean": float(values["gap_clean"].max() - values["gap_clean"].min()),
        "delta_gap_polluted": float(values["gap_polluted"].max() - values["gap_polluted"].min()),
        "delta_w_mal_gap": float(values["w_mal_gap"].max() - values["w_mal_gap"].min()),
    }


def _stage_delta_row(factor: str, comparison: str, before_df: pd.DataFrame, after_df: pd.DataFrame) -> Dict:
    before_mean = before_df[["gap_clean", "gap_polluted", "w_mal_gap"]].mean()
    after_mean = after_df[["gap_clean", "gap_polluted", "w_mal_gap"]].mean()
    return {
        "factor": factor,
        "comparison": comparison,
        "delta_gap_clean": float(after_mean["gap_clean"] - before_mean["gap_clean"]),
        "delta_gap_polluted": float(after_mean["gap_polluted"] - before_mean["gap_polluted"]),
        "delta_w_mal_gap": float(after_mean["w_mal_gap"] - before_mean["w_mal_gap"]),
    }


def build_ablation_table(f_roots: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for root in f_roots:
        resolved = _resolve_root(root)
        table_a = _read_csv(resolved / "tables/table_A_methods.csv")
        if table_a.empty:
            continue
        sub = table_a[
            table_a["attack"].astype(str).isin(ATTACKS)
            & table_a["method"].astype(str).isin(METHODS)
        ].copy()
        sub["f"] = pd.to_numeric(sub["mal_nodes"], errors="coerce").astype(int)
        frames.append(
            sub[
                [
                    "attack",
                    "f",
                    "method",
                    "clean_f1_m",
                    "clean_f1_s",
                    "polluted_f1_m",
                    "polluted_f1_s",
                    "delta_f1_m",
                    "w_mal_m",
                    "count",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    method_order = {m: i for i, m in enumerate(METHODS)}
    out["method_order"] = out["method"].map(method_order)
    out = out.sort_values(["attack", "f", "method_order"]).drop(columns=["method_order"])
    return out


def build_evolution_table(stages: Iterable[Tuple[str, Path]]) -> pd.DataFrame:
    rows: List[Dict] = []
    for stage_name, raw_root in stages:
        root = _resolve_root(raw_root)
        table_a = _read_csv(root / "tables/table_A_methods.csv")
        table_b = _read_csv(root / "tables/table_B_weighted_reputation_summary.csv")
        for attack in ATTACKS:
            w = _get_method_row(table_a, attack, 3, "weighted")
            benign, malicious = _get_mech_rows(table_b, attack, 3)
            rows.append(
                {
                    "stage": stage_name,
                    "attack": attack,
                    "clean_f1_m": float(w["clean_f1_m"]),
                    "polluted_f1_m": float(w["polluted_f1_m"]),
                    "delta_f1_m": _row_delta_f1(w),
                    "w_mal_m": float(w["w_mal_m"]),
                    "R4_benign_m": float(benign["R4_m"]),
                    "R4_malicious_m": float(malicious["R4_m"]),
                    "Rep_benign_m": float(benign["Rep_m"]),
                    "Rep_malicious_m": float(malicious["Rep_m"]),
                    "pass_benign_m": float(benign["passed_gate_m"]),
                    "pass_malicious_m": float(malicious["passed_gate_m"]),
                }
            )
    return pd.DataFrame(rows)


def build_factor_table(ablation_df: pd.DataFrame, stages: Iterable[Tuple[str, Path]]) -> pd.DataFrame:
    gap_rows: List[Dict] = []
    pivot = (
        ablation_df[ablation_df["method"].isin(["weighted", "median"])]
        .pivot_table(index=["attack", "f"], columns="method", values=["clean_f1_m", "polluted_f1_m", "w_mal_m"], aggfunc="first")
    )
    for (attack, f_value), row in pivot.iterrows():
        gap_rows.append(
            {
                "attack": attack,
                "f": int(f_value),
                "gap_clean": float(row[("clean_f1_m", "weighted")] - row[("clean_f1_m", "median")]),
                "gap_polluted": float(row[("polluted_f1_m", "weighted")] - row[("polluted_f1_m", "median")]),
                "w_mal_gap": float(row[("w_mal_m", "median")] - row[("w_mal_m", "weighted")]),
            }
        )
    gaps = pd.DataFrame(gap_rows)

    attack_means = gaps.groupby("attack", as_index=False)[["gap_clean", "gap_polluted", "w_mal_gap"]].mean()
    attack_worst = gaps.groupby("attack", as_index=False)[["gap_clean", "gap_polluted", "w_mal_gap"]].min()
    f_means = gaps.groupby("f", as_index=False)[["gap_clean", "gap_polluted", "w_mal_gap"]].mean()
    f_worst = gaps.groupby("f", as_index=False)[["gap_clean", "gap_polluted", "w_mal_gap"]].min()

    rows = [
        _range_row("attack_type", "mean_range_across_attacks", attack_means),
        _range_row("attack_type", "worst_range_across_attacks", attack_worst),
        _range_row("f", "mean_range_across_f", f_means),
        _range_row("f", "worst_range_across_f", f_worst),
    ]

    stage_gap_map = {stage_name: _stage_gap_metrics(_resolve_root(raw_root), mal_nodes=3) for stage_name, raw_root in stages}
    rows.extend(
        [
            _stage_delta_row("suppressive_noise", "E1_minus_E0", stage_gap_map["E0_base_original"], stage_gap_map["E1_noise_only"]),
            _stage_delta_row("reputation_stage", "E2_minus_E1", stage_gap_map["E1_noise_only"], stage_gap_map["E2_fair_tuned_v1"]),
            _stage_delta_row("reputation_stage", "E3_minus_E2", stage_gap_map["E2_fair_tuned_v1"], stage_gap_map["E3_current_fair_mainline"]),
        ]
    )

    out = pd.DataFrame(rows)
    out = out.reindex(
        out.assign(
            _sort1=out["delta_gap_polluted"].abs(),
            _sort2=out["delta_gap_clean"].abs(),
            _sort3=out["delta_w_mal_gap"].abs(),
        )
        .sort_values(["_sort1", "_sort2", "_sort3"], ascending=False)
        .index
    ).reset_index(drop=True)
    out["priority_rank"] = range(1, len(out) + 1)
    return out[["factor", "comparison", "delta_gap_clean", "delta_gap_polluted", "delta_w_mal_gap", "priority_rank"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="artifacts/fair_ablation_tables")
    ap.add_argument("--ablation-root", type=str, default="artifacts/fair_f_ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f_roots = [Path(args.ablation_root) / f"f{f_value}" for f_value in [1, 2, 3, 4, 5]]
    ablation_df = build_ablation_table(f_roots)
    evolution_df = build_evolution_table(STAGES)
    factor_df = build_factor_table(ablation_df, STAGES)

    ablation_df.to_csv(out_dir / "table_ablation_f_suppressive.csv", index=False)
    evolution_df.to_csv(out_dir / "table_evolution_fair_mainline.csv", index=False)
    factor_df.to_csv(out_dir / "table_factor_effects.csv", index=False)

    print(f"[ok] wrote {out_dir / 'table_ablation_f_suppressive.csv'}")
    print(f"[ok] wrote {out_dir / 'table_evolution_fair_mainline.csv'}")
    print(f"[ok] wrote {out_dir / 'table_factor_effects.csv'}")


if __name__ == "__main__":
    main()
