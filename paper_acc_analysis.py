from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


ATTACKS = ["label_flip", "stealth_amp", "dt_logit_scale"]
METHODS = ["weighted", "median", "mean", "trimmed_mean"]


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


def _round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out.loc[:, numeric_cols] = out.loc[:, numeric_cols].round(digits)
    return out


def build_mainline_acc_table(root: Path, *, mal_nodes: int = 3) -> pd.DataFrame:
    table_a = _read_csv(_resolve_root(root) / "tables/table_A_methods.csv")
    sub = table_a[
        table_a["attack"].astype(str).isin(ATTACKS)
        & (pd.to_numeric(table_a["mal_nodes"], errors="coerce") == int(mal_nodes))
        & table_a["method"].astype(str).isin(METHODS)
    ].copy()
    out = sub[
        [
            "attack",
            "method",
            "clean_acc_m",
            "clean_acc_s",
            "polluted_acc_m",
            "polluted_acc_s",
            "count",
        ]
    ].rename(
        columns={
            "clean_acc_m": "reference_acc_m",
            "clean_acc_s": "reference_acc_s",
            "polluted_acc_m": "deploy_acc_m",
            "polluted_acc_s": "deploy_acc_s",
        }
    )
    method_order = {name: idx for idx, name in enumerate(METHODS)}
    out["method_order"] = out["method"].map(method_order)
    out = out.sort_values(["attack", "method_order"]).drop(columns=["method_order"])
    return _round_numeric(out)


def _read_ablation_tables(ablation_root: Path) -> Iterable[pd.DataFrame]:
    for f_value in [1, 2, 3, 4, 5]:
        root = _resolve_root(ablation_root / f"f{f_value}")
        table_a = _read_csv(root / "tables/table_A_methods.csv")
        if table_a.empty:
            continue
        sub = table_a[
            table_a["attack"].astype(str).isin(ATTACKS)
            & table_a["method"].astype(str).isin(["weighted", "median"])
        ].copy()
        sub["f"] = pd.to_numeric(sub["mal_nodes"], errors="coerce").astype(int)
        yield sub


def build_acc_ablation_table(ablation_root: Path) -> pd.DataFrame:
    frames = list(_read_ablation_tables(ablation_root))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(
        index=["attack", "f"],
        columns="method",
        values=["clean_acc_m", "polluted_acc_m"],
        aggfunc="first",
    )
    rows: List[Dict] = []
    for (attack, f_value), row in pivot.iterrows():
        rows.append(
            {
                "attack": str(attack),
                "f": int(f_value),
                "weighted_reference_acc": float(row[("clean_acc_m", "weighted")]),
                "median_reference_acc": float(row[("clean_acc_m", "median")]),
                "gap_reference_acc": float(
                    row[("clean_acc_m", "weighted")] - row[("clean_acc_m", "median")]
                ),
                "weighted_deploy_acc": float(row[("polluted_acc_m", "weighted")]),
                "median_deploy_acc": float(row[("polluted_acc_m", "median")]),
                "gap_deploy_acc": float(
                    row[("polluted_acc_m", "weighted")] - row[("polluted_acc_m", "median")]
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(["attack", "f"]).reset_index(drop=True)
    return _round_numeric(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--current-root",
        type=str,
        default="artifacts/99_workspace_archive/current_mainline_b0s4",
    )
    ap.add_argument(
        "--ablation-root",
        type=str,
        default="artifacts/99_workspace_archive/fair_f_ablation",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/02_writing_records/appendix",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mainline_acc = build_mainline_acc_table(Path(args.current_root), mal_nodes=3)
    ablation_acc = build_acc_ablation_table(Path(args.ablation_root))

    mainline_acc.to_csv(out_dir / "table_acc_mainline_f3.csv", index=False)
    ablation_acc.to_csv(out_dir / "table_acc_ablation_compact.csv", index=False)

    print(f"[ok] wrote {out_dir / 'table_acc_mainline_f3.csv'}")
    print(f"[ok] wrote {out_dir / 'table_acc_ablation_compact.csv'}")


if __name__ == "__main__":
    main()
