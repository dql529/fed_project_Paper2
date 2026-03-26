from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from dt_r4.plotting import plot_acc_summary_vs_f, plot_metric_vs_round_grid


ATTACKS = ["label_flip", "stealth_amp", "dt_logit_scale"]
ROUND_ATTACKS = ["label_flip", "dt_logit_scale"]
METHODS = ["weighted", "median", "mean", "trimmed_mean"]
F_VALUES = [1, 2, 3, 4, 5]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ablation-root",
        type=str,
        default="artifacts/99_workspace_archive/fair_f_ablation",
    )
    ap.add_argument(
        "--round-root",
        type=str,
        default="artifacts/99_workspace_archive/acc_rounds_f_ablation",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/01_appendix_upload/figures",
    )
    ap.add_argument("--dt-level", type=str, default="D0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _concat_csvs(_collect_ablation_csvs(Path(args.ablation_root), "summary.csv"))
    rounds_df = _concat_csvs(_collect_ablation_csvs(Path(args.round_root), "rounds.csv"))

    plot_acc_summary_vs_f(
        summary_df,
        attacks=ATTACKS,
        methods=METHODS,
        dt_level=str(args.dt_level),
        mal_nodes=F_VALUES,
        out_path=out_dir / "FigA1_acc_summary_vs_f.png",
        top_metric="clean_acc",
        bottom_metric="polluted_acc",
    )

    plot_metric_vs_round_grid(
        rounds_df,
        dt_level=str(args.dt_level),
        attack="label_flip",
        mal_nodes=F_VALUES,
        methods=METHODS,
        metric="round_clean_acc",
        out_path=out_dir / "FigA2_round_clean_acc_label_flip_f1to5.png",
    )
    plot_metric_vs_round_grid(
        rounds_df,
        dt_level=str(args.dt_level),
        attack="dt_logit_scale",
        mal_nodes=F_VALUES,
        methods=METHODS,
        metric="round_clean_acc",
        out_path=out_dir / "FigA3_round_clean_acc_dt_logit_scale_f1to5.png",
    )
    plot_metric_vs_round_grid(
        rounds_df,
        dt_level=str(args.dt_level),
        attack="label_flip",
        mal_nodes=F_VALUES,
        methods=METHODS,
        metric="round_polluted_acc",
        out_path=out_dir / "FigA4_round_deploy_acc_label_flip_f1to5.png",
    )
    plot_metric_vs_round_grid(
        rounds_df,
        dt_level=str(args.dt_level),
        attack="dt_logit_scale",
        mal_nodes=F_VALUES,
        methods=METHODS,
        metric="round_polluted_acc",
        out_path=out_dir / "FigA5_round_deploy_acc_dt_logit_scale_f1to5.png",
    )

    print(f"[ok] wrote ACC appendix figures to {out_dir}")


if __name__ == "__main__":
    main()
