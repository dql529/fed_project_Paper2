"""
paper_figs.py

Generate the 4 paper figures from CSV artifacts.

Expected inputs:
  - summary.csv : grouped mean/std (from r4_agg_minitest.py)
  - rounds.csv  : per-round W_mal logs (from r4_agg_minitest.py; weighted only)
  - nodes.csv   : per-node final rep/R2/R3/R4 logs (from r4_agg_minitest.py; weighted only)
  - ablation_summary.csv (optional) : grouped mean/std for ablation runs

Example:
  python paper_figs.py --dt D1

If you have ablation files:
  python r4_agg_minitest.py --attack-modes dt_logit_scale --dt-levels D1 --mal-nodes 0,3,5 \\
    --methods weighted_full,weighted_r4only,weighted_r2only,weighted_nogate \\
    --out-runs ablation_runs.csv --out-summary ablation_summary.csv
  python paper_figs.py --dt D1 --ablation-summary ablation_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dt_r4.plotting import (
    load_nodes_csv,
    load_rounds_csv,
    load_summary_csv,
    plot_ablation_multiattack,
    plot_clean_holdout_vs_f,
    plot_r4_distribution,
    plot_wmal_vs_round,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--figs",
        type=str,
        default="1,2,3,4",
        help="Comma list of figure indices to generate (e.g., 1,2,3,4 or 1,4)",
    )
    ap.add_argument("--dt", type=str, default="D1", help="DT level to plot (D0/D1/D2)")
    ap.add_argument("--summary", type=str, default="summary.csv")
    ap.add_argument("--rounds", type=str, default="rounds.csv")
    ap.add_argument("--nodes", type=str, default="nodes.csv")
    ap.add_argument("--ablation-summary", type=str, default=None)
    ap.add_argument(
        "--ablation-attacks",
        type=str,
        default="label_flip,dt_logit_scale",
        help="Comma list of attacks to include in Fig.4 ablation",
    )
    ap.add_argument("--out-dir", type=str, default="paper_figs")

    ap.add_argument(
        "--mal-nodes",
        type=str,
        default="0,3,5",
        help="Comma list for x-axis f (malicious clients)",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="weighted,mean,median,trimmed_mean",
        help="Comma list of baseline methods for Fig.1",
    )
    ap.add_argument("--f-fig2", type=int, default=5, help="f (mal nodes) for Fig.2")
    ap.add_argument("--f-fig3", type=int, default=5, help="f (mal nodes) for Fig.3")
    args = ap.parse_args()

    dt = str(args.dt).strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = {int(x.strip()) for x in args.figs.split(",") if x.strip()}
    mal_nodes = [int(x.strip()) for x in args.mal_nodes.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Fig.1: clean holdout macro-F1 vs f (3 attacks; 3 subplots)
    if 1 in figs:
        summary_df = load_summary_csv(args.summary)
        plot_clean_holdout_vs_f(
            summary_df,
            attacks=["label_flip", "stealth_amp", "dt_logit_scale"],
            methods=methods,
            dt_level=dt,
            mal_nodes=mal_nodes,
            out_path=out_dir / f"Fig1_cleanF1_vs_f_dt{dt}.png",
            metric="clean_f1",
            label_flip_level="L1",
        )

    # Fig.2: W_mal vs round (focus f=5) for dt_logit_scale + label_flip
    if 2 in figs:
        rounds_df = load_rounds_csv(args.rounds)
        plot_wmal_vs_round(
            rounds_df,
            dt_level=dt,
            mal_nodes=int(args.f_fig2),
            attacks=["label_flip", "dt_logit_scale"],
            out_path=out_dir / f"Fig2_Wmal_vs_round_dt{dt}_f{int(args.f_fig2)}.png",
            method="weighted",
            label_flip_level="L1",
        )

    # Fig.3: R4/KL distribution (dt_logit_scale) benign vs malicious
    if 3 in figs:
        nodes_df = load_nodes_csv(args.nodes)
        plot_r4_distribution(
            nodes_df,
            attack="dt_logit_scale",
            dt_level=dt,
            mal_nodes=int(args.f_fig3),
            out_path=out_dir / f"Fig3_R4_distribution_dt{dt}_f{int(args.f_fig3)}.png",
            method="weighted",
            label_flip_level="L1",
        )

    # Fig.4: Ablation (dt_logit_scale) if the file is provided
    if 4 in figs and args.ablation_summary:
        ab_sum = load_summary_csv(args.ablation_summary)
        attacks = [x.strip() for x in args.ablation_attacks.split(",") if x.strip()]
        plot_ablation_multiattack(
            ab_sum,
            dt_level=dt,
            attacks=attacks,
            out_path=out_dir / f"Fig4_ablation_dt{dt}.png",
            mal_nodes=mal_nodes,
            metric="polluted_f1",
        )


if __name__ == "__main__":
    main()
