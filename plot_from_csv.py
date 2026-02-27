"""
plot_from_csv.py

Standalone plotting entrypoint.

Usage (default paths):
  python plot_from_csv.py

Custom:
  python plot_from_csv.py --runs runs.csv --summary summary.csv --out-dir plots
"""

from __future__ import annotations

import argparse

import dt_r4.config as C
from dt_r4.plotting import make_plots_from_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="runs.csv", help="Path to runs.csv")
    ap.add_argument(
        "--summary", type=str, default="summary.csv", help="Path to summary.csv"
    )
    ap.add_argument("--out-dir", type=str, default=".", help="Output directory")

    ap.add_argument(
        "--attacks",
        type=str,
        default=None,
        help="Comma list of attacks to plot (default: from CSV)",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="weighted,mean,median,trimmed_mean",
        help="Comma list of methods to plot in FigA",
    )
    ap.add_argument(
        "--mal-nodes",
        type=str,
        default=None,
        help="Comma list of mal_nodes to plot (default: from CSV)",
    )
    ap.add_argument(
        "--dt-levels",
        type=str,
        default=None,
        help="Comma list of dt levels to plot (default: from CSV)",
    )
    ap.add_argument(
        "--label-flip-level",
        type=str,
        default="L1",
        help="Which label_flip level to plot when multiple exist",
    )
    ap.add_argument(
        "--num-nodes",
        type=int,
        default=int(getattr(C, "NUM_NODES", 10)),
        help="Total nodes (for malicious ratio on x-axis)",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="polluted_f1",
        choices=["polluted_f1", "polluted_acc"],
        help="FigA y-axis metric",
    )
    args = ap.parse_args()

    attacks = (
        [x.strip() for x in args.attacks.split(",") if x.strip()]
        if args.attacks
        else None
    )
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    mal_nodes = (
        [int(x.strip()) for x in args.mal_nodes.split(",") if x.strip()]
        if args.mal_nodes
        else None
    )
    dt_levels = (
        [x.strip() for x in args.dt_levels.split(",") if x.strip()]
        if args.dt_levels
        else None
    )

    make_plots_from_csv(
        runs_csv=args.runs,
        summary_csv=args.summary,
        out_dir=args.out_dir,
        attacks=attacks,
        methods=methods,
        mal_nodes=mal_nodes,
        dt_levels=dt_levels,
        label_flip_level=args.label_flip_level,
        num_nodes=args.num_nodes,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()

