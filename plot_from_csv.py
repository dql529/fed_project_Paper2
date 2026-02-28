"""
Standalone plotting entrypoint.

Usage (default paths):
  python plot_from_csv.py

Custom:
  python plot_from_csv.py --runs runs.csv --summary summary.csv --out-dir plots
"""

from __future__ import annotations

import argparse

import dt_r4.config as C
from dt_r4.plotting import make_plots_from_csv, parse_csv_list


def _parse_float_list_csv(value: str | None):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    vals = []
    for x in text.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="runs.csv", help="Path to runs.csv")
    ap.add_argument(
        "--summary", type=str, default="summary.csv", help="Path to summary.csv"
    )
    ap.add_argument("--out-dir", type=str, default=".", help="Output directory")
    ap.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Filter a single tau_gate value when plotting/summarizing",
    )
    ap.add_argument(
        "--tau-sweep",
        type=str,
        default=None,
        help="Comma list of tau_gate values for filtering",
    )
    ap.add_argument(
        "--lam-m",
        type=float,
        default=None,
        help="Filter adaptive-mimic lambda_m value",
    )
    ap.add_argument(
        "--ref-size",
        type=int,
        default=None,
        help="Filter reference subset size",
    )
    ap.add_argument(
        "--audit-size",
        type=int,
        default=None,
        help="Filter server audit subset size",
    )
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
        help="Y metric for clean_holdout-like plot",
    )
    args = ap.parse_args()

    attacks = (
        parse_csv_list(args.attacks)
        if args.attacks
        else None
    )
    methods = parse_csv_list(args.methods) or [
        "weighted",
        "mean",
        "median",
        "trimmed_mean",
    ]
    mal_nodes = (
        [int(x) for x in parse_csv_list(args.mal_nodes) or []]
        if args.mal_nodes
        else None
    )
    dt_levels = (
        [x.strip() for x in parse_csv_list(args.dt_levels) or []]
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
        tau=args.tau,
        tau_sweep=_parse_float_list_csv(args.tau_sweep),
        lam_m=args.lam_m,
        ref_size=args.ref_size,
        audit_size=args.audit_size,
    )


if __name__ == "__main__":
    main()
