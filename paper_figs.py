from __future__ import annotations

"""Generate paper/appendix figures from CSV artifacts."""

import argparse
from pathlib import Path
import pandas as pd
from typing import List, Optional

from dt_r4.plotting import (
    load_nodes_csv,
    load_rounds_csv,
    load_summary_csv,
    parse_csv_list,
    validate_plot_inputs,
    plot_adaptive_mimic_vs_lambda,
    plot_ablation_multiattack,
    plot_auditsize_sensitivity,
    plot_clean_holdout_vs_f,
    plot_cleanf1_vs_tau,
    plot_fp_benign_vs_tau,
    plot_passrate_vs_round,
    plot_r4_distribution,
    plot_refsize_sensitivity,
    plot_sdt_vs_round,
    plot_wmal_vs_round,
    plot_wmal_vs_tau,
    plot_fallback_prob_table,
)


def _parse_int_csv_list(values: str | None) -> Optional[List[int]]:
    if values is None:
        return None
    parsed = parse_csv_list(values)
    if parsed is None:
        return None
    return [int(x) for x in parsed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--figs",
        type=str,
        default="1,2,3,5,8,9",
        help="Comma list of figure indices to generate (1-9). 6/7 are deprecated from mainline and off by default.",
    )
    ap.add_argument(
        "--dt", type=str, default="D0", help="DT level to plot (D0/D1/D2)"
    )
    ap.add_argument("--artifacts-root", type=str, default="artifacts")
    ap.add_argument("--exp-group", type=str, default="base")
    ap.add_argument("--summary", type=str, default="")
    ap.add_argument("--rounds", type=str, default="")
    ap.add_argument("--nodes", type=str, default="")
    ap.add_argument("--runs", type=str, default="")
    ap.add_argument(
        "--mal-nodes",
        type=str,
        default="0,3,5",
        help="Comma list of malicious client counts",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="weighted,mean,median,trimmed_mean",
        help="Comma list of methods",
    )
    ap.add_argument(
        "--attacks",
        type=str,
        default="label_flip,stealth_amp,dt_logit_scale",
        help="Comma list of attacks",
    )
    ap.add_argument(
        "--tau-plot-mode",
        type=str,
        default="auto",
        choices=["auto", "single", "line"],
        help="Tau plot mode for Fig.6: auto=decide from available tau_gate count, single=force point summary, line=force line",
    )
    ap.add_argument(
        "--passrate-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot benign/malicious pass rates in separate subpanels for Fig.7",
    )
    ap.add_argument(
        "--sdt-relative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use S_DT relative normalization in Fig.5",
    )
    ap.add_argument(
        "--sdt-show-ratio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show S_DT_ratio panel in Fig.5 when available",
    )
    ap.add_argument(
        "--f-fig2",
        type=int,
        default=5,
        help="f (malicious clients) for Fig.2-like curve",
    )
    ap.add_argument(
        "--f-sdt",
        type=int,
        default=5,
        help="f (malicious clients) for S_DT curves",
    )
    ap.add_argument(
        "--f-fig3",
        type=int,
        default=5,
        help="f (malicious clients) for Fig.3 R4 distribution",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory",
    )
    ap.add_argument(
        "--ablation-attacks",
        type=str,
        default="label_flip,dt_logit_scale",
        help="Attacks in fig4 ablation",
    )
    ap.add_argument(
        "--tau-gate",
        type=float,
        default=None,
        help="Optional tau filter for sensitivity plots; used as gate-line reference for Fig.3 when set (or default 0.70).",
    )
    ap.add_argument(
        "--lambda-m",
        type=float,
        default=None,
        help="Optional adaptive-mimic lambda filter",
    )
    ap.add_argument(
        "--ref-size",
        type=int,
        default=None,
        help="Optional |X_ref| filter",
    )
    ap.add_argument(
        "--audit-size",
        type=int,
        default=None,
        help="Optional audit size filter",
    )
    ap.add_argument(
        "--allow-deprecated-figs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow generating deprecated mainline figures (6,7).",
    )
    args = ap.parse_args()

    figs = {int(x.strip()) for x in args.figs.split(",") if x.strip()}
    group_root = Path(args.artifacts_root) / args.exp_group
    out_dir = Path(
        args.out_dir
        if args.out_dir
        else str(Path(args.artifacts_root) / "paper" / args.exp_group)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_csv_path(arg_value: str, filename: str) -> str:
        return arg_value if arg_value else str(group_root / filename)

    summary_df = load_summary_csv(_resolve_csv_path(args.summary, "summary.csv"))
    rounds_df = load_rounds_csv(_resolve_csv_path(args.rounds, "rounds.csv"))
    nodes_df = load_nodes_csv(_resolve_csv_path(args.nodes, "nodes.csv"))
    runs_source = _resolve_csv_path(args.runs, "runs.csv")
    runs_df = pd.read_csv(runs_source) if Path(runs_source).exists() else pd.DataFrame()
    dt = str(args.dt).strip().upper()

    attacks = parse_csv_list(args.attacks) or ["label_flip", "stealth_amp", "dt_logit_scale"]
    methods = parse_csv_list(args.methods) or [
        "weighted",
        "mean",
        "median",
        "trimmed_mean",
    ]
    mal_nodes = _parse_int_csv_list(args.mal_nodes) or [0, 3, 5]

    validate_plot_inputs(
        summary_df=summary_df,
        rounds_df=rounds_df,
        nodes_df=nodes_df,
        runs_df=runs_df,
        attacks=attacks,
        methods=methods,
        mal_nodes=mal_nodes,
        dt_levels=[dt],
    )

    # --- Figure 1: clean holdout macro-F1 vs f ---
    if 1 in figs:
        fig1 = out_dir / f"Fig1_cleanF1_vs_f_dt{dt}.png"
        plot_clean_holdout_vs_f(
            summary_df,
            attacks=["label_flip", "stealth_amp", "dt_logit_scale"],
            methods=methods,
            dt_level=dt,
            mal_nodes=mal_nodes,
            out_path=str(fig1),
            metric="clean_f1",
        )

    # --- Figure 2: W_mal vs round ---
    if 2 in figs:
        fig2 = out_dir / f"Fig2_Wmal_vs_round_dt{dt}_f{args.f_fig2}.png"
        plot_wmal_vs_round(
            rounds_df,
            dt_level=dt,
            mal_nodes=int(args.f_fig2),
            attacks=["label_flip", "stealth_amp", "dt_logit_scale"],
            out_path=str(fig2),
            method="weighted",
        )

    # --- Figure 3: R4 distribution ---
    if 3 in figs:
        fig3_f = int(args.f_fig3)
        fig3 = out_dir / f"Fig3_R4_distribution_dt{dt}_f{fig3_f}.png"
        fig3_tau_gate = 0.70 if args.tau_gate is None else float(args.tau_gate)
        plot_r4_distribution(
            nodes_df,
            attack="dt_logit_scale",
            dt_level=dt,
            mal_nodes=fig3_f,
            out_path=str(fig3),
            method="weighted",
            show_title=True,
            tau_gate=fig3_tau_gate,
            show_scatter=True,
        )

    # --- Figure 4: ablation ---
    if 4 in figs:
        print(
            "[warn] Figure 4 is deprecated from mainline; use Figure 4 only if explicitly requested."
        )
        ab_attacks = [x.strip() for x in args.ablation_attacks.split(",") if x.strip()]
        fig4 = out_dir / f"Fig4_ablation_dt{dt}.png"
        plot_ablation_multiattack(
            summary_df,
            dt_level=dt,
            attacks=ab_attacks,
            out_path=str(fig4),
            mal_nodes=mal_nodes,
            metric="polluted_f1",
        )

    # --- Figure 5: S_DT traces + fallback probability table ---
    if 5 in figs:
        fig5 = out_dir / f"Fig5_SDT_vs_round_dt{dt}_f{args.f_sdt}.png"
        plot_sdt_vs_round(
            rounds_df,
            dt_level=dt,
            mal_nodes=int(args.f_sdt),
            attacks=attacks,
            out_path=str(fig5),
            methods=["weighted"],
            show_relative=args.sdt_relative,
            show_ratio=args.sdt_show_ratio,
        )

        fallback_table = plot_fallback_prob_table(
            runs_df if not runs_df.empty else summary_df,
            attacks=attacks,
            dt_levels=[dt],
            methods=["weighted"],
            out_path=str(out_dir / f"Fig5_fallback_prob_dt{dt}.csv"),
        )
        if not fallback_table.empty:
            fallback_table.to_csv(out_dir / f"Fig5_fallback_prob_dt{dt}.csv", index=False)

    # --- Figure 6: tau sensitivity (mainline deprecated) ---
    if 6 in figs:
        if not args.allow_deprecated_figs:
            print(
                "[warn] Figure 6 is deprecated from mainline and skipped. "
                "Use --allow-deprecated-figs if you want to generate it."
            )
        else:
            df_tau = summary_df.copy()
            df_tau = df_tau[df_tau["dt_level"].astype(str) == str(dt)]
            if "method" in df_tau.columns:
                df_tau = df_tau[df_tau["method"].astype(str) == "weighted"]
            if mal_nodes is not None and "mal_nodes" in df_tau.columns:
                df_tau = df_tau[df_tau["mal_nodes"].isin(list(mal_nodes))]
            tau_g = pd.to_numeric(df_tau["tau_gate"], errors="coerce").dropna() if "tau_gate" in df_tau.columns else pd.Series(dtype=float)
            unique_tau = pd.unique(tau_g)
            if args.tau_plot_mode == "auto":
                tau_plot_single = len(unique_tau) < 2
            elif args.tau_plot_mode == "single":
                tau_plot_single = True
            else:
                tau_plot_single = False

            plot_cleanf1_vs_tau(
                summary_df,
                out_path=str(out_dir / f"Fig6_cleanF1_vs_tau_dt{dt}.png"),
                attacks=attacks,
                dt_levels=[dt],
                methods=["weighted"],
                label_flip_level="L1",
                metric="clean_f1",
                single_point_fallback=tau_plot_single,
            )
            plot_wmal_vs_tau(
                summary_df,
                out_path=str(out_dir / f"Fig6_Wmal_vs_tau_dt{dt}.png"),
                attacks=attacks,
                dt_levels=[dt],
                methods=["weighted"],
                label_flip_level="L1",
                single_point_fallback=tau_plot_single,
            )
            plot_fp_benign_vs_tau(
                summary_df,
                out_path=str(out_dir / f"Fig6_FPerr_vs_tau_dt{dt}.png"),
                attacks=attacks,
                dt_levels=[dt],
                methods=["weighted"],
                label_flip_level="L1",
                single_point_fallback=tau_plot_single,
            )

    # --- Figure 7: pass-rate curves over rounds ---
    # --- Figure 7: pass-rate curves over rounds (mainline deprecated) ---
    if 7 in figs:
        if not args.allow_deprecated_figs:
            print(
                "[warn] Figure 7 is deprecated from mainline and skipped. "
                "Use --allow-deprecated-figs if you want to generate it."
            )
        else:
            plot_passrate_vs_round(
                rounds_df,
                dt_level=dt,
                mal_nodes=int(args.f_fig2),
                attacks=attacks,
                out_path=str(out_dir / f"Fig7_passrate_vs_round_dt{dt}_f{args.f_fig2}.png"),
                method="weighted",
                metric="benign_pass_rate",
                split_metric=args.passrate_split,
            )

    # --- Figure 8: ref-size and audit-size sensitivity ---
    if 8 in figs:
        plot_refsize_sensitivity(
            summary_df,
            out_dir=str(out_dir / "tau_refsize"),
            prefix=f"Fig8_dt{dt}_f{args.f_fig2}",
            attacks=attacks,
            dt_levels=[dt],
            mal_nodes=[args.f_fig2],
            methods=["weighted"],
            fixed_tau=args.tau_gate,
            fixed_lambda=args.lambda_m,
            fixed_audit_size=args.audit_size,
        )
        plot_auditsize_sensitivity(
            summary_df,
            out_dir=str(out_dir / "tau_auditsize"),
            prefix=f"Fig8_dt{dt}_f{args.f_fig2}",
            attacks=attacks,
            dt_levels=[dt],
            mal_nodes=[args.f_fig2],
            methods=["weighted"],
            fixed_tau=args.tau_gate,
            fixed_lambda=args.lambda_m,
            fixed_ref_size=args.ref_size,
        )

    # --- Figure 9: adaptive mimic ----
    if 9 in figs:
        run_df = pd.read_csv(args.runs) if args.runs else pd.DataFrame()
        if "adaptive_mimic" in attacks:
            plot_adaptive_mimic_vs_lambda(
                run_df,
                nodes_df=nodes_df,
                dt_level=dt,
                out_dir=str(out_dir / "adaptive_mimic"),
                mal_nodes=mal_nodes,
                method="weighted",
                fixed_ref_size=args.ref_size,
                fixed_audit_size=args.audit_size,
                out_prefix=f"Fig9_dt{dt}",
            )


if __name__ == "__main__":
    main()
