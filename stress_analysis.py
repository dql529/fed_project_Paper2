from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _parse_csv_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _resolve_root(raw: str) -> Path:
    p = Path(raw)
    if (p / "config.json").exists():
        return p
    direct = [d for d in p.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if len(direct) == 1:
        return direct[0]
    preferred = [d for d in direct if d.name == "heterobenign"]
    if len(preferred) == 1:
        return preferred[0]
    raise FileNotFoundError(f"Could not resolve experiment root from: {raw}")


def _experiment_id(root: Path) -> str:
    return root.parent.name if root.name == "heterobenign" else root.name


def _load_csv(root: Path, rel: str) -> pd.DataFrame:
    path = root / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _infer_family(profile: str, cfg: Dict) -> str:
    tokens = {x.strip().lower() for x in str(profile).split(",") if x.strip()}
    specs = {
        str(cfg.get("benign_noise_light_spec", "")),
        str(cfg.get("benign_noise_medium_spec", "")),
        str(cfg.get("benign_noise_heavy_spec", "")),
    }
    if tokens & {"mediump", "mediumn", "heavyp", "heavyn"}:
        return "bipolar_drift"
    if any("_pos_" in x or "_neg_" in x for x in specs):
        return "bipolar_drift"
    if any(str(x).startswith("drift_") for x in specs):
        return "drift"
    return "iid"


def _deploy_surface_name(cfg: Dict) -> str:
    scenario = str(cfg.get("scenario", "A")).upper()
    if scenario == "A":
        return "local_test_pool"
    return f"noise_variant_{int(cfg.get('deploy_variant', 0))}"


def _get_method_row(df: pd.DataFrame, attack: str, mal_nodes: int, method: str) -> pd.Series:
    sub = df[
        (df["attack"].astype(str) == str(attack))
        & (pd.to_numeric(df["mal_nodes"], errors="coerce") == int(mal_nodes))
        & (df["method"].astype(str) == str(method))
    ]
    if sub.empty:
        raise KeyError((attack, mal_nodes, method))
    return sub.iloc[0]


def _role_gap(table_d: pd.DataFrame, attack: str, mal_nodes: int) -> Tuple[float, float, float]:
    sub = table_d[
        (table_d["attack"].astype(str) == str(attack))
        & (pd.to_numeric(table_d["mal_nodes"], errors="coerce") == int(mal_nodes))
        & (table_d["method"].astype(str) == "weighted")
    ]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    heavy = sub[sub["node_noise_role"].astype(str) == "benign_heavy"]
    mal = sub[sub["node_noise_role"].astype(str) == "malicious"]
    if heavy.empty or mal.empty:
        return float("nan"), float("nan"), float("nan")
    h = heavy.iloc[0]
    m = mal.iloc[0]
    return (
        _safe_float(h["R4_m"]) - _safe_float(m["R4_m"]),
        _safe_float(h["Rep_m"]) - _safe_float(m["Rep_m"]),
        _safe_float(h["passed_gate_m"]) - _safe_float(m["passed_gate_m"]),
    )


def _build_summary(base_root: Path, candidate_roots: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_table = _load_csv(base_root, "tables/table_A_methods.csv")
    summary_rows: List[Dict] = []
    nodes_rows: List[pd.DataFrame] = []

    for raw_root in candidate_roots:
        root = _resolve_root(str(raw_root))
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        exp_id = _experiment_id(root)
        profile = str(cfg.get("selected_benign_noise_profile", ""))
        family = _infer_family(profile, cfg)
        table_a = _load_csv(root, "tables/table_A_methods.csv")
        table_d = _load_csv(root, "tables/table_D_weighted_noise_roles.csv")
        nodes = _load_csv(root, "nodes.csv")

        if not nodes.empty:
            nodes = nodes.copy()
            nodes["experiment_id"] = exp_id
            nodes["family"] = family
            nodes["profile"] = profile
            nodes["reference_surface"] = "full_reference_pool"
            nodes["deploy_surface"] = _deploy_surface_name(cfg)
            nodes_rows.append(nodes)

        if table_a.empty:
            continue

        weighted_rows = table_a[table_a["method"].astype(str) == "weighted"]
        for _, w_row in weighted_rows.iterrows():
            attack = str(w_row["attack"])
            mal_nodes = int(_safe_float(w_row["mal_nodes"]))
            try:
                m_row = _get_method_row(table_a, attack, mal_nodes, "median")
                base_w = _get_method_row(base_table, attack, mal_nodes, "weighted")
                base_m = _get_method_row(base_table, attack, mal_nodes, "median")
            except KeyError:
                continue

            r4_gap, rep_gap, pass_gap = _role_gap(table_d, attack, mal_nodes)
            median_reference_drop_abs = _safe_float(base_m["clean_f1_m"]) - _safe_float(m_row["clean_f1_m"])
            median_deploy_drop_abs = _safe_float(base_m["polluted_f1_m"]) - _safe_float(m_row["polluted_f1_m"])
            weighted_reference_drop_abs = _safe_float(base_w["clean_f1_m"]) - _safe_float(w_row["clean_f1_m"])
            weighted_deploy_drop_abs = _safe_float(base_w["polluted_f1_m"]) - _safe_float(w_row["polluted_f1_m"])
            base_m_ref = max(abs(_safe_float(base_m["clean_f1_m"])), 1e-12)
            base_m_dep = max(abs(_safe_float(base_m["polluted_f1_m"])), 1e-12)
            success_abs10_or_rel10 = bool(
                (median_deploy_drop_abs >= 0.10)
                or (median_deploy_drop_abs / base_m_dep >= 0.10)
            )
            success_strict = bool(
                success_abs10_or_rel10
                and weighted_deploy_drop_abs < median_deploy_drop_abs
                and _safe_float(w_row["w_mal_m"]) < _safe_float(m_row["w_mal_m"])
                and np.isfinite(r4_gap)
                and np.isfinite(rep_gap)
                and r4_gap > 0
                and rep_gap > 0
            )

            summary_rows.append(
                {
                    "experiment_id": exp_id,
                    "family": family,
                    "profile": profile,
                    "attack": attack,
                    "mal_nodes": mal_nodes,
                    "reference_surface": "full_reference_pool",
                    "deploy_surface": _deploy_surface_name(cfg),
                    "reference_f1_median": _safe_float(m_row["clean_f1_m"]),
                    "deploy_f1_median": _safe_float(m_row["polluted_f1_m"]),
                    "reference_f1_weighted": _safe_float(w_row["clean_f1_m"]),
                    "deploy_f1_weighted": _safe_float(w_row["polluted_f1_m"]),
                    "median_reference_drop_abs": median_reference_drop_abs,
                    "median_deploy_drop_abs": median_deploy_drop_abs,
                    "median_reference_drop_rel": median_reference_drop_abs / base_m_ref,
                    "median_deploy_drop_rel": median_deploy_drop_abs / base_m_dep,
                    "weighted_reference_drop_abs": weighted_reference_drop_abs,
                    "weighted_deploy_drop_abs": weighted_deploy_drop_abs,
                    "weighted_minus_median_reference": _safe_float(w_row["clean_f1_m"]) - _safe_float(m_row["clean_f1_m"]),
                    "weighted_minus_median_deploy": _safe_float(w_row["polluted_f1_m"]) - _safe_float(m_row["polluted_f1_m"]),
                    "weighted_w_mal": _safe_float(w_row["w_mal_m"]),
                    "median_w_mal": _safe_float(m_row["w_mal_m"]),
                    "R4_gap_heavy_minus_mal": r4_gap,
                    "Rep_gap_heavy_minus_mal": rep_gap,
                    "pass_gap_heavy_minus_mal": pass_gap,
                    "stealth_max_amp": _safe_float(cfg.get("stealth_max_amp", float("nan"))),
                    "stealth_amp_step": _safe_float(cfg.get("stealth_amp_step", float("nan"))),
                    "stealth_noise_base": _safe_float(cfg.get("stealth_noise_base", float("nan"))),
                    "stealth_noise_step": _safe_float(cfg.get("stealth_noise_step", float("nan"))),
                    "dt_attack_scale_start": _safe_float(cfg.get("dt_attack_scale_start", float("nan"))),
                    "dt_attack_scale_end": _safe_float(cfg.get("dt_attack_scale_end", float("nan"))),
                    "dt_attack_scale_step": _safe_float(cfg.get("dt_attack_scale_step", float("nan"))),
                    "label_flip_grad_scale": _safe_float(cfg.get("label_flip_grad_scale", float("nan"))),
                    "success_abs10_or_rel10": int(success_abs10_or_rel10),
                    "success_strict": int(success_strict),
                }
            )

    return pd.DataFrame(summary_rows), pd.concat(nodes_rows, ignore_index=True) if nodes_rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-root", type=str, default="artifacts/base")
    ap.add_argument("--candidate-roots", type=str, required=True, help="Comma list of experiment roots")
    ap.add_argument("--out-dir", type=str, default="artifacts/stress_analysis")
    args = ap.parse_args()

    base_root = _resolve_root(args.base_root)
    candidate_roots = [_resolve_root(x) for x in _parse_csv_list(args.candidate_roots)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, nodes_df = _build_summary(base_root, candidate_roots)
    summary_df.sort_values(["attack", "experiment_id"]).to_csv(out_dir / "stress_summary.csv", index=False)
    if not nodes_df.empty:
        keep_cols = [
            "experiment_id",
            "family",
            "profile",
            "attack",
            "dt_level",
            "mal_nodes",
            "method",
            "seed",
            "node_id",
            "node_noise_role",
            "node_noise_spec",
            "R4",
            "Rep",
            "KL_q_p",
            "passed_gate",
            "rep_config",
            "reference_surface",
            "deploy_surface",
        ]
        existing = [c for c in keep_cols if c in nodes_df.columns]
        nodes_df[existing].to_csv(out_dir / "stress_nodes_long.csv", index=False)
    print(f"[ok] wrote {out_dir / 'stress_summary.csv'}")
    if not nodes_df.empty:
        print(f"[ok] wrote {out_dir / 'stress_nodes_long.csv'}")


if __name__ == "__main__":
    main()
