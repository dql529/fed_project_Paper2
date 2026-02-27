import argparse
from typing import List, Dict, Optional

import numpy as np
import torch

import dt_r4.config as C
from dt_r4 import aggregators as agg
from dt_r4.data import (
    build_noise_variants_fixed,
    load_node_splits,
    load_reference_data,
    load_and_clean_csv,
    get_teacher_model,
)
from dt_r4.federated import CentralServer, DroneNode
from dt_r4.runtime import set_seeds, device
from dt_r4.twin import build_twin_mismatch_context, get_twin_logits
from dt_r4.utils import weighted_average_aggregation
from dt_r4.config import R2_IGNORE_MALICIOUS


def _tensor_from_df(df):
    x = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32, device=device)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long, device=device)
    return x, y


def _macro_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.cpu().numpy()
    tgt_np = target.cpu().numpy()
    num_classes = max(int(tgt_np.max()) + 1, int(pred_np.max()) + 1)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((pred_np == c) & (tgt_np == c))
        fp = np.sum((pred_np == c) & (tgt_np != c))
        fn = np.sum((pred_np != c) & (tgt_np == c))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
        elif tp == 0 and fp == 0 and fn == 0:
            f1s.append(1.0)
        else:
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))


def eval_model(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        f1 = _macro_f1(pred, y)
    return float(acc), float(f1)


def _flatten_model(model: torch.nn.Module) -> torch.Tensor:
    sd = model.state_dict()
    return torch.cat(
        [v.detach().reshape(-1).to("cpu", dtype=torch.float32) for v in sd.values()]
    )


def _pairwise_dist_matrix(vecs: List[torch.Tensor]) -> torch.Tensor:
    mat = torch.stack(vecs, dim=0)
    sq = (mat * mat).sum(dim=1, keepdim=True)
    d2 = sq + sq.t() - 2.0 * (mat @ mat.t())
    d2.clamp_(min=0)
    return torch.sqrt(d2 + 1e-12)


def flip_labels_multiclass(y: torch.Tensor, flip_ratio: float = 1.0) -> torch.Tensor:
    """Flip labels to other classes with given ratio; supports multi-class."""
    if flip_ratio <= 0:
        return y
    y2 = y.clone()
    num_classes = int(y2.max().item()) + 1 if y2.numel() > 0 else 0
    if num_classes <= 1:
        return y2

    if flip_ratio >= 1.0:
        mask = torch.ones_like(y2, dtype=torch.bool)
    else:
        mask = torch.rand_like(y2.float()) < float(flip_ratio)
    if not mask.any():
        return y2

    if num_classes == 2:
        y2[mask] = 1 - y2[mask]
    else:
        rand = torch.randint(0, num_classes - 1, size=(mask.sum(),), device=y2.device)
        orig = y2[mask]
        rand = rand + (rand >= orig).long()
        y2[mask] = rand
    return y2


def run_once(
    method: str,
    rounds: int,
    deploy_variant: int,
    scenario: str,
    seed: int = 0,
    diag: Optional[List[Dict]] = None,
    round_rows: Optional[List[Dict]] = None,
    node_rows: Optional[List[Dict]] = None,
    meta: Optional[Dict] = None,
) -> Dict[str, float]:
    """Run one FL simulation and return (overall_metrics, rep_table).

    Important: keep the control-flow structure correct; otherwise we may accidentally
    return after creating only one node (which breaks logs/figures).
    """

    method = str(method).strip()
    set_seeds(seed)

    # ---- data ----
    node_data_objects, _ = load_node_splits(
        C.NUM_NODES,
        C.CSV_PATH,
        seed=0,  # fixed split; randomness comes from optimizer + attacks
        malicious_nodes=C.MALICIOUS_NODES if C.PRE_SPLIT_POISON else 0,
        attack_mode=C.MAL_ATTACK_MODE,
        label_flip_ratio=float(getattr(C, "LABEL_FLIP_RATIO", 0.0)),
        pre_split_poison=bool(getattr(C, "PRE_SPLIT_POISON", False)),
    )

    # Clean holdout used ONLY for final evaluation (GLOBAL_REF_CSV).
    # Server-side R4 reference may be a degraded/subsampled version of it.
    ref_x, ref_y = load_reference_data(C.GLOBAL_REF_CSV)
    r4_ref_x = ref_x

    # Local train/test concatenations (no global prior)
    clean_eval_x = torch.cat([d["test"].x for d in node_data_objects], dim=0)
    clean_eval_y = torch.cat([d["test"].y for d in node_data_objects], dim=0)
    train_eval_x = torch.cat([d["train"].x for d in node_data_objects], dim=0)
    train_eval_y = torch.cat([d["train"].y for d in node_data_objects], dim=0)

    # Deploy data: scenario A => local clean test; scenario B => chosen noise variant
    if scenario.upper() == "A":
        dep_x, dep_y = clean_eval_x, clean_eval_y
    else:
        noise_variants = build_noise_variants_fixed(C.CSV_PATH, dataset_seed=123)
        deploy_variant = max(0, min(deploy_variant, len(noise_variants) - 1))
        dep_df = load_and_clean_csv(noise_variants[deploy_variant]["path"])
        dep_x, dep_y = _tensor_from_df(dep_df)

    # ---- method / ablation ----
    use_reputation = method.startswith("weighted")

    # Weighted ablation variants for Fig.4
    #   weighted_full   : R2+R3+R4 + gate (default)
    #   weighted_r4only  : R4 only
    #   weighted_r2only  : R2 only
    #   weighted_nogate  : full but disable gate
    rep_ablation = method
    if rep_ablation in {"weighted", "weighted_full"}:
        rep_ablation = "weighted_full"
    elif use_reputation and rep_ablation not in {
        "weighted_full",
        "weighted_r4only",
        "weighted_r2only",
        "weighted_nogate",
    }:
        rep_ablation = "weighted_full"

    orig_tau = float(getattr(C, "R4_GATE_TAU", 0.5))
    if use_reputation and rep_ablation == "weighted_nogate":
        C.R4_GATE_TAU = 0.0

    try:
        if not use_reputation:
            ablation_config = ""
        elif rep_ablation == "weighted_r4only":
            ablation_config = "R4"
        elif rep_ablation == "weighted_r2only":
            ablation_config = "R2"
        else:
            ablation_config = "R2,R3,R4"

        active = {x.strip() for x in ablation_config.split(",") if x.strip()}
        need_r4 = use_reputation and ("R4" in active)
        dummy_logits = torch.zeros((1, 2), device=device)

        # ---- server ----
        server = CentralServer(
            ablation_config=ablation_config,
            r4_alpha=4.0,
            use_perf_penalty=False,
        )

        if need_r4:
            teacher = get_teacher_model()
            twin_ctx = build_twin_mismatch_context(ref_x, C.TWIN_MISMATCH_SPECS[0])
            twin_logits_ref = get_twin_logits(teacher, ref_x, twin_ctx)

            # DT fidelity: degrade twin logits and/or subsample the reference.
            dt_level = getattr(C, "_DT_ACTIVE_LEVEL", "D0")
            dt_cfg = C.DT_MISMATCH_LEVELS.get(dt_level, C.DT_MISMATCH_LEVELS["D0"])
            keep_ratio = float(dt_cfg.get("keep_ratio", 1.0))
            noise_std = float(dt_cfg.get("noise_std", 0.0))

            if keep_ratio < 1.0:
                keep_n = max(1, int(len(ref_x) * keep_ratio))
                idx = torch.randperm(len(ref_x))[:keep_n]
                r4_ref_x = ref_x[idx]
                twin_logits_ref = twin_logits_ref[idx]

            if noise_std > 0.0:
                twin_logits_ref = twin_logits_ref + noise_std * torch.randn_like(twin_logits_ref)

            server.set_twin_reference(
                twin_logits_for_probs=twin_logits_ref,
                twin_logits_for_mask=twin_logits_ref,
                temperature=1.0,
            )

        # ---- init nodes ----
        nodes: List[DroneNode] = []
        for i in range(C.NUM_NODES):
            node = DroneNode(
                drone_id=i,
                is_malicious=(i < C.MALICIOUS_NODES),
                attack_mode=C.MAL_ATTACK_MODE,
            )
            node.data = node_data_objects[i]["train"]
            node.test_data = node_data_objects[i]["test"]
            server.data_age[node.drone_id] = 0
            server.reputation[node.drone_id] = 1.0
            nodes.append(node)

        # ---- training loop ----
        last_rep: Dict[int, float] = {}
        last_comp: Dict[int, tuple] = {}

        for r in range(1, rounds + 1):
            local_models = []
            node_perfs: Dict[int, float] = {}
            node_logits: Dict[int, torch.Tensor] = {}

            for node in nodes:
                node.receive_global_model(server.global_model)
                node.train()
                node.local_model.drone_id = node.drone_id
                local_models.append(node.local_model)

                with torch.no_grad():
                    test_x = node.test_data.x
                    test_y = node.test_data.y
                    if (
                        not getattr(C, "PRE_SPLIT_POISON", False)
                        and node.is_malicious
                        and getattr(C, "POISON_LOCAL_TEST", False)
                        and C.MAL_ATTACK_MODE == "label_flip"
                    ):
                        test_y = flip_labels_multiclass(
                            test_y,
                            flip_ratio=float(getattr(C, "LABEL_FLIP_RATIO", 1.0)),
                        )

                    test_logits = node.local_model(test_x)
                    preds = test_logits.argmax(dim=1)
                    acc = (preds == test_y).float().mean().item()
                    if node.is_malicious and R2_IGNORE_MALICIOUS:
                        acc = 1.0
                    node_perfs[node.drone_id] = float(acc)

                    if need_r4:
                        node_logits[node.drone_id] = node.local_model(r4_ref_x)

            # ---- scale / reverse malicious updates (optional; controlled by config) ----
            scale = float(getattr(C, "MAL_GRAD_SCALE", 1.0))
            scale_map = getattr(C, "MAL_GRAD_SCALE_MAP", None)
            if scale_map and getattr(C, "MAL_ATTACK_MODE", None) in scale_map:
                scale = float(scale_map[getattr(C, "MAL_ATTACK_MODE")])

            if getattr(C, "MAL_ATTACK_MODE", None) and scale != 1.0:
                base_sd = server.global_model.state_dict()
                for idx, node in enumerate(nodes):
                    if not node.is_malicious:
                        continue
                    sd = local_models[idx].state_dict()
                    for k, v in sd.items():
                        delta = v - base_sd[k]
                        sd[k] = base_sd[k] + scale * delta
                    local_models[idx].load_state_dict(sd)

                # Make reputation see the submitted (scaled) params.
                if use_reputation:
                    for node in nodes:
                        with torch.no_grad():
                            test_x = node.test_data.x
                            test_y = node.test_data.y
                            if (
                                not getattr(C, "PRE_SPLIT_POISON", False)
                                and node.is_malicious
                                and getattr(C, "POISON_LOCAL_TEST", False)
                                and C.MAL_ATTACK_MODE == "label_flip"
                            ):
                                test_y = flip_labels_multiclass(
                                    test_y,
                                    flip_ratio=float(getattr(C, "LABEL_FLIP_RATIO", 1.0)),
                                )

                            test_logits = local_models[node.drone_id](test_x)
                            preds = test_logits.argmax(dim=1)
                            acc = (preds == test_y).float().mean().item()
                            if node.is_malicious and R2_IGNORE_MALICIOUS:
                                acc = 1.0
                            node_perfs[node.drone_id] = float(acc)

                            if need_r4:
                                node_logits[node.drone_id] = local_models[node.drone_id](r4_ref_x)

            # ---- compute reputation ----
            server.reputation = {}
            last_rep = {}
            last_comp = {}
            if use_reputation:
                for node in nodes:
                    rep, comps = server.compute_reputation(
                        drone_id=node.drone_id,
                        performance=node_perfs[node.drone_id],
                        data_age=0,
                        local_logits=node_logits.get(node.drone_id, dummy_logits),
                        current_round=r,
                    )
                    server.reputation[node.drone_id] = float(rep)
                    last_rep[node.drone_id] = float(rep)
                    last_comp[node.drone_id] = comps
            else:
                for node in nodes:
                    server.reputation[node.drone_id] = 1.0

            # per-round mechanism log (weighted only)
            if use_reputation and round_rows is not None:
                total = sum(server.reputation.values()) or 1.0
                mal_ids = [n.drone_id for n in nodes if n.is_malicious]
                w_mal_round = sum(server.reputation.get(i, 0.0) for i in mal_ids) / total
                round_rows.append(
                    {
                        "attack": (meta or {}).get("attack", getattr(C, "MAL_ATTACK_MODE", "")),
                        "level": (meta or {}).get("level", ""),
                        "dt_level": (meta or {}).get("dt_level", getattr(C, "_DT_ACTIVE_LEVEL", "D0")),
                        "mal_nodes": int((meta or {}).get("mal_nodes", getattr(C, "MALICIOUS_NODES", 0))),
                        "method": (meta or {}).get("method", method),
                        "seed": int((meta or {}).get("seed", seed)),
                        "round": int(r),
                        "w_mal": float(w_mal_round),
                    }
                )

            # diagnostics for byzantine-style methods
            diag_methods = {"byzantine_median", "krum", "bulyan", "trimmed_mean"}
            if diag is not None and method in diag_methods:
                base_vec = _flatten_model(server.global_model)
                flat_locals = [_flatten_model(m) for m in local_models]
                dmat = _pairwise_dist_matrix(flat_locals)
                l2_to_global = [torch.dist(v, base_vec).item() for v in flat_locals]

                krum_scores = [None] * len(nodes)
                krum_selected = [False] * len(nodes)
                bulyan_selected = [False] * len(nodes)
                trimmed_flag = [False] * len(nodes)

                # krum scores / winner
                if method in {"krum", "bulyan"}:
                    n = len(nodes)
                    f = C.MALICIOUS_NODES
                    nb_in_score = max(1, n - f - 2)
                    for i in range(n):
                        s = np.sort(dmat[i][np.arange(n) != i].numpy())[:nb_in_score].sum()
                        krum_scores[i] = float(s)
                    winner = int(
                        np.argmin([s if s is not None else 1e9 for s in krum_scores])
                    )
                    krum_selected[winner] = True

                # bulyan winners
                if method == "bulyan":
                    n = len(nodes)
                    f = C.MALICIOUS_NODES
                    m = max(1, n - 2 * f)
                    scores_sorted_idx = np.argsort(
                        [s if s is not None else 1e9 for s in krum_scores]
                    )
                    winners = scores_sorted_idx[:m]
                    for w in winners:
                        bulyan_selected[int(w)] = True

                # trimmed_mean: flag nodes that would be trimmed by overall update norm
                if method == "trimmed_mean":
                    trim = int(len(nodes) * 0.2)
                    order = np.argsort(l2_to_global)
                    low = set(order[:trim])
                    high = set(order[-trim:]) if trim > 0 else set()
                    for i in range(len(nodes)):
                        if i in low or i in high:
                            trimmed_flag[i] = True

                for idx, node in enumerate(nodes):
                    diag.append(
                        {
                            "method": method,
                            "round": r,
                            "node_id": node.drone_id,
                            "is_malicious": int(node.is_malicious),
                            "l2_to_global": l2_to_global[idx],
                            "krum_score": (
                                krum_scores[idx] if krum_scores[idx] is not None else ""
                            ),
                            "krum_selected": int(krum_selected[idx]),
                            "bulyan_selected": int(bulyan_selected[idx]),
                            "trimmed_flag": int(trimmed_flag[idx]),
                        }
                    )

            # ---- aggregation ----
            if method.startswith("weighted"):
                agg_sd = weighted_average_aggregation(local_models, server.reputation)
            elif method == "mean":
                agg_sd = agg.mean_aggregation(local_models)
            elif method == "median":
                agg_sd = agg.median_aggregation(local_models)
            elif method == "byzantine_median":
                agg_sd = agg.byzantine_median_aggregation(local_models)
            elif method == "trimmed_mean":
                agg_sd = agg.trimmed_mean_aggregation(local_models, trim_ratio=0.2)
            elif method == "krum":
                agg_sd = agg.krum_aggregation(local_models, f=C.MALICIOUS_NODES)
            elif method == "bulyan":
                agg_sd = agg.bulyan_aggregation(local_models, f=C.MALICIOUS_NODES)
            else:
                raise ValueError(f"Unknown method: {method}")

            server.global_model.load_state_dict(agg_sd)

        # ---- evaluation ----
        clean_acc, clean_f1 = eval_model(server.global_model, clean_eval_x, clean_eval_y)
        deploy_acc, deploy_f1 = eval_model(server.global_model, dep_x, dep_y)
        train_acc, train_f1 = eval_model(server.global_model, train_eval_x, train_eval_y)
        final_acc, final_f1 = eval_model(server.global_model, ref_x, ref_y)

        # W_mal_total at the final round
        w_mal = None
        if use_reputation and last_rep:
            total_rep = sum(last_rep.values()) or 1.0
            mal_ids = [n.drone_id for n in nodes if n.is_malicious]
            w_mal = sum(last_rep.get(i, 0.0) for i in mal_ids) / total_rep

        # per-node final log (weighted only)
        if use_reputation and node_rows is not None and last_rep and last_comp:
            tau_gate = float(getattr(C, "R4_GATE_TAU", 0.5))
            for node in nodes:
                nid = int(node.drone_id)
                comps = last_comp.get(nid, (float("nan"), float("nan"), float("nan")))
                node_rows.append(
                    {
                        "attack": (meta or {}).get("attack", getattr(C, "MAL_ATTACK_MODE", "")),
                        "level": (meta or {}).get("level", ""),
                        "dt_level": (meta or {}).get("dt_level", getattr(C, "_DT_ACTIVE_LEVEL", "D0")),
                        "mal_nodes": int((meta or {}).get("mal_nodes", getattr(C, "MALICIOUS_NODES", 0))),
                        "method": (meta or {}).get("method", method),
                        "seed": int((meta or {}).get("seed", seed)),
                        "round": int(rounds),
                        "node_id": nid,
                        "is_malicious": int(bool(node.is_malicious)),
                        "rep": float(last_rep.get(nid, float("nan"))),
                        "R2": float(comps[0]),
                        "R3": float(comps[1]),
                        "R4": float(comps[2]),
                        "r4_kl": float(server.last_r4_kl.get(nid, float("nan"))),
                        "r4_gate_hit": int(bool(server.last_r4_gate_hit.get(nid, False))),
                        "tau_gate": float(tau_gate),
                    }
                )

        overall = {
            "train_acc": train_acc,
            "train_f1": train_f1,
            "eval_acc": deploy_acc,
            "eval_f1": deploy_f1,
            "final_acc": final_acc,
            "final_f1": final_f1,
            "eval_drop": clean_acc - deploy_acc,
            "w_mal": w_mal,
        }

        rep_table = {
            nid: {
                "rep": last_rep[nid],
                "R2": last_comp[nid][0],
                "R3": last_comp[nid][1],
                "R4": last_comp[nid][2],
            }
            for nid in last_rep
        }

        return overall, rep_table

    finally:
        if use_reputation:
            C.R4_GATE_TAU = orig_tau
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        type=str,
        default="weighted,mean,median,trimmed_mean",
    )
    ap.add_argument("--rounds", type=int, default=C.NUM_ROUNDS)
    ap.add_argument(
        "--deploy-variant",
        type=int,
        default=C.EVAL_DEPLOY_VARIANT,
        help="index into NOISE_VARIANTS (0=clean)",
    )
    ap.add_argument(
        "--scenario",
        type=str,
        default=C.EVAL_SCENARIO,
        help="A=train clean/eval clean, B=train clean/eval noise",
    )
    ap.add_argument(
        "--diag-path",
        type=str,
        default=None,
        help="Optional CSV to dump per-round diagnostics (byzantine methods only)",
    )
    ap.add_argument(
        "--attack-modes",
        type=str,
        default="label_flip,stealth_amp,dt_logit_scale",
        help="Comma list of attack modes to sweep",
    )
    ap.add_argument(
        "--mal-nodes",
        type=str,
        default="0,3,5",
        help="Comma list of malicious node counts to sweep",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma list of seeds for averaging (e.g., 0,1,2,3,4)",
    )
    ap.add_argument(
        "--dt-levels",
        type=str,
        default="D0,D1,D2",
        help="Comma list of DT fidelity levels (D0/D1/D2); only affects weighted R4",
    )
    ap.add_argument(
        "--out-runs",
        type=str,
        default=None,
        help="Optional CSV to save per-seed metrics",
    )
    ap.add_argument(
        "--out-summary",
        type=str,
        default=None,
        help="Optional CSV to save mean/std grouped metrics",
    )
    ap.add_argument(
        "--out-rounds",
        type=str,
        default=None,
        help="Optional CSV to save per-round mechanism logs (weighted only)",
    )
    ap.add_argument(
        "--out-nodes",
        type=str,
        default=None,
        help="Optional CSV to save per-node final scores (weighted only)",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    attack_modes = [m.strip() for m in args.attack_modes.split(",") if m.strip()]
    mal_nodes_list = [int(x) for x in args.mal_nodes.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dt_levels = [lv.strip() for lv in args.dt_levels.split(",") if lv.strip()]
    # 固定单一 label_flip 级别（L1），不再扫描 L0/L2
    levels = ["L1"]

    print("method, polluted_acc, polluted_f1, clean_acc, clean_f1, W_mal_total")
    print("-------------------------------------------------------------------")
    rep_logs = {}
    diag_rows: List[Dict] = []
    summary = {}  # (attack, level, dt_level, mal, method) -> metrics
    runs_rows: List[Dict] = []
    rounds_rows: List[Dict] = []
    nodes_rows: List[Dict] = []

    for attack in attack_modes:
        level_list = levels if attack == "label_flip" else [""]
        for level in level_list:
            # apply per-level overrides for label_flip
            if attack == "label_flip":
                # 固定为 L1：50% 翻转 + 0.06 学习率
                C.LABEL_FLIP_RATIO = 0.5
                C.LABEL_FLIP_LR = 0.06
                C.MAL_GRAD_SCALE_MAP["label_flip"] = 0.5
            for dt_level in dt_levels:
                dt_cfg = C.DT_MISMATCH_LEVELS.get(dt_level, C.DT_MISMATCH_LEVELS["D0"])
                C._DT_ACTIVE_LEVEL = dt_level  # stash for use in run_once
                for mal_n in mal_nodes_list:
                    C.MAL_ATTACK_MODE = attack
                    C.MALICIOUS_NODES = mal_n
                    print(
                        f"\n=== attack={attack}, dt={dt_level}, mal_nodes={mal_n} ==="
                    )
                    for m in methods:
                        metrics = []
                        rep_table_last = {}
                        for seed in seeds:
                            overall, rep_table = run_once(
                                m,
                                rounds=args.rounds,
                                deploy_variant=args.deploy_variant,
                                scenario=args.scenario,
                                seed=seed,
                                diag=diag_rows,
                                round_rows=rounds_rows,
                                node_rows=nodes_rows,
                                meta={
                                    "attack": attack,
                                    "level": level,
                                    "dt_level": dt_level,
                                    "mal_nodes": mal_n,
                                    "method": m,
                                    "seed": seed,
                                },
                            )
                            metrics.append(overall)
                            rep_table_last = rep_table  # keep last
                            runs_rows.append(
                                {
                                    "attack": attack,
                                    "level": level,
                                    "dt_level": dt_level,
                                    "mal_nodes": mal_n,
                                    "method": m,
                                    "seed": seed,
                                    "polluted_acc": overall["eval_acc"],
                                    "polluted_f1": overall["eval_f1"],
                                    "clean_acc": overall["final_acc"],
                                    "clean_f1": overall["final_f1"],
                                    "w_mal": overall["w_mal"],
                                }
                            )
                        # aggregate
                        poll_acc = [o["eval_acc"] for o in metrics]
                        poll_f1 = [o["eval_f1"] for o in metrics]
                        clean_acc = [o["final_acc"] for o in metrics]
                        clean_f1 = [o["final_f1"] for o in metrics]
                        wmal_vals = [
                            o["w_mal"] for o in metrics if o["w_mal"] is not None
                        ]

                        def mean_std(arr):
                            arr = np.array(arr, dtype=float)
                            return float(arr.mean()), float(arr.std())

                        pa_m, pa_s = mean_std(poll_acc)
                        pf_m, pf_s = mean_std(poll_f1)
                        ca_m, ca_s = mean_std(clean_acc)
                        cf_m, cf_s = mean_std(clean_f1)
                        w_m, w_s = (
                            mean_std(wmal_vals)
                            if wmal_vals
                            else (float("nan"), float("nan"))
                        )
                        summary[(attack, level, dt_level, mal_n, m)] = {
                            "poll_acc_m": pa_m,
                            "poll_acc_s": pa_s,
                            "poll_f1_m": pf_m,
                            "poll_f1_s": pf_s,
                            "clean_acc_m": ca_m,
                            "clean_acc_s": ca_s,
                            "clean_f1_m": cf_m,
                            "clean_f1_s": cf_s,
                            "w_mal_m": w_m,
                            "w_mal_s": w_s,
                        }
                        rep_logs[(attack, level, dt_level, mal_n, m)] = rep_table_last
                        wmal_field = (
                            f"{w_m:>10.4f}" if not np.isnan(w_m) else f"{'':>10}"
                        )
                        print(
                            f"{m:<15}{pa_m:>12.4f}{pf_m:>12.4f}"
                            f"{ca_m:>12.4f}{cf_m:>12.4f}{wmal_field}"
                        )

    rep_methods = [m for m in methods if m in {"weighted"}]
    if rep_methods and rep_logs:
        print("\nmethod,node_id,rep,R2,R3,R4 (weighted; last sweep combo)")
        last_key = sorted(rep_logs.keys())[-1]
        if last_key[4] in rep_methods:
            for nid, vals in sorted(rep_logs[last_key].items()):
                print(
                    f"{last_key[4]},{nid},{vals['rep']:.6f},{vals['R2']:.4f},{vals['R3']:.4f},{vals['R4']:.4f}"
                )

    if args.diag_path and diag_rows:
        import csv

        with open(args.diag_path, "w", newline="") as f:
            fieldnames = [
                "method",
                "round",
                "node_id",
                "is_malicious",
                "l2_to_global",
                "krum_score",
                "krum_selected",
                "bulyan_selected",
                "trimmed_flag",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in diag_rows:
                writer.writerow(row)

    # dump per-seed runs
    # default filenames if user didn't pass
    out_runs = args.out_runs or "runs.csv"
    out_summary = args.out_summary or "summary.csv"
    out_rounds = args.out_rounds or "rounds.csv"
    out_nodes = args.out_nodes or "nodes.csv"

    if out_runs and runs_rows:
        import csv

        with open(out_runs, "w", newline="") as f:
            fieldnames = [
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "seed",
                "polluted_acc",
                "polluted_f1",
                "clean_acc",
                "clean_f1",
                "w_mal",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in runs_rows:
                writer.writerow(row)

    # dump grouped summary mean/std
    if out_summary and summary:
        import csv

        with open(out_summary, "w", newline="") as f:
            fieldnames = [
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "polluted_acc_m",
                "polluted_acc_s",
                "polluted_f1_m",
                "polluted_f1_s",
                "clean_acc_m",
                "clean_acc_s",
                "clean_f1_m",
                "clean_f1_s",
                "w_mal_m",
                "w_mal_s",
                "count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (attack, level, dt_level, mal, m), vals in summary.items():
                writer.writerow(
                    {
                        "attack": attack,
                        "level": level,
                        "dt_level": dt_level,
                        "mal_nodes": mal,
                        "method": m,
                        "polluted_acc_m": vals["poll_acc_m"],
                        "polluted_acc_s": vals["poll_acc_s"],
                        "polluted_f1_m": vals["poll_f1_m"],
                        "polluted_f1_s": vals["poll_f1_s"],
                        "clean_acc_m": vals["clean_acc_m"],
                        "clean_acc_s": vals["clean_acc_s"],
                        "clean_f1_m": vals["clean_f1_m"],
                        "clean_f1_s": vals["clean_f1_s"],
                        "w_mal_m": vals["w_mal_m"],
                        "w_mal_s": vals["w_mal_s"],
                        "count": len(seeds),
                    }
                )

    # dump per-round mechanism logs (weighted only)
    if out_rounds and rounds_rows:
        import csv

        with open(out_rounds, "w", newline="") as f:
            fieldnames = [
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "seed",
                "round",
                "w_mal",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rounds_rows:
                writer.writerow(row)

    # dump per-node final scores (weighted only)
    if out_nodes and nodes_rows:
        import csv

        with open(out_nodes, "w", newline="") as f:
            fieldnames = [
                "attack",
                "level",
                "dt_level",
                "mal_nodes",
                "method",
                "seed",
                "round",
                "node_id",
                "is_malicious",
                "rep",
                "R2",
                "R3",
                "R4",
                "r4_kl",
                "r4_gate_hit",
                "tau_gate",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in nodes_rows:
                writer.writerow(row)

    # Plots are generated from CSV artifacts so you can re-draw without re-running:
    #   python plot_from_csv.py --runs runs.csv --summary summary.csv
    try:
        from dt_r4.plotting import make_plots_from_csv

        make_plots_from_csv(
            runs_csv=out_runs,
            summary_csv=out_summary,
            out_dir=".",
            attacks=attack_modes,
            methods=methods,
            mal_nodes=mal_nodes_list,
            dt_levels=dt_levels,
            label_flip_level="L1",
            num_nodes=C.NUM_NODES,
            metric="polluted_f1",
        )
    except Exception as e:
        print(f"[warn] plotting skipped: {e}")


if __name__ == "__main__":
    main()
