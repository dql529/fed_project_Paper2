# demo_r4_label_flip_teacher_agree.py
# ------------------------------------------------------------
# 目的：
# - 攻击：label flip（恶意客户端训练时翻转标签，自己本地loss依然能下降）
# - R4：用 server-side twin/teacher 在 reference 上的伪标签做“语义一致性”
#       agreement = mean(pred_client(Xref) == pred_teacher(Xref))
#       R4 = exp(-k_agree * (1 - agreement))
#       weights ∝ R4^beta
# - 对比：mean / median / trimmed-mean / r4w
# - 训练：clean (Xg split)
# - R4 reference：clean balanced ref (Xref)
# - 评估：clean (Xtest) + deploy shift+noise (Xdep)
# ------------------------------------------------------------

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dt_r4.config as C
from dt_r4.federated import js_divergence, confusion_trace

# -------------------------
# Repro
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Data: 2D Gaussian binary
# -------------------------
def make_gaussian(n, shift=(0.0, 0.0), noise=1.0):
    c0 = np.random.randn(n // 2, 2) * noise + np.array([-1.0, -1.0]) + np.array(shift)
    c1 = (
        np.random.randn(n - n // 2, 2) * noise
        + np.array([+1.0, +1.0])
        + np.array(shift)
    )
    X = np.vstack([c0, c1]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def to_tensor(X, y):
    return torch.tensor(X, device=device), torch.tensor(y, device=device)


def split_noniid(X, y, n_clients=10, alpha=0.2):
    y = np.asarray(y)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    props = np.random.dirichlet([alpha, alpha], size=n_clients)
    client_idx = [[] for _ in range(n_clients)]

    p0 = props[:, 0] / props[:, 0].sum()
    cuts0 = (np.cumsum(p0) * len(idx0)).astype(int)
    s = 0
    for i, e in enumerate(cuts0):
        client_idx[i].extend(idx0[s:e].tolist())
        s = e

    p1 = props[:, 1] / props[:, 1].sum()
    cuts1 = (np.cumsum(p1) * len(idx1)).astype(int)
    s = 0
    for i, e in enumerate(cuts1):
        client_idx[i].extend(idx1[s:e].tolist())
        s = e

    return [np.array(ci, dtype=np.int64) for ci in client_idx]


# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def eval_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
        return float((pred == y).float().mean().item())


def eval_loss(model, X, y):
    model.eval()
    with torch.no_grad():
        return float(F.cross_entropy(model(X), y).item())


def get_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_params(model, vec):
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[off : off + n].view_as(p))
        off += n


def _extract_delta(vec, vec0):
    return vec - vec0


def per_dim_clamp_malicious_delta(
    param_list, vec0, mal_ids, z: float = 1.5, eps: float = 1e-12
):
    """
    Clamp malicious *deltas* (vec - vec0) into a per-dimension IQR band
    estimated from benign deltas, then reconstruct vec = vec0 + delta.
    """
    if not param_list:
        return param_list

    # stack deltas
    deltas = torch.stack([v - vec0 for v in param_list], dim=0)  # [n,d]

    mal_mask = torch.tensor(
        [i in mal_ids for i in range(len(param_list))],
        device=deltas.device,
        dtype=torch.bool,
    )
    if not mal_mask.any():
        return param_list

    benign = deltas[~mal_mask]
    if benign.shape[0] < 2:
        return param_list

    q1 = benign.quantile(0.25, dim=0)
    q3 = benign.quantile(0.75, dim=0)
    iqr = (q3 - q1).clamp_min(eps)

    lo = q1 - z * iqr
    hi = q3 + z * iqr

    out = []
    for i, v in enumerate(param_list):
        if i in mal_ids:
            d = v - vec0
            d2 = torch.max(torch.min(d, hi), lo)
            out.append(vec0 + d2)
        else:
            out.append(v)
    return out


# -------------------------
# Aggregators
# -------------------------
def mean_agg(param_list):
    return torch.stack(param_list).mean(0)


def coord_median_agg(param_list):
    return torch.stack(param_list).median(0).values


def trimmed_mean_agg(param_list, trim_ratio=0.2):
    stacked = torch.stack(param_list)  # [n,d]
    n = stacked.shape[0]
    k = int(math.floor(trim_ratio * n))
    sorted_vals, _ = stacked.sort(dim=0)
    kept = sorted_vals[k : n - k] if (n - 2 * k) > 0 else sorted_vals
    return kept.mean(0)


def r4_weighted_agg(param_list, r4_scores, beta=4.0, eps=1e-12):
    w = torch.tensor([max(eps, float(s)) ** beta for s in r4_scores], device=device)
    w = w / w.sum()
    out = (w.view(-1, 1) * torch.stack(param_list)).sum(0)
    return out, w


# -------------------------
# Local training + label flip attack
# -------------------------
def flip_labels_binary(y: torch.Tensor, flip_ratio: float = 1.0) -> torch.Tensor:
    """Binary label flip: y in {0,1} -> 1-y, optionally only on a subset."""
    if flip_ratio <= 0:
        return y
    y2 = y.clone()
    if flip_ratio >= 1.0:
        return 1 - y2
    mask = torch.rand_like(y2.float()) < float(flip_ratio)
    y2[mask] = 1 - y2[mask]
    return y2


def local_train(model, X, y, lr=0.05, epochs=1):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(X), y).backward()
        opt.step()


# -------------------------
# Twin prior: pretrain a benign "teacher" once (centralized on clean)
# -------------------------
def train_twin(Xg_t, yg_t, epochs=60, lr=0.05):
    twin = MLP().to(device)
    opt = torch.optim.SGD(twin.parameters(), lr=lr)
    for _ in range(epochs):
        twin.train()
        opt.zero_grad()
        loss = F.cross_entropy(twin(Xg_t), yg_t)
        loss.backward()
        opt.step()
    return twin


# -------------------------
# R4 / reputation (reuse full version semantics)
# -------------------------
@torch.no_grad()
def build_teacher_ref(teacher: MLP, Xref_t: torch.Tensor):
    logits = teacher(Xref_t)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    mask = torch.ones_like(conf, dtype=torch.bool)
    if getattr(C, "R4_USE_ONLY_CONFIDENT", True):
        mask &= conf >= getattr(C, "R4_CONF_THRESH", 0.0)
    if getattr(C, "R4_SEMANTIC_ONLY_NORMAL", False):
        mask &= pred == getattr(C, "NORMAL_CLASS_INDEX", 0)

    weights = conf * mask.float()
    return probs.detach(), pred.detach(), mask.detach(), weights.detach()


def compute_r4_teacher(student_logits: torch.Tensor, ref_ctx):
    """
    Teacher-student semantic consistency (JS + optional confusion trace),
    aligned with dt_r4/federated.py.
    """
    teacher_probs_ref, teacher_preds_ref, teacher_mask, weights = ref_ctx

    softmax = nn.Softmax(dim=1)
    student_probs = softmax(student_logits)

    if teacher_mask is not None:
        teacher_probs = teacher_probs_ref[teacher_mask]
        teacher_preds = teacher_preds_ref[teacher_mask]
        student_probs = student_probs[teacher_mask]
        w = weights[teacher_mask] if weights is not None else None
    else:
        teacher_probs = teacher_probs_ref
        teacher_preds = teacher_preds_ref
        w = weights

    if student_probs.numel() == 0:
        return 0.5

    js_vals = js_divergence(teacher_probs, student_probs)
    if w is not None and w.numel() == js_vals.numel() and w.sum().item() > 1e-12:
        js_mean = float((w * js_vals).sum().item() / float(w.sum().item()))
    else:
        js_mean = float(js_vals.mean().item())
    r_js = math.exp(-float(getattr(C, "R4_JS_ALPHA", 6.0)) * js_mean)

    mix = float(getattr(C, "R4_CONFUSION_MIX", 0.0))
    if mix > 0:
        student_preds = student_probs.argmax(dim=1)
        trace_val = confusion_trace(
            student_preds, teacher_preds, num_classes=student_probs.shape[1]
        )
        if not math.isnan(trace_val):
            r_conf = math.exp(
                -float(getattr(C, "R4_CONFUSION_BETA", 8.0)) * (1.0 - trace_val)
            )
            r4 = (1.0 - mix) * r_js + mix * r_conf
        else:
            r4 = r_js
    else:
        r4 = r_js
    return float(max(0.0, min(1.0, r4)))


# -------------------------
# Main
# -------------------------
def main():
    # ---- knobs ----
    n_clients = 10
    mal_ids = {0, 1, 2}  # 按你之前诉求：不改恶意数量（3/10）
    rounds = 30

    # Non-IID：为了让 label flip 更“像正常异质”，建议更小一点
    dirichlet_alpha = 0.05  # 0.2 太温和；0.05 更容易让 median/trim 被误导

    local_lr = 0.05
    local_epochs = 1

    # Label flip attack strength
    flip_ratio = 1.0  # 1.0=全翻；想更隐蔽可用 0.5

    k_loss = 3.0  # 1.0~5.0 常用；太大容易权重塌缩
    beta = 6.0
    tau_r4 = 0.0  # 先关硬门控

    # Data
    Xg, yg = make_gaussian(6000, shift=(0.0, 0.0), noise=1.0)  # train (clients)
    Xref, yref = make_gaussian(
        2000, shift=(0.8, 0.8), noise=1.4
    )  # R4 reference (deploy-like shift+noise, used for loss_ref)
    Xtest, ytest = make_gaussian(2000, shift=(0.0, 0.0), noise=1.0)  # clean eval
    Xdep, ydep = make_gaussian(2000, shift=(0.8, 0.8), noise=1.4)

    Xg_t, yg_t = to_tensor(Xg, yg)
    Xref_t, yref_t = to_tensor(Xref, yref)
    Xtest_t, ytest_t = to_tensor(Xtest, ytest)
    Xdep_t, ydep_t = to_tensor(Xdep, ydep)

    # teacher reference for semantic R4
    teacher = train_twin(Xg_t, yg_t, epochs=60, lr=0.05)
    teacher_ref_ctx = build_teacher_ref(teacher, Xref_t)

    splits = split_noniid(Xg, yg, n_clients=n_clients, alpha=dirichlet_alpha)

    # Maintain separate global models per aggregator (fair)
    methods = ["mean", "median", "trim20", "r4w"]
    from collections import defaultdict

    hist = {m: defaultdict(list) for m in methods}

    global_vec = {}
    for m in methods:
        init = MLP().to(device)
        global_vec[m] = get_params(init).detach().clone()

    # Optional per-round print
    LOG_EVERY = 0  # set 5/10 if you want traces

    if LOG_EVERY:
        print("round,method,clean_acc,deploy_acc,W_mal_weight")

    for r in range(1, rounds + 1):
        ids = list(range(n_clients))

        for method in methods:
            local_params = []
            r4_scores = []

            # local training
            for cid in ids:
                mloc = MLP().to(device)
                set_params(mloc, global_vec[method])

                idx = splits[cid]
                Xc, yc = to_tensor(Xg[idx], yg[idx])

                eta = 0.5  # 0.3~0.6
                proj_norm = 0.8  # 0.6~1.0

                vec0 = global_vec[method].detach()

                if cid in mal_ids:
                    # train clean branch
                    m_clean = MLP().to(device)
                    set_params(m_clean, vec0)
                    local_train(m_clean, Xc, yc, lr=local_lr, epochs=local_epochs)
                    vec_clean = get_params(m_clean).detach()

                    # train flip branch
                    m_flip = MLP().to(device)
                    set_params(m_flip, vec0)
                    yc_flip = flip_labels_binary(yc, flip_ratio=1.0)  # 全翻
                    local_train(m_flip, Xc, yc_flip, lr=local_lr, epochs=local_epochs)
                    vec_flip = get_params(m_flip).detach()

                    # blend update
                    delta = (1 - eta) * (vec_clean - vec0) + eta * (vec_flip - vec0)

                    # optional L2 clip
                    nrm = float(delta.norm().item())
                    if nrm > proj_norm:
                        delta = delta * (proj_norm / (nrm + 1e-12))

                    vec_mal = vec0 + delta
                    local_params.append(vec_mal)

                    if method == "r4w":
                        tmp = MLP().to(device)
                        set_params(tmp, vec_mal)
                        # R4 = exp(-k_loss * loss_ref)
                        loss_ref = eval_loss(
                            tmp, Xref_t, yref_t
                        )  # 注意：用 yref_t 的真标签
                        r4 = math.exp(-k_loss * loss_ref)

                        if tau_r4 > 0 and r4 < tau_r4:
                            r4 = 0.0
                        r4_scores.append(r4)

                else:
                    mloc = MLP().to(device)
                    set_params(mloc, vec0)
                    local_train(mloc, Xc, yc, lr=local_lr, epochs=local_epochs)
                    vec = get_params(mloc).detach()
                    local_params.append(vec)

                    if method == "r4w":
                        loss_ref = eval_loss(mloc, Xref_t, yref_t)
                        r4 = math.exp(-k_loss * loss_ref)
                        if tau_r4 > 0 and r4 < tau_r4:
                            r4 = 0.0
                        r4_scores.append(r4)

            # ---- NEW: per-dim in-range clamp on malicious updates (to beat trim20) ----
            z_inrange = 1.5  # 1.5~2.0; smaller => more stealth vs trim
            local_params = per_dim_clamp_malicious_delta(
                local_params, global_vec[method], mal_ids, z=z_inrange
            )

            # ---- NEW: compute reputation-based scores AFTER clamp ----
            if method == "r4w":
                r4_scores = []
                beta_rep = float(getattr(C, "R4_ONLY_BETA", 10.0))
                eps = 1e-12
                for i, vec_i in enumerate(local_params):
                    tmp = MLP().to(device)
                    set_params(tmp, vec_i)
                    logits_ref = tmp(Xref_t)
                    r4 = compute_r4_teacher(logits_ref, teacher_ref_ctx)
                    if tau_r4 > 0 and r4 < tau_r4:
                        r4 = 0.0
                    # reputation mapping (R4-only branch from full code)
                    rep = math.exp(beta_rep * (float(r4) - 0.5))
                    # feed pre-rooted so that aggregator's ^beta recovers rep scale
                    r4_scores.append(max(eps, rep) ** (1.0 / beta))

            # aggregate
            W_mal_weight = float("nan")

            if method == "mean":
                new_vec = mean_agg(local_params)
            elif method == "median":
                new_vec = coord_median_agg(local_params)
            elif method == "trim20":
                new_vec = trimmed_mean_agg(local_params, trim_ratio=0.2)
            elif method == "r4w":
                new_vec, w = r4_weighted_agg(local_params, r4_scores, beta=beta)
                W_mal_weight = float(
                    sum(w[i].item() for i, cid in enumerate(ids) if cid in mal_ids)
                )
            else:
                raise ValueError("unknown method")

            global_vec[method] = new_vec.detach()

            # eval
            mtmp = MLP().to(device)
            set_params(mtmp, global_vec[method])
            ca = eval_acc(mtmp, Xtest_t, ytest_t)
            da = eval_acc(mtmp, Xdep_t, ydep_t)

            hist[method]["clean"].append(ca)
            hist[method]["dep"].append(da)
            hist[method]["drop"].append(ca - da)
            if method == "r4w":
                hist[method]["W_mal"].append(W_mal_weight)

            if LOG_EVERY and (r % LOG_EVERY == 0 or r == 1 or r == rounds):
                print(
                    f"{r:02d},{method},{ca:.4f},{da:.4f},{W_mal_weight if method=='r4w' else 'nan'}"
                )

    # summary
    def _best(v):
        v = list(v)
        i = int(np.argmax(v))
        return float(v[i]), i + 1

    print("\n=== Final Summary (single run) ===")
    header = f"{'method':<7} {'clean@final':>10} {'dep@final':>10} {'drop':>8} {'dep@best':>10} {'best_r':>6} {'W_mal@final':>11}"
    print(header)
    print("-" * len(header))

    final_dep_baselines = []
    for method in methods:
        clean_final = float(hist[method]["clean"][-1])
        dep_final = float(hist[method]["dep"][-1])
        drop_final = clean_final - dep_final
        dep_best, best_r = _best(hist[method]["dep"])

        if method != "r4w":
            final_dep_baselines.append(dep_final)

        if method == "r4w":
            wmal_final = (
                float(hist[method]["W_mal"][-1])
                if hist[method]["W_mal"]
                else float("nan")
            )
            print(
                f"{method:<7} {clean_final:>10.4f} {dep_final:>10.4f} {drop_final:>8.4f} {dep_best:>10.4f} {best_r:>6d} {wmal_final:>11.4f}"
            )
        else:
            print(
                f"{method:<7} {clean_final:>10.4f} {dep_final:>10.4f} {drop_final:>8.4f} {dep_best:>10.4f} {best_r:>6d} {'-':>11}"
            )

    r4_dep_final = float(hist["r4w"]["dep"][-1])
    best_base_dep_final = (
        max(final_dep_baselines) if final_dep_baselines else float("nan")
    )
    gain = r4_dep_final - best_base_dep_final
    print(
        f"\nR4 gain vs best baseline (deploy@final): {r4_dep_final:.4f} - {best_base_dep_final:.4f} = {gain:+.4f}"
    )


if __name__ == "__main__":
    main()
