import torch
import numpy as np

torch.manual_seed(0)
import collections
from torch import nn


def weighted_average_aggregation(
    models_to_aggregate, reputations, top_k_ratio=0.7, alpha=3
):
    """
    先用 Fed-MAE 思路选出 MAE 最小的 top_k_ratio 模型，
    再在这 subset 上按 reputations 做非线性加权平均。
    """

    # —— 1. 收集所有 state_dict 并计算 MAE 分数 ——
    model_states = [list(m.values())[0].state_dict() for m in models_to_aggregate]
    n = len(model_states)
    if n == 0:
        return {}
    keys = model_states[0].keys()

    mae_scores = []
    for i in range(n):
        diff_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            for k in keys:
                diff_sum += torch.mean(
                    torch.abs(model_states[i][k] - model_states[j][k])
                ).item()
        mae_scores.append(diff_sum / (n - 1))

    top_k = max(1, int(n * top_k_ratio))
    top_indices = sorted(range(n), key=lambda i: mae_scores[i])[:top_k]
    selected_models = [models_to_aggregate[i] for i in top_indices]

    # —— 2. 过滤声誉为0的模型 ——
    filtered = [
        m for m in selected_models if reputations.get(list(m.keys())[0], 0.0) > 0.0
    ]
    # 如果全被过滤掉，就退回到原始 selected_models
    if not filtered:
        filtered = selected_models

    # —— 2. 在 selected_models 上按声誉加权平均 ——
    # 先计算分母
    total_rep = (
        sum(reputations.get(list(m.keys())[0], 0.0) ** alpha for m in selected_models)
        or 1.0
    )

    aggregated = {}
    for m in selected_models:
        drone_id, lm = next(iter(m.items()))
        w = reputations.get(drone_id, 0.0) ** alpha / total_rep
        for k, v in lm.state_dict().items():
            aggregated[k] = aggregated.get(k, 0.0) + v * w

    return aggregated


def average_aggregation(models_to_aggregate):
    torch.manual_seed(0)
    print(" average_aggregation used")
    num_models = len(models_to_aggregate)

    aggregated_model = {}

    for model_dict in models_to_aggregate:
        for _, local_model in model_dict.items():
            if not aggregated_model:
                aggregated_model = {
                    k: v / num_models for k, v in local_model.state_dict().items()
                }
            else:
                for k, v in local_model.state_dict().items():
                    aggregated_model[k] += v / num_models

    return aggregated_model


import torch


def fed_mae_aggregation(models_to_aggregate):
    """
    正确的 MAE 聚合：对每个参数 k，
      aggregated[k] = median_i ( w_i[k] )
    即对每个坐标取所有客户端模型权重的中位数，
    对应最小化 L1 范数距离的鲁棒聚合。
    """

    N = len(models_to_aggregate)
    if N == 0:
        return {}

    # 拿到每个模型的 state_dict
    states = [list(m.values())[0].state_dict() for m in models_to_aggregate]
    keys = states[0].keys()

    # 初始化 aggregated
    aggregated = {}

    # 对每个参数逐元素取中位数
    for k in keys:
        # 将每个模型对应参数取出来堆叠成一个tensor
        stacked = torch.stack([state[k].float() for state in states], dim=0)
        # 在第0维上取中位数
        median_val = torch.median(stacked, dim=0).values
        # 保持原dtype
        aggregated[k] = median_val.to(dtype=states[0][k].dtype)

    return aggregated


def median_aggregation(models_to_aggregate):
    """
    对每个参数坐标直接取中位数，中位数对极端值几乎完全免疫。
    """
    print("median_aggregation used")
    states = [list(m.values())[0].state_dict() for m in models_to_aggregate]
    M = len(states)
    if M == 0:
        return {}
    keys = states[0].keys()

    aggregated = {}
    for k in keys:
        stacked = torch.stack([s[k] for s in states], dim=0)
        # median 返回 (values, indices)
        aggregated[k] = stacked.median(dim=0).values
    return aggregated


# def fed_mae_aggregation(models_to_aggregate):
#     """
#     纯 MAE 聚合：
#       对每个参数 k：
#         aggregated[k] = (1 / (N*(N-1))) * sum_{i≠j} | w_i[k] - w_j[k] |
#     """
#     N = len(models_to_aggregate)
#     if N == 0:
#         return {}

#     # 1. 提取所有模型 state_dict
#     states = [list(m.values())[0].state_dict() for m in models_to_aggregate]
#     keys = states[0].keys()

#     # 2. 初始化聚合结果
#     aggregated = {k: torch.zeros_like(states[0][k]) for k in keys}

#     # 3. 累加所有模型对之间的 MAE
#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 continue
#             for k in keys:
#                 aggregated[k] += torch.abs(states[i][k] - states[j][k])

#     # 4. 平均：除以 N*(N-1)
#     factor = 1.0 / (N * (N - 1))
#     for k in keys:
#         aggregated[k] *= factor

#     return aggregated


def trimmed_mean_aggregation(models_to_aggregate, trim_ratio=0.1):
    """
    对每个参数坐标，去掉最高 trim_ratio 和最低 trim_ratio 的值，再对剩下的平均。
    trim_ratio 推荐在 0.05～0.2 之间微调。
    """
    states = [list(m.values())[0].state_dict() for m in models_to_aggregate]
    M = len(states)
    if M == 0:
        return {}
    keys = states[0].keys()
    trim = int(M * trim_ratio)

    aggregated = {}
    for k in keys:
        # stack 后形状 [M, *param_shape]
        stacked = torch.stack([s[k] for s in states], dim=0)
        # 排序
        sorted_vals, _ = torch.sort(stacked, dim=0)
        # 去掉最小 trim 个和最大 trim 个
        trimmed = sorted_vals[trim : M - trim] if M > 2 * trim else sorted_vals
        # 均值
        aggregated[k] = trimmed.mean(dim=0)
    return aggregated


# ========== Robust Aggregators: Krum / Multi-Krum / Bulyan ==========
def krum_aggregation(models_to_aggregate, f=0):
    """
    Krum：选择一个最“接近多数”的模型直接作为聚合结果。
    返回: state_dict（和你原有加载方式兼容）
    约定:
      - models_to_aggregate: 列表[{drone_id: nn.Module}]
      - 若 n <= 2f + 2: 退化到 median_aggregation(models_to_aggregate)
    """
    # 取出各模型的 state_dict
    try:
        states_raw = [next(iter(m.values())).state_dict() for m in models_to_aggregate]
    except Exception as e:
        raise TypeError(
            f"[krum] 非预期的输入格式，期望 {{id: nn.Module}}，实际: {type(models_to_aggregate)}"
        ) from e

    n = len(states_raw)
    if n == 0:
        return {}  # 与既有实现保持一致

    if n <= 2 * f + 2:
        return median_aggregation(models_to_aggregate)

    # 内联: 统一到 CPU/float 以做距离
    def _to_cpu_float(sd):
        return collections.OrderedDict(
            (k, v.detach().to("cpu", dtype=torch.float32)) for k, v in sd.items()
        )

    def _flatten(sd_cpu_float):
        return torch.cat([v.view(-1) for v in sd_cpu_float.values()])

    def _pairwise(states_cpu_float):
        vecs = [_flatten(sd) for sd in states_cpu_float]
        mat = torch.stack(vecs, dim=0)  # [n, D] on CPU
        sq = (mat * mat).sum(dim=1, keepdim=True)  # [n,1]
        d2 = sq + sq.t() - 2.0 * (mat @ mat.t())  # [n,n]
        d2.clamp_(min=0)
        return torch.sqrt(d2 + 1e-12)

    states_cpu = [_to_cpu_float(sd) for sd in states_raw]
    dists = _pairwise(states_cpu).numpy()

    nb_in_score = max(1, n - f - 2)  # paper 建议
    scores = []
    for i in range(n):
        s = np.sort(dists[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(s)
    winner = int(np.argmin(scores))

    # 将 CPU/float 的 winner 结果按第一个模型的 dtype/device 还原
    ref = states_raw[0]
    out = collections.OrderedDict()
    for k in ref.keys():
        out[k] = states_cpu[winner][k].to(dtype=ref[k].dtype, device=ref[k].device)
    return out


def multi_krum_aggregation(models_to_aggregate, f=0, m=None):
    """
    Multi-Krum：选出 m 个“好”模型，再逐坐标均值。
    约定:
      - models_to_aggregate: 列表[{drone_id: nn.Module}]
      - m 默认 n - f - 2（至少 1）
      - 若 n <= 2f + 2: 退化到 median_aggregation
    """
    try:
        states_raw = [next(iter(m.values())).state_dict() for m in models_to_aggregate]
    except Exception as e:
        raise TypeError(f"[multi_krum] 非预期的输入格式，期望 {{id: nn.Module}}") from e

    n = len(states_raw)
    if n == 0:
        return {}

    if m is None:
        m = max(1, n - f - 2)
    m = max(1, min(m, n))  # 边界保护

    if n <= 2 * f + 2:
        return median_aggregation(models_to_aggregate)

    def _to_cpu_float(sd):
        return collections.OrderedDict(
            (k, v.detach().to("cpu", dtype=torch.float32)) for k, v in sd.items()
        )

    def _flatten(sd_cpu_float):
        return torch.cat([v.view(-1) for v in sd_cpu_float.values()])

    def _pairwise(states_cpu_float):
        vecs = [_flatten(sd) for sd in states_cpu_float]
        mat = torch.stack(vecs, dim=0)
        sq = (mat * mat).sum(dim=1, keepdim=True)
        d2 = sq + sq.t() - 2.0 * (mat @ mat.t())
        d2.clamp_(min=0)
        return torch.sqrt(d2 + 1e-12)

    states_cpu = [_to_cpu_float(sd) for sd in states_raw]
    dists = _pairwise(states_cpu).numpy()

    nb_in_score = max(1, n - f - 2)
    scores = []
    for i in range(n):
        s = np.sort(dists[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(s)
    winners = np.argsort(scores)[:m].tolist()

    # 在 winners 上逐坐标均值（CPU/float 上做，再还原 dtype/device）
    keys = list(states_cpu[0].keys())
    agg_cpu = {}
    for k in keys:
        stacked = torch.stack([states_cpu[i][k] for i in winners], dim=0)
        agg_cpu[k] = stacked.mean(dim=0)

    ref = states_raw[0]
    out = collections.OrderedDict(
        (k, agg_cpu[k].to(dtype=ref[k].dtype, device=ref[k].device)) for k in keys
    )
    return out


def bulyan_aggregation(models_to_aggregate, f=0):
    """
    Bulyan：先 Multi-Krum 选出 m = n - 2f，再在该集合上做逐坐标 trimmed-mean（去掉 f 个最大/最小）。
    约定:
      - models_to_aggregate: 列表[{drone_id: nn.Module}]
      - 若 n <= 4f + 3: 退化到 trimmed_mean_aggregation(models_to_aggregate, ...)
    """
    try:
        states_raw = [next(iter(m.values())).state_dict() for m in models_to_aggregate]
    except Exception as e:
        raise TypeError(f"[bulyan] 非预期的输入格式，期望 {{id: nn.Module}}") from e

    n = len(states_raw)
    if n == 0:
        return {}

    if n <= 4 * f + 3:
        # 保持你原有退化路径
        return trimmed_mean_aggregation(
            models_to_aggregate, trim_ratio=min(0.1, (f / max(1, n)))
        )

    def _to_cpu_float(sd):
        return collections.OrderedDict(
            (k, v.detach().to("cpu", dtype=torch.float32)) for k, v in sd.items()
        )

    def _flatten(sd_cpu_float):
        return torch.cat([v.view(-1) for v in sd_cpu_float.values()])

    def _pairwise(states_cpu_float):
        vecs = [_flatten(sd) for sd in states_cpu_float]
        mat = torch.stack(vecs, dim=0)
        sq = (mat * mat).sum(dim=1, keepdim=True)
        d2 = sq + sq.t() - 2.0 * (mat @ mat.t())
        d2.clamp_(min=0)
        return torch.sqrt(d2 + 1e-12)

    states_cpu = [_to_cpu_float(sd) for sd in states_raw]
    dists = _pairwise(states_cpu).numpy()

    # 1) 选出集合 winners（大小 m = n - 2f）
    m = max(1, n - 2 * f)
    nb_in_score = max(1, n - f - 2)
    scores = []
    for i in range(n):
        s = np.sort(dists[i][np.arange(n) != i])[:nb_in_score].sum()
        scores.append(s)
    winners = np.argsort(scores)[:m].tolist()

    # 2) 在 winners 上做逐坐标 trimmed-mean（去掉 f 个最大与 f 个最小）
    keys = list(states_cpu[0].keys())
    agg_cpu = {}
    for k in keys:
        stacked = torch.stack([states_cpu[i][k] for i in winners], dim=0)  # [m, *]
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if m > 2 * f:
            trimmed = sorted_vals[f : m - f]
        else:
            trimmed = sorted_vals
        agg_cpu[k] = trimmed.mean(dim=0)

    ref = states_raw[0]
    out = collections.OrderedDict(
        (k, agg_cpu[k].to(dtype=ref[k].dtype, device=ref[k].device)) for k in keys
    )
    return out
