"""
dt_r4/federated.py

联邦模拟核心组件：
- DroneNode：本地训练 + 恶意攻击（可选）
- CentralServer：声誉计算 + 聚合 + R4 一致性
"""

from __future__ import annotations
import dt_r4.config as C
from dt_r4.data import (
    build_noise_variants_fixed,
    get_teacher_model,
    load_node_splits,
    load_reference_data,
    SimpleData,
)
import copy
import math
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from .config import (
    USE_PERF_PENALTY,
    PERF_THRESH,
    PERF_PENALTY_FACTOR,
)
from .runtime import device
from .models import StudentNet
from .utils import weighted_average_aggregation, sigmoid, exponential_decay
from .twin import make_r4_mask_and_weights


def apply_backdoor_trigger(
    x: torch.Tensor,
    trigger_idx: List[int],
    *,
    mode: str = "add",
    scale: float = 3.0,
    value: Optional[float] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply a simple backdoor trigger to a batch.
    mode="add": add scale * per-feature std (value ignored)
    mode="set": set features to `value` if given else to `scale`.
    """
    if x is None:
        return x
    if mask is None:
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    if not mask.any():
        return x

    idx = [int(i) for i in trigger_idx if 0 <= int(i) < x.shape[1]]
    if not idx:
        return x

    x_out = x.clone()
    sub = x_out[mask][:, idx]

    mode = str(mode or "add").lower()
    if mode == "set":
        fill_val = value if value is not None else scale
        sub = torch.full_like(sub, float(fill_val))
    else:
        std = torch.std(x_out[:, idx], dim=0, unbiased=False)
        std = torch.clamp(std, min=1e-6)
        sub = sub + float(scale) * std

    if clip_min is not None or clip_max is not None:
        lo = -float("inf") if clip_min is None else float(clip_min)
        hi = float("inf") if clip_max is None else float(clip_max)
        sub = torch.clamp(sub, min=lo, max=hi)

    x_out[mask][:, idx] = sub
    return x_out


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence per-sample (assumes p,q are probs)."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def confusion_trace(
    student_preds: torch.Tensor, teacher_preds: torch.Tensor, num_classes: int
) -> float:
    """Per-class agreement (trace of normalized confusion) averaged across classes present."""
    if student_preds.numel() == 0 or teacher_preds.numel() == 0:
        return float("nan")
    traces = []
    for c in range(num_classes):
        mask = teacher_preds == c
        if mask.any():
            acc = (student_preds[mask] == c).float().mean()
            traces.append(acc)
    if not traces:
        return float("nan")
    return float(torch.stack(traces).mean().item())


class DroneNode:
    def __init__(
        self,
        drone_id: int,
        is_malicious: bool = False,
        num_epochs: int = 1,
        learning_rate: float = 0.01,
        attack_mode: Optional[str] = None,
        distill_mode: str = "none",
        distill_alpha: float = 0.0,
        distill_temp: float = 1.0,
        distill_apply_to_malicious: bool = False,
        # legacy stealth_amp
        warmup_rounds: Optional[int] = None,
        max_amp: Optional[float] = None,
        amp_step: Optional[float] = None,
        noise_base: Optional[float] = None,
        noise_step: Optional[float] = None,
        label_flip_ratio: Optional[float] = None,
        label_flip_warmup: Optional[int] = None,
        label_flip_epochs: Optional[int] = None,
        label_flip_lr: Optional[float] = None,
        label_flip_pick_strategy: Optional[str] = None,
    ):
        self.drone_id = drone_id
        self.is_malicious = is_malicious
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.attack_mode = (
            str(attack_mode)
            if attack_mode is not None
            else str(getattr(C, "MAL_ATTACK_MODE", "none"))
        )
        self.distill_mode = str(distill_mode or "none").lower()
        self.distill_alpha = float(distill_alpha)
        self.distill_temp = float(distill_temp)
        self.distill_apply_to_malicious = bool(distill_apply_to_malicious)
        self.distill_logits: Optional[torch.Tensor] = None

        self.data = None
        self.test_data = None
        self.local_model: Optional[StudentNet] = None

        self.round_idx = 0

        # legacy stealth_amp params
        self.warmup_rounds = int(
            warmup_rounds
            if warmup_rounds is not None
            else getattr(C, "STEALTH_WARMUP_ROUNDS", 1)
        )
        self.max_amp = float(
            max_amp if max_amp is not None else getattr(C, "STEALTH_MAX_AMP", 0.2)
        )
        self.amp_step = float(
            amp_step if amp_step is not None else getattr(C, "STEALTH_AMP_STEP", 0.05)
        )
        self.noise_base = float(
            noise_base
            if noise_base is not None
            else getattr(C, "STEALTH_NOISE_BASE", 0.0)
        )
        self.noise_step = float(
            noise_step
            if noise_step is not None
            else getattr(C, "STEALTH_NOISE_STEP", 0.0)
        )
        self.ref_model_for_attack: Optional[StudentNet] = None
        self.label_flip_ratio = float(
            label_flip_ratio
            if label_flip_ratio is not None
            else getattr(C, "LABEL_FLIP_RATIO", 0.0)
        )
        self.label_flip_warmup = int(
            label_flip_warmup
            if label_flip_warmup is not None
            else getattr(C, "LABEL_FLIP_WARMUP_ROUNDS", 0)
        )
        self.label_flip_epochs = int(
            label_flip_epochs
            if label_flip_epochs is not None
            else getattr(C, "LABEL_FLIP_EPOCHS", 1)
        )
        self.label_flip_lr = float(
            label_flip_lr
            if label_flip_lr is not None
            else getattr(C, "LABEL_FLIP_LR", 0.01)
        )
        self.label_flip_pick_strategy = str(
            label_flip_pick_strategy
            if label_flip_pick_strategy is not None
            else getattr(C, "LABEL_FLIP_PICK_STRATEGY", "best")
        )
        # optional noise-ascent attack (malicious only)
        self.noise_attack_data: Optional[SimpleData] = None
        self.noise_ascent_beta: float = float(getattr(C, "NOISE_ASCENT_BETA", 0.0))

    def receive_global_model(self, global_model: StudentNet):
        self.local_model = copy.deepcopy(global_model).to(device)
        self.ref_model_for_attack = copy.deepcopy(global_model).to(device)

    def train(self):
        if self.local_model is None or self.data is None or self.test_data is None:
            return

        x_train = self.data.x
        y_train = self.data.y
        if self.is_malicious and self.attack_mode == "backdoor_trigger":
            x_train, y_train = self._apply_backdoor_trigger(x_train, y_train)
        if self.is_malicious and self.attack_mode == "label_flip":
            y_train = self._flip_labels(y_train)

        epochs = self.num_epochs
        lr = self.learning_rate
        pick_strategy = "best"
        if self.is_malicious and self.attack_mode == "label_flip":
            if self.label_flip_epochs > 0:
                epochs = self.label_flip_epochs
            if self.label_flip_lr > 0:
                lr = self.label_flip_lr
            pick_strategy = str(self.label_flip_pick_strategy).lower()

        pick_best = pick_strategy == "best"
        pick_worst = pick_strategy == "worst"

        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        noise_steps = max(1, int(getattr(C, "NOISE_ASCENT_STEPS", 1)))
        noise_std_scale = float(getattr(C, "NOISE_STD_SCALE", 0.5))
        noise_dropout_p = float(getattr(C, "NOISE_DROPOUT_P", 0.5))
        noise_fgsm_eps = float(getattr(C, "NOISE_FGSM_EPS", 0.0))

        best_acc = -1.0
        best_sd = None
        worst_acc = float("inf")
        worst_sd = None

        for _ in range(epochs):
            self.local_model.train()
            optimizer.zero_grad()
            out = self.local_model(x_train)
            ce_loss = criterion(out, y_train)
            # malicious-only: push noise domain bad (gradient ascent on noise CE)
            if self.is_malicious and self.noise_ascent_beta > 0:
                for _ in range(noise_steps):
                    if self.noise_attack_data is not None:
                        noise_x = self.noise_attack_data.x
                        noise_y = self.noise_attack_data.y
                    else:
                        # on-the-fly noisy clone of clean batch (no extra dataset)
                        noise_x = x_train.detach()
                        std = torch.std(noise_x, dim=0, unbiased=False).clamp(min=1e-6)
                        noise_x = noise_x + noise_std_scale * std * torch.randn_like(
                            noise_x
                        )
                        if noise_dropout_p > 0:
                            dropout_mask = torch.rand_like(noise_x) < noise_dropout_p
                            noise_x = noise_x.masked_fill(dropout_mask, 0.0)
                        noise_y = y_train.detach()
                    if noise_x is not None and noise_y is not None:
                        noise_out = self.local_model(noise_x)
                        ce_noise = criterion(noise_out, noise_y)
                        # optional FGSM-style perturbation on noise_x to amplify damage
                        if noise_fgsm_eps > 0:
                            noise_x.requires_grad_(True)
                            noise_out_adv = self.local_model(noise_x)
                            ce_adv = criterion(noise_out_adv, noise_y)
                            grad = torch.autograd.grad(
                                ce_adv, noise_x, retain_graph=False, create_graph=False
                            )[0]
                            noise_x_adv = noise_x + noise_fgsm_eps * grad.sign()
                            noise_x = noise_x_adv.detach()
                            noise_out = self.local_model(noise_x)
                            ce_noise = criterion(noise_out, noise_y)
                        ce_loss = ce_loss - self.noise_ascent_beta * ce_noise
            loss = ce_loss
            use_distill = (
                self.distill_mode != "none"
                and self.distill_logits is not None
                and self.distill_alpha > 0
            )
            if self.is_malicious and not self.distill_apply_to_malicious:
                use_distill = False
            if use_distill:
                T = float(self.distill_temp)
                target_logits = self.distill_logits.to(out.device)
                target_probs = F.softmax(target_logits / T, dim=1)
                student_logp = F.log_softmax(out / T, dim=1)
                distill_loss = F.kl_div(
                    student_logp, target_probs, reduction="batchmean"
                ) * (T * T)
                loss = ce_loss + self.distill_alpha * distill_loss
            loss.backward()
            optimizer.step()

            if pick_best or pick_worst:
                self.local_model.eval()
                with torch.no_grad():
                    test_out = self.local_model(self.test_data.x)
                    preds = test_out.argmax(dim=1)
                    acc = accuracy_score(
                        self.test_data.y.detach().cpu(), preds.detach().cpu()
                    )
                    if pick_best and acc > best_acc:
                        best_acc = acc
                        best_sd = copy.deepcopy(self.local_model.state_dict())
                    if pick_worst and acc < worst_acc:
                        worst_acc = acc
                        worst_sd = copy.deepcopy(self.local_model.state_dict())

        if pick_best and best_sd is not None:
            self.local_model.load_state_dict(best_sd)
        elif pick_worst and worst_sd is not None:
            self.local_model.load_state_dict(worst_sd)

        if self.is_malicious:
            self._apply_malicious_attack()

        self.round_idx += 1

    def _apply_malicious_attack(self):
        if self.local_model is None:
            return
        if self.attack_mode == "none":
            return
        if self.attack_mode in {"label_flip", "backdoor_trigger"}:
            return

        if self.attack_mode == "dt_logit_scale":
            # Phase 3: R2 不敏感（argmax不变）、DT一致性敏感（softmax变）
            if self.round_idx <= getattr(C, "DT_ATTACK_WARMUP_ROUNDS", 1):
                return
            step = self.round_idx - getattr(C, "DT_ATTACK_WARMUP_ROUNDS", 1)
            scale = float(getattr(C, "DT_ATTACK_SCALE_START", 1.0)) - step * float(
                getattr(C, "DT_ATTACK_SCALE_STEP", 0.1)
            )
            scale = max(float(getattr(C, "DT_ATTACK_SCALE_END", 0.01)), float(scale))
            scale = max(1e-3, scale)

            with torch.no_grad():
                self.local_model.fc2.weight.data.mul_(scale)
                self.local_model.fc2.bias.data.mul_(scale)

            if getattr(C, "PRINT_DEBUG", False) and self.round_idx in getattr(
                C, "DEBUG_ROUNDS", set()
            ):
                print(
                    f"[Malicious {self.drone_id}] dt_logit_scale: round={self.round_idx}, scale={scale:.4f}"
                )
            return

        if self.attack_mode == "stealth_amp":
            self._apply_stealth_amp_attack()
            return

        raise ValueError(f"Unknown MAL_ATTACK_MODE={self.attack_mode}")

    def _apply_backdoor_trigger(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not getattr(C, "BACKDOOR_ATTACK_ENABLED", True):
            return x, y
        ratio = float(getattr(C, "BACKDOOR_INJECT_RATIO", 0.0))
        if ratio <= 0:
            return x, y
        if x is None or y is None or x.shape[0] != y.shape[0]:
            return x, y

        mask = torch.rand(y.shape, device=x.device) < ratio
        if not mask.any():
            return x, y

        trigger_idx = getattr(C, "BACKDOOR_TRIGGER_IDX", [0, 1, 2])
        mode = getattr(C, "BACKDOOR_TRIGGER_MODE", "add")
        scale = float(getattr(C, "BACKDOOR_TRIGGER_SCALE", 3.0))
        value = getattr(C, "BACKDOOR_TRIGGER_VALUE", None)
        clip_min = getattr(C, "BACKDOOR_CLIP_MIN", None)
        clip_max = getattr(C, "BACKDOOR_CLIP_MAX", None)
        target_label = int(getattr(C, "BACKDOOR_TARGET_LABEL", 1))

        x_bd = apply_backdoor_trigger(
            x,
            trigger_idx=trigger_idx,
            mode=mode,
            scale=scale,
            value=value,
            clip_min=clip_min,
            clip_max=clip_max,
            mask=mask,
        )
        y_bd = y.clone()
        y_bd[mask] = target_label
        return x_bd, y_bd

    def _flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_flip_ratio <= 0:
            return labels
        if self.round_idx < self.label_flip_warmup:
            return labels
        if labels.numel() == 0:
            return labels

        y_flip = labels.clone()
        mask = torch.rand(y_flip.shape, device=y_flip.device) < self.label_flip_ratio
        if not mask.any():
            return y_flip

        num_classes = int(y_flip.max().item()) + 1
        if num_classes <= 1:
            return y_flip
        if num_classes == 2:
            y_flip[mask] = 1 - y_flip[mask]
            return y_flip

        rand = torch.randint(
            0, num_classes - 1, size=(mask.sum(),), device=y_flip.device
        )
        orig = y_flip[mask]
        rand = rand + (rand >= orig).long()
        y_flip[mask] = rand
        return y_flip

    def _apply_stealth_amp_attack(self):
        if self.round_idx <= self.warmup_rounds:
            return
        if self.ref_model_for_attack is None or self.local_model is None:
            return

        step = self.round_idx - self.warmup_rounds
        amp = min(self.max_amp, step * self.amp_step)
        noise_std = self.noise_base + step * self.noise_step

        ref_sd = self.ref_model_for_attack.state_dict()
        with torch.no_grad():
            for name, p in self.local_model.named_parameters():
                if name in ref_sd and torch.is_floating_point(p):
                    ref = ref_sd[name].to(p.device).data
                    delta = p.data - ref
                    p.add_(amp * delta)
                    if noise_std > 0:
                        p.add_(torch.randn_like(p) * noise_std)

        if getattr(C, "PRINT_DEBUG", False) and self.round_idx in getattr(
            C, "DEBUG_ROUNDS", set()
        ):
            print(
                f"[Malicious {self.drone_id}] stealth_amp: round={self.round_idx}, amp={amp:.4f}, noise={noise_std:.4f}"
            )


class CentralServer:
    def __init__(
        self,
        ablation_config: str,
        r4_alpha: float,
        use_perf_penalty: bool = USE_PERF_PENALTY,
    ):
        self.global_model = StudentNet(num_output_features=2).to(device)
        self.ablation_config = ablation_config
        self.r4_alpha = float(r4_alpha)
        self.use_perf_penalty = bool(use_perf_penalty)

        self.reputation: Dict[int, float] = {}
        self.data_age: Dict[int, int] = {}
        self.low_performance_counts: Dict[int, int] = {}
        self.reputation_log: List[Dict[str, Any]] = []
        self.last_broadcast_state = None

        # Phase 2：预计算 twin reference（softmax + mask + weights）
        self.twin_probs_ref: Optional[torch.Tensor] = None  # [N,2]
        self.r4_mask: Optional[torch.Tensor] = None  # [N] bool
        self.r4_weights: Optional[torch.Tensor] = None  # [N] float
        self.r4_weights_att: Optional[torch.Tensor] = (
            None  # [N] float, 非 normal 类的权重
        )
        self.teacher_probs_ref: Optional[torch.Tensor] = None  # [N,C]
        self.teacher_preds_ref: Optional[torch.Tensor] = None  # [N]
        self.teacher_mask: Optional[torch.Tensor] = None  # [N] bool
        self.r4_mask_frac: float = float("nan")
        self.r4_mask_count: int = 0
        self.r4_mask_mean_conf: float = float("nan")
        self.r4_mask_top1_frac: float = float("nan")
        self.r4_mask_att_frac: float = float("nan")
        self.r4_mask_att_count: int = 0
        self.r4_mask_att_top1_frac: float = float("nan")
        self.last_r4_kl: Dict[int, float] = {}
        self.last_r4_kl_norm: Dict[int, float] = {}
        self.last_r4_kl_att: Dict[int, float] = {}
        self.last_r4_js: Dict[int, float] = {}
        self.last_r4_trace: Dict[int, float] = {}
        self.last_r4_gate_hit: Dict[int, bool] = {}

    def set_twin_reference(
        self,
        twin_logits_for_probs: torch.Tensor,
        twin_logits_for_mask: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        twin_logits_for_probs: 用于 KL 的孪生分布（当前 L0/L1/L2）
        twin_logits_for_mask : 仅用于定义 normal prior 区域的 mask（固定用 L0）
        """
        # clear teacher reference to force twin-based path when both are present
        self.teacher_probs_ref = None
        self.teacher_preds_ref = None
        self.teacher_mask = None
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)

            # KL 用的 twin probs（可随 L1/L2 变化）
            twin_probs = softmax(twin_logits_for_probs / temperature)

            # mask 用的 twin probs（固定用 L0）
            mask_probs = softmax(twin_logits_for_mask / temperature)

            # 先拿到置信度与预测，以便构造“类均衡”权重
            conf, pred = mask_probs.max(dim=1)
            mask_conf = torch.ones_like(conf, dtype=torch.bool)
            if C.R4_USE_ONLY_CONFIDENT:
                mask_conf &= conf >= C.R4_CONF_THRESH

            # Normal 与非 Normal 各自一套 mask/权重，后续 KL 平均
            mask_norm = mask_conf & (pred == C.NORMAL_CLASS_INDEX)
            mask_att = mask_conf & (pred != C.NORMAL_CLASS_INDEX)

            weights_norm = conf * mask_norm.float()
            weights_att = conf * mask_att.float()

            # 权重集中度（便于诊断）
            def _top1_frac(w: torch.Tensor) -> float:
                s = float(w.sum().item())
                return float(w.max().item()) / s if s > 1e-12 else float("nan")

            top1_norm = _top1_frac(weights_norm)
            top1_att = _top1_frac(weights_att)

            self.twin_probs_ref = twin_probs.detach()
            self.r4_mask = mask_norm.detach()
            self.r4_weights = weights_norm.detach()
            self.r4_weights_att = weights_att.detach()

            self.r4_mask_frac = float(mask_norm.float().mean().item())
            self.r4_mask_count = int(mask_norm.sum().item())
            self.r4_mask_mean_conf = (
                float(conf[mask_norm].mean().item())
                if mask_norm.any()
                else float("nan")
            )
            self.r4_mask_top1_frac = float(top1_norm)

            self.r4_mask_att_frac = float(mask_att.float().mean().item())
            self.r4_mask_att_count = int(mask_att.sum().item())
            self.r4_mask_att_top1_frac = float(top1_att)

    def set_teacher_reference(
        self,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_probs: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        """
        Cache teacher outputs on reference set for semantic R4:
          - teacher_probs: [N,C] softmax on reference (preferred)
          - teacher_logits: alternative input; will be softmaxed here
        Uses the same confidence gating as twin mode; optionally keep only NORMAL class
        if C.R4_SEMANTIC_ONLY_NORMAL is True.
        """
        if teacher_probs is None and teacher_logits is None:
            raise ValueError("set_teacher_reference requires teacher_probs or teacher_logits.")

        with torch.no_grad():
            # clear twin reference to avoid mixing modes
            self.twin_probs_ref = None
            self.r4_weights_att = None

            if teacher_probs is None:
                probs = F.softmax(teacher_logits / float(temperature), dim=1)
            else:
                probs = teacher_probs

            probs = probs.detach()
            conf, pred = probs.max(dim=1)
            mask = torch.ones_like(conf, dtype=torch.bool)
            if getattr(C, "R4_USE_ONLY_CONFIDENT", True):
                mask &= conf >= getattr(C, "R4_CONF_THRESH", 0.0)
            if getattr(C, "R4_SEMANTIC_ONLY_NORMAL", False):
                mask &= pred == getattr(C, "NORMAL_CLASS_INDEX", 0)

            weights = conf * mask.float()
            weight_sum = float(weights.sum().item())
            top1_frac = (
                float(weights.max().item()) / weight_sum if weight_sum > 1e-12 else float("nan")
            )

            self.teacher_probs_ref = probs
            self.teacher_preds_ref = pred.detach()
            self.teacher_mask = mask.detach()

            # reuse mask/weight fields for diagnostics + weighting
            self.r4_mask = self.teacher_mask
            self.r4_weights = weights.detach()
            self.r4_weights_att = None

            self.r4_mask_frac = float(mask.float().mean().item())
            self.r4_mask_count = int(mask.sum().item())
            self.r4_mask_mean_conf = (
                float(conf[mask].mean().item()) if mask.any() else float("nan")
            )
            self.r4_mask_top1_frac = float(top1_frac)
            self.r4_mask_att_frac = 0.0
            self.r4_mask_att_count = 0
            self.r4_mask_att_top1_frac = float("nan")

    def aggregate_models(self, local_models):
        for i, m in enumerate(local_models):
            m.drone_id = i
        # rep-aware bulyan hybrid (filter low-rep then bulyan) if requested via aggs
        agg_method = getattr(self, "agg_method_override", None)
        if agg_method == "r4_bulyan":
            # keep nodes with rep above median, then run bulyan on filtered set
            reps = torch.tensor(
                [self.reputation.get(m.drone_id, 1.0) for m in local_models]
            )
            med = reps.median()
            keep = reps >= med
            filtered = [m for m, k in zip(local_models, keep) if k]
            # if all filtered out, fall back to weighted
            if len(filtered) >= 2:
                agg_sd = weighted_average_aggregation(filtered, self.reputation)
            else:
                agg_sd = weighted_average_aggregation(local_models, self.reputation)
        else:
            agg_sd = weighted_average_aggregation(local_models, self.reputation)
        self.global_model.load_state_dict(agg_sd)
        self.last_broadcast_state = copy.deepcopy(self.global_model.state_dict())

    def compute_r4(self, local_logits: torch.Tensor, temperature: float = 1.0) -> float:
        """
        Semantic R4 (preferred): teacher vs student consistency on reference.
        Fallback: twin-based KL if teacher reference not provided.
        """
        if self.teacher_probs_ref is not None:
            return self._compute_r4_teacher(local_logits, temperature=temperature)
        if self.twin_probs_ref is not None and self.r4_weights is not None:
            return self._compute_r4_twin(local_logits, temperature=temperature)
        return 0.5

    def _compute_r4_twin(self, local_logits: torch.Tensor, temperature: float = 1.0) -> float:
        """Original twin-based KL (kept for backward compatibility)."""
        with torch.no_grad():
            self._last_r4_js_tmp = float("nan")
            self._last_r4_trace_tmp = float("nan")

            softmax = nn.Softmax(dim=1)
            student_probs = softmax(local_logits / temperature)
            twin_probs = self.twin_probs_ref

            weights_norm = self.r4_weights
            ws_norm = weights_norm.sum().item()
            if ws_norm < 1e-6:
                return 0.5

            kl = F.kl_div(
                torch.log(student_probs + 1e-8), twin_probs, reduction="none"
            ).sum(dim=1)

            kl_norm = (weights_norm * kl).sum() / weights_norm.sum()
            self._last_r4_kl_norm_tmp = float(kl_norm.item())

            kl_att_val = float("nan")
            if self.r4_weights_att is not None:
                w_att = self.r4_weights_att
                if w_att.sum().item() > 1e-6:
                    kl_att = (w_att * kl).sum() / w_att.sum()
                    kl_att_val = float(kl_att.item())
            self._last_r4_kl_att_tmp = kl_att_val

            if math.isnan(kl_att_val):
                kl_combined = kl_norm
            else:
                kl_combined = 0.5 * kl_norm + 0.5 * kl_att

            self._last_r4_kl_tmp = float(kl_combined.item())

            r4 = math.exp(-float(kl_combined.item()))
            return float(max(0.0, min(1.0, r4)))

    def _compute_r4_teacher(
        self, local_logits: torch.Tensor, temperature: float = 1.0
    ) -> float:
        """Teacher-student semantic consistency (JS + optional confusion trace)."""
        if self.teacher_probs_ref is None:
            return 0.5

        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            student_probs = softmax(local_logits / temperature)

            teacher_probs = self.teacher_probs_ref
            teacher_preds = self.teacher_preds_ref
            mask = self.teacher_mask
            weights = self.r4_weights

            if mask is not None:
                teacher_probs = teacher_probs[mask]
                teacher_preds = teacher_preds[mask]
                student_probs = student_probs[mask]
                if weights is not None:
                    weights = weights[mask]

            # JS term
            if student_probs.numel() == 0:
                js_mean = float("nan")
                r_js = 0.5
            else:
                js_vals = js_divergence(teacher_probs, student_probs)
                if weights is not None and weights.numel() == js_vals.numel():
                    ws = float(weights.sum().item())
                    if ws > 1e-12:
                        js_mean = float((weights * js_vals).sum().item() / ws)
                    else:
                        js_mean = float(js_vals.mean().item())
                else:
                    js_mean = float(js_vals.mean().item())
                alpha = float(getattr(C, "R4_JS_ALPHA", 6.0))
                r_js = math.exp(-alpha * js_mean)

            # Confusion-trace term (targets label-flip)
            mix = float(getattr(C, "R4_CONFUSION_MIX", 0.0))
            trace_val = float("nan")
            r_conf = None
            if mix > 0 and student_probs.numel() > 0:
                student_preds = student_probs.argmax(dim=1)
                trace_val = confusion_trace(
                    student_preds, teacher_preds, num_classes=student_probs.shape[1]
                )
                if not math.isnan(trace_val):
                    beta_conf = float(getattr(C, "R4_CONFUSION_BETA", 8.0))
                    r_conf = math.exp(-beta_conf * (1.0 - trace_val))

            if r_conf is not None:
                mix = max(0.0, min(1.0, mix))
                r4 = (1.0 - mix) * r_js + mix * r_conf
            else:
                r4 = r_js

            # diagnostics
            self._last_r4_js_tmp = float(js_mean)
            self._last_r4_trace_tmp = float(trace_val)
            self._last_r4_kl_tmp = float(js_mean)
            self._last_r4_kl_norm_tmp = float(js_mean)
            self._last_r4_kl_att_tmp = float("nan")

            return float(max(0.0, min(1.0, r4)))

    def compute_reputation(
        self,
        drone_id: int,
        performance: float,
        data_age: int,
        local_logits: torch.Tensor,
        current_round: int,
    ):
        """
        Final (recommended) reputation:
        - R2 = sigmoid(performance)             (if enabled)
        - R3 = exp(-0.8 * data_age)             (if enabled)
        - R4 = teacher-student semantic score (JS/confusion) if teacher_ref is set,
              else exp(-KL(twin || student))

        Rep is computed in *log-linear* space to avoid "difference compression":
        rep = exp( z ), where
        z = beta2*(R2-0.5) + beta3*(R3-0.5) + beta4*(R4-0.5)

        Special case:
        - If active_r == {"R4"}: use stronger mapping:
                rep = exp( beta*(R4-0.5) )
            where beta = max(self.r4_alpha, C.R4_ONLY_BETA)  (ensures >0)
        Mixed amplification:
        - If R4 in active set and not R4-only, add extra strength:
                beta4_eff = C.BETA_R4 + C.MIX_R4_BETA
            (set MIX_R4_BETA=0 to disable)
        """
        active_r = {
            name.strip() for name in self.ablation_config.split(",") if name.strip()
        }

        # ----- component scores -----
        R2 = sigmoid(performance) if "R2" in active_r else 0.5
        R3 = exponential_decay(data_age) if "R3" in active_r else 0.5
        R4 = self.compute_r4(local_logits) if "R4" in active_r else 0.5
        self.last_r4_kl[drone_id] = float(
            getattr(self, "_last_r4_kl_tmp", float("nan"))
        )
        self.last_r4_kl_norm[drone_id] = float(
            getattr(self, "_last_r4_kl_norm_tmp", float("nan"))
        )
        self.last_r4_kl_att[drone_id] = float(
            getattr(self, "_last_r4_kl_att_tmp", float("nan"))
        )
        self.last_r4_js[drone_id] = float(
            getattr(self, "_last_r4_js_tmp", float("nan"))
        )
        self.last_r4_trace[drone_id] = float(
            getattr(self, "_last_r4_trace_tmp", float("nan"))
        )

        # R4 基于 twin 的准入门控
        r4_shrink = 1.0
        beta4_eff = 0.0
        tau = float(getattr(C, "R4_GATE_TAU", 0.50))
        if "R4" in active_r:
            eps = float(getattr(C, "R4_GATE_EPS", 1e-8))
            soft = float(getattr(C, "R4_GATE_SOFT", 0.0))  # 0=硬，>0=软缩放
            if float(R4) < tau:
                if soft <= 0.0:
                    self.last_r4_gate_hit[drone_id] = True
                    return float(eps), (
                        float(R2),
                        float(R3),
                        float(R4),
                    )  # 硬拒绝：直接给极小信誉，保持返回格式一致
                shrink = max(0.0, float(R4) / max(tau, 1e-12))
                r4_shrink = max(eps, shrink)
        self.last_r4_gate_hit[drone_id] = r4_shrink < 0.999

        # ----- compute rep in log space -----
        z = 0.0
        z_r2 = 0.0
        z_r3 = 0.0
        z_r4 = 0.0

        # R4-only: strong mapping (what your minitest shows as effective)
        if active_r == {"R4"}:
            beta = float(max(self.r4_alpha, getattr(C, "R4_ONLY_BETA", 10.0)))
            z_r4 = beta * (float(R4) - 0.5)
            z = z_r4

        else:
            # R2/R3 terms (can be weakened by lowering BETA_R2/BETA_R3)
            if "R2" in active_r:
                beta2 = float(getattr(C, "BETA_R2", 0.5))
                z_r2 = beta2 * (float(R2) - 0.5)
                z += z_r2
            if "R3" in active_r:
                beta3 = float(getattr(C, "BETA_R3", 0.0))
                z_r3 = beta3 * (float(R3) - 0.5)
                z += z_r3

            # R4 term (base + optional mixed amplification)
            if "R4" in active_r:
                beta4 = float(getattr(C, "BETA_R4", 1.0))
                mix_beta = float(getattr(C, "MIX_R4_BETA", 0.0))
                beta4_eff = beta4 + mix_beta  # set MIX_R4_BETA=0 to disable
                z_r4 = beta4_eff * (float(R4) - 0.5)
                z += z_r4

        # clamp for numeric stability
        z_clamp = float(getattr(C, "REP_Z_CLAMP", 20.0))
        z = max(-z_clamp, min(z_clamp, z))
        rep = math.exp(z) * r4_shrink

        # ----- optional performance penalty (keep your original option) -----
        penalized = False
        if self.use_perf_penalty and performance < PERF_THRESH:
            self.low_performance_counts[drone_id] = (
                self.low_performance_counts.get(drone_id, 0) + 1
            )
            rep *= PERF_PENALTY_FACTOR
            penalized = True
        else:
            self.low_performance_counts[drone_id] = 0

        # ----- log (keep consistent keys for analysis) -----
        entry = {
            "Round": current_round,
            "Drone_ID": drone_id,
            "Perf": float(performance),
            "R2": float(R2),
            "R3": float(R3),
            "R4": float(R4),
            "Rep_Z": float(z),
            "Final_Rep": float(rep),
            "Penalized": bool(penalized),
            # 额外诊断字段
            "z_R2": float(z_r2),
            "z_R3": float(z_r3),
            "z_R4": float(z_r4),
            "r4_shrink": float(r4_shrink),
            "beta4_eff": float(beta4_eff),
            "tau_gate": float(tau),
            "R4_KL": float(self.last_r4_kl.get(drone_id, float("nan"))),
            "R4_JS": float(self.last_r4_js.get(drone_id, float("nan"))),
            "R4_Trace": float(self.last_r4_trace.get(drone_id, float("nan"))),
        }
        self.reputation_log.append(entry)

        return float(rep), (float(R2), float(R3), float(R4))
