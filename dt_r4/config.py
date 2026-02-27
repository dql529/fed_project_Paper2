"""
dt_r4/config.py

集中管理配置：节点/路径/孪生/R4/攻击/调试等。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 基本与场景
# ---------------------------------------------------------------------------
NUM_NODES = 10
MALICIOUS_NODES = 3
SEED_LIST = list(range(2))
NUM_ROUNDS = 15

# 评估场景：A=train clean / eval clean；B=train clean / eval noise
EVAL_SCENARIO = "A"  # "A" 或 "B"
EVAL_DEPLOY_VARIANT = (
    1  # 仅 B 用，对 NOISE_VARIANTS 索引（0=clean，1=feat_noise_20,...）
)

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
TEACHER_CKPT = "UAV9-main/UAVCAN/large_model.pt"
CSV_PATH = "UAV9-main/UAVCAN/split_data/type10_student_pool.csv"
GLOBAL_REF_CSV = "UAV9-main/UAVCAN/split_data/type10_global_test.csv"
NOISE_DATA_DIR = "UAV9-main/UAVCAN/split_data/noise_variants"

# ---------------------------------------------------------------------------
# 数据 / 噪声变体
# ---------------------------------------------------------------------------
TARGET_NOISESET_NAMES = [
    "clean",
    "feat_noise_20",
    "feat_noise_25_dropout_20",
    "feat_noise_35_dropout_30",
    "feat_noise_50_dropout_50",
]
NOISE_VARIANTS = [
    {
        "name": "clean",
        "feature_noise_frac": 0.0,
        "label_flip_ratio": 0.0,
        "feature_dropout_ratio": 0.0,
    },
    {
        "name": "feat_noise_20",
        "feature_noise_frac": 0.20,
        "label_flip_ratio": 0.0,
        "feature_dropout_ratio": 0.0,
    },
    {
        "name": "feat_noise_25_dropout_20",
        "feature_noise_frac": 0.25,
        "label_flip_ratio": 0.0,
        "feature_dropout_ratio": 0.20,
    },
    {
        "name": "feat_noise_35_dropout_30",
        "feature_noise_frac": 0.35,
        "label_flip_ratio": 0.0,
        "feature_dropout_ratio": 0.30,
    },
    {
        "name": "feat_noise_50_dropout_50",
        "feature_noise_frac": 0.50,
        "label_flip_ratio": 0.0,
        "feature_dropout_ratio": 0.50,
    },
]

# ---------------------------------------------------------------------------
# 孪生 / 参考
# ---------------------------------------------------------------------------
TWIN_MISMATCH_SPECS = [
    {
        "name": "twin_L0_match",
        "desc": "无漂移/无丢维/无logit偏置",
        "drift": 0.0,
        "drop_dims": [],
        "logit_bias": 0.0,
    },
    {
        "name": "twin_L1_mid",
        "desc": "中等失配：漂移+丢维+logit偏置",
        "drift": 0.10,
        "drop_dims": [7],
        "logit_bias": 0.25,
    },
    {
        "name": "twin_L2_high",
        "desc": "强失配：更大漂移/丢维/偏置",
        "drift": 0.20,
        "drop_dims": [7, 8],
        "logit_bias": 0.50,
    },
]
ENFORCE_TWIN_MONOTONIC = True
TWIN_MIN_ACC_DROP = 0.005
TWIN_CALIB_SCALE = 1.25
TWIN_CALIB_MAX_ITERS = 20
TWIN_DRIFT_MAX = 3.0
TWIN_LOGIT_BIAS_MAX = 5.0
STUDENT_FEATURE_IDX = list(range(9))

# ---------------------------------------------------------------------------
# R4 / 信誉
# ---------------------------------------------------------------------------
R4_ALPHA_LIST = [0, 4]
ABLATION_CONFIGS = ["R2,R3", "R2,R3,R4", "R4"]

R4_ONLY_BETA = 10.0  # R4-only 映射
MIX_R4_BETA = 12.0  # 混合时额外放大
BETA_R2 = 0.1  # 保留性能分量，用于对比 R2 高 / R4 低
BETA_R3 = 0.0
BETA_R4 = 8.0
REP_Z_CLAMP = 20.0

# mask / gate
NORMAL_CLASS_INDEX = 0
R4_USE_ONLY_CONFIDENT = True
R4_ONLY_NORMAL = True
R4_CONF_THRESH = 0.90
R4_GATE_TAU = 0.70  # 提高门槛，让恶意节点显著降权
R4_GATE_SOFT = 1.0  # 软缩放，不直接清零
R4_GATE_EPS = 1e-6

# 语义一致性（teacher vs student）
R4_JS_ALPHA = 15.0
R4_CONFUSION_BETA = 16.0
R4_CONFUSION_MIX = 1.0
R4_SEMANTIC_ONLY_NORMAL = False

# alpha warmup / ramp
R4_WARMUP_ROUNDS = 5
R4_RAMP_ROUNDS = 5
R4_WARMUP_ALPHA = 1.0

# ---------------------------------------------------------------------------
# 攻击
# ---------------------------------------------------------------------------
MAL_ATTACK_MODE = (
    "label_flip"  # "none" | "stealth_amp" | "dt_logit_scale" | "label_flip"
)

# label flip
LABEL_FLIP_RATIO = 0.5
LABEL_FLIP_WARMUP_ROUNDS = 0
LABEL_FLIP_EPOCHS = 8  # 提升恶意本地拟合强度，使其更好地学到翻转标签
LABEL_FLIP_LR = 0.06  # 保持 label flip 隐蔽配置
LABEL_FLIP_PICK_STRATEGY = "worst"
PRE_SPLIT_POISON = True  # 恶意节点在 train/test 划分前先整体做 label flip
POISON_LOCAL_TEST = False  # 若未启用 pre-split，则可单独翻转本地 test
# 可选：让 R2 对恶意节点“盲目”高，便于只靠 R4 识别
R2_IGNORE_MALICIOUS = False
# 额外放大/反向恶意更新；<1 让攻击更隐蔽、接近正常更新
MAL_GRAD_SCALE = 0.5  # label_flip 默认缩放
# 按攻击类型的缩放映射（可覆盖 MAL_GRAD_SCALE），方便只强化特定攻击
MAL_GRAD_SCALE_MAP = {
    "label_flip": 0.5,
    "stealth_amp": -3.0,  # 更强反向放大
    "dt_logit_scale": -3.0,  # 更强反向放大
}
# DT/Teacher 先验质量（仅作用于 weighted 的 R4）
DT_MISMATCH_LEVELS = {
    "D0": {"noise_std": 0.0, "keep_ratio": 1.0},
    "D1": {"noise_std": 0.5, "keep_ratio": 0.5},
    "D2": {"noise_std": 1.0, "keep_ratio": 0.1},
}

# dt_logit_scale
DT_ATTACK_WARMUP_ROUNDS = 0
DT_ATTACK_SCALE_START = 3.0
DT_ATTACK_SCALE_END = 0.3
DT_ATTACK_SCALE_STEP = 0.30

# stealth_amp
STEALTH_WARMUP_ROUNDS = 0
STEALTH_MAX_AMP = 1.0
STEALTH_AMP_STEP = 0.20
STEALTH_NOISE_BASE = 0.07
STEALTH_NOISE_STEP = 0.07

# backdoor trigger
BACKDOOR_ATTACK_ENABLED = True
BACKDOOR_TRIGGER_IDX = [0, 1, 2]
BACKDOOR_TRIGGER_MODE = "add"
BACKDOOR_TRIGGER_SCALE = 3.0
BACKDOOR_TRIGGER_VALUE = None
BACKDOOR_INJECT_RATIO = 0.025
BACKDOOR_TARGET_LABEL = 1
BACKDOOR_CLIP_MIN = None
BACKDOOR_CLIP_MAX = None

# noise-ascent (可选)
NOISE_ATTACK_PATH = None
NOISE_ASCENT_BETA = 6.0
NOISE_ASCENT_STEPS = 2
NOISE_STD_SCALE = 1.2
NOISE_DROPOUT_P = 0.8
NOISE_FGSM_EPS = 0.0

# ---------------------------------------------------------------------------
# 性能门控 / 调试 / 输出
# ---------------------------------------------------------------------------
USE_PERF_PENALTY = False
PERF_THRESH = 0.65
PERF_PENALTY_FACTOR = 0.1
UPDATE_NORM_GAMMA = 8.0
REP_EMA = 0.6

PRINT_DEBUG = False
DEBUG_ROUNDS = {1, 2, 3, 5, 10, 20}
DIAG_Q = 0.2

OUT_PER_ROUND = "results_per_round.csv"
OUT_PER_NODE = "results_per_node.csv"
OUT_SUMMARY = "summary_final.csv"
OUT_DIAG = "diagnostics_final.csv"

RUN_TWIN_SANITY_ONLY = False
