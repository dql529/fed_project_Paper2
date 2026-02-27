"""
dt_r4/runtime.py

运行时环境：可复现性设置 + device。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 强可复现（注意：可能会降低性能，且某些算子在 GPU 上可能报错）
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
