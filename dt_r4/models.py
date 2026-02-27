"""
dt_r4/models.py

模型定义：StudentNet / TeacherNet
"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F

from .config import STUDENT_FEATURE_IDX


class StudentNet(nn.Module):
    def __init__(
        self, num_output_features: int = 2, feature_idx: Optional[List[int]] = None
    ):
        super().__init__()
        self.feature_idx = feature_idx or STUDENT_FEATURE_IDX
        in_dim = len(self.feature_idx)
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, num_output_features)

    def forward(self, x):
        x = x[:, self.feature_idx]
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TeacherNet(nn.Module):
    def __init__(self, num_output_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
