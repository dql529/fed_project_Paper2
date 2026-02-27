import os
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Config
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = SCRIPT_DIR / "type10_scaled.csv"
OUT_DIR = SCRIPT_DIR / "split_data"

TEACHER_RATIO = 0.7
STUDENT_RATIO = 0.2
TEST_RATIO = 0.1

SEED = 42

assert abs(TEACHER_RATIO + STUDENT_RATIO + TEST_RATIO - 1.0) < 1e-6

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
df = pd.read_csv(INPUT_CSV, low_memory=False)

print(f"Loaded {len(df)} samples from {INPUT_CSV}")

# =========================
# Shuffle
# =========================
np.random.seed(SEED)
indices = np.random.permutation(len(df))

n_teacher = int(len(df) * TEACHER_RATIO)
n_student = int(len(df) * STUDENT_RATIO)

teacher_idx = indices[:n_teacher]
student_idx = indices[n_teacher : n_teacher + n_student]
test_idx = indices[n_teacher + n_student :]

df_teacher = df.iloc[teacher_idx]
df_student = df.iloc[student_idx]
df_test = df.iloc[test_idx]

# =========================
# Save
# =========================
teacher_path = OUT_DIR / "type10_teacher.csv"
student_path = OUT_DIR / "type10_student_pool.csv"
test_path = OUT_DIR / "type10_global_test.csv"

df_teacher.to_csv(teacher_path, index=False)
df_student.to_csv(student_path, index=False)
df_test.to_csv(test_path, index=False)

print("Split finished:")
print(f" Teacher : {len(df_teacher)} -> {teacher_path}")
print(f" Student : {len(df_student)} -> {student_path}")
print(f" Test    : {len(df_test)} -> {test_path}")
