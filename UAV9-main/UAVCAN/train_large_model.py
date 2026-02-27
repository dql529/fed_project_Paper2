import os
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# =========================
# Config
# =========================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Needed for deterministic CUDA matmul with CuBLAS when torch.use_deterministic_algorithms(True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent

TEACHER_CSV = SCRIPT_DIR / "split_data" / "type10_teacher.csv"
GLOBAL_TEST_CSV = SCRIPT_DIR / "split_data" / "type10_global_test.csv"

OUTPUT_CKPT = SCRIPT_DIR / "large_model.pt"
OUTPUT_XGB = SCRIPT_DIR / "large_model_xg.pt"
OUTPUT_BEST_META = SCRIPT_DIR / "large_model_best.json"

INPUT_DIM = 9  # type10 features
NUM_CLASSES = 2

LR = 1e-3  # 比你之前的 0.05 稳定得多
EPOCHS = 40  # 对 ~75k 样本足够
BATCH_SIZE = 2048  # teacher 可以用大 batch



# XGBoost (SOTA baseline for teacher)
RUN_XGBOOST = True
XGB_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LR = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8
XGB_EARLY_STOPPING = 20
XGB_TREE_METHOD = "hist"

# =========================
# Reproducibility
# =========================
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _xgb_fit(model, x_tr, y_tr, x_va, y_va):
    fit_kwargs = {"eval_set": [(x_va, y_va)], "verbose": False}
    if XGB_EARLY_STOPPING > 0:
        try:
            sig = inspect.signature(model.fit)
        except (TypeError, ValueError):
            sig = None
        if sig is not None and "early_stopping_rounds" in sig.parameters:
            fit_kwargs["early_stopping_rounds"] = XGB_EARLY_STOPPING
        else:
            try:
                model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING)
            except Exception:
                pass
    model.fit(x_tr, y_tr, **fit_kwargs)

set_seed(SEED)


# =========================
# Model Definition
# =========================
class TeacherNet(nn.Module):
    def __init__(self, input_dim=9, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# =========================
# Load Data
# =========================
print("Loading teacher training data...")


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]

    # Coerce all columns to numeric; drop rows missing the target; fill feature gaps with 0.
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df[feature_cols] = df[feature_cols].fillna(0)
    df[target_col] = df[target_col].astype(int)
    return df


df_train = load_dataset(TEACHER_CSV)
df_test = load_dataset(GLOBAL_TEST_CSV)

X_train = torch.tensor(df_train.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(df_train.iloc[:, -1].values, dtype=torch.long)

X_test = torch.tensor(df_test.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(df_test.iloc[:, -1].values, dtype=torch.long)

X_train_np = df_train.iloc[:, :-1].values.astype(np.float32)
y_train_np = df_train.iloc[:, -1].values.astype(np.int64)
X_test_np = df_test.iloc[:, :-1].values.astype(np.float32)
y_test_np = df_test.iloc[:, -1].values.astype(np.int64)

print(f"Teacher train samples: {len(X_train)}")
print(f"Global test samples : {len(X_test)}")

# =========================
# DataLoader
# =========================
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)

# =========================
# Train Teacher
# =========================
model = TeacherNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
best_state = None

print("\n=== Training Digital Twin (Large Model) ===")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # ===== Evaluation on global test =====
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test.to(DEVICE))
        preds = logits_test.argmax(dim=1)
        acc = accuracy_score(y_test.numpy(), preds.cpu().numpy())

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"TrainLoss={avg_loss:.4f} | "
        f"GlobalTestAcc={acc:.4f}"
    )

    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())

# =========================
# Train XGBoost + Save Best
# =========================
best_mlp_acc = best_acc
xgb_acc = None

if RUN_XGBOOST:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
    xgb_model = XGBClassifier(
        n_estimators=XGB_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=XGB_TREE_METHOD,
        n_jobs=-1,
        random_state=SEED,
    )
    _xgb_fit(xgb_model, X_train_np, y_train_np, X_test_np, y_test_np)
    xgb_preds = xgb_model.predict(X_test_np)
    xgb_acc = accuracy_score(y_test_np, xgb_preds)
    xgb_model.save_model(OUTPUT_XGB)

if best_state is None:
    raise RuntimeError("No best MLP state captured.")

torch.save(best_state, OUTPUT_CKPT)

best_type = "mlp"
best_acc = best_mlp_acc
best_path = OUTPUT_CKPT
if xgb_acc is not None and xgb_acc > best_mlp_acc:
    best_type = "xgboost"
    best_acc = xgb_acc
    best_path = OUTPUT_XGB

best_info = {
    "best_type": best_type,
    "best_acc": float(best_acc),
    "mlp_acc": float(best_mlp_acc),
    "xgb_acc": None if xgb_acc is None else float(xgb_acc),
    "mlp_path": str(OUTPUT_CKPT),
    "xgb_path": None if xgb_acc is None else str(OUTPUT_XGB),
    "best_path": str(best_path),
}
OUTPUT_BEST_META.write_text(json.dumps(best_info, indent=2), encoding="utf-8")

print("\n====================================")
print(f"Best MLP Global Test Accuracy: {best_mlp_acc:.4f}")
if xgb_acc is not None:
    print(f"XGBoost Global Test Accuracy: {xgb_acc:.4f}")
print(f"Best model: {best_type} (acc={best_acc:.4f})")
print(f"Saved MLP to: {OUTPUT_CKPT}")
if xgb_acc is not None:
    print(f"Saved XGBoost to: {OUTPUT_XGB}")
print(f"Saved best meta to: {OUTPUT_BEST_META}")
print("====================================")
