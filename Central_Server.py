import psutil, os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ 完全屏蔽GPU
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # 正确的环境变量名
import warnings

warnings.filterwarnings(
    "ignore", message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
)
import numpy as np
from openpyxl import load_workbook
from flask import Flask, request, jsonify
from Model import Net54, Net18, GAT18, GAT54
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import io
import copy

from queue import Queue
from aggregation_solution import (
    weighted_average_aggregation,
    average_aggregation,
    fed_mae_aggregation,
    median_aggregation,
    trimmed_mean_aggregation,
    krum_aggregation,
    multi_krum_aggregation,
    bulyan_aggregation,
)
import threading
import json
from tools import plot_accuracy_vs_epoch, sigmoid, exponential_decay
from matplotlib import pyplot as plt
import logging
import sys
import random
from wifi_traffic_dataset.compare import compare_model_weights
import pandas as pd

random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
learning_rate = 0.01
num_output_features = 2
criterion = nn.CrossEntropyLoss()
num_nodes = 5
# 读取数据
server_train = torch.load(
    "./wifi_traffic_dataset/Mdata_object/server_train.pt", map_location="cpu"
)
server_test = torch.load(
    "./wifi_traffic_dataset/data_object/server_test.pt", map_location="cpu"
)
data_device = server_train.to(device)
data_test_device = server_test.to(device)

scaler = StandardScaler()
R1_list = []


class CentralServer:
    def __init__(self, use_reputation="C"):
        self.use_reputation = use_reputation
        self.global_model = Net18(num_output_features).to(device)
        self.aggregated_global_model = None
        self.reputation = {}
        self.local_models = Queue()
        self.lock = threading.Lock()
        self.new_model_event = threading.Event()
        self.drone_nodes = {}
        self.aggregation_accuracies = [0.5]
        self.num_aggregations = 1
        self.data_age = {}
        self.low_performance_counts = {}
        self.aggregation_times = []  # 用于记录每轮聚合的时间
        self.agg_cost_log = []
        self.train_comm_log = []  # ✅ 用于保存每轮 Drone 的训练时间与通信量

        # **NEW: 历史/方向所需的缓存**
        self.last_broadcast_state = None  # 最近一次广播出去的模型参数
        self.perf_history = []  # [(drone_id, performance), ...]
        self.reputation_ema = {}  # EWMA 平滑后的声誉

        # # 加载并评估初始化模型
        # self.load_initial_model_and_evaluate()
        threading.Thread(target=self.check_and_aggregate_models).start()

    # **NEW: state_dict -> 1D 向量（仅取浮点参数）**
    def _state_to_vec(self, state_dict):
        parts = []
        for v in state_dict.values():
            if torch.is_floating_point(v):
                parts.append(v.detach().float().view(-1).cpu())
        return torch.cat(parts) if len(parts) else torch.tensor([], dtype=torch.float32)

    # **NEW: 余弦相似度到 [0,1]**
    def _cos01(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0 or b.numel() == 0:
            return 0.5
        denom = (a.norm() * b.norm()).item()
        if denom == 0.0:
            return 0.5
        cos = float((a @ b) / denom)
        return max(0.0, min(1.0, 0.5 * (cos + 1.0)))

    def load_initial_model_and_evaluate(self):
        # 加载已保存的 global_model.pt
        if os.path.exists("global_model.pt"):
            self.global_model.load_state_dict(
                torch.load("global_model.pt", map_location="cpu")
            )
            print("Loaded initial global model from global_model.pt")

            # 评估初始模型的准确率
            initial_accuracy = self.fed_evaluate(self.global_model, data_test_device)[0]
            self.aggregation_accuracies.append(initial_accuracy)
            print(f"Initial model accuracy: {initial_accuracy:.4f}")
        else:
            print(
                "No existing global_model.pt found, skipping initial accuracy evaluation."
            )

    def fed_evaluate(self, model, data_test_device):
        model.eval()
        with torch.no_grad():
            outputs_test = model(data_test_device)
            predictions_test = self.to_predictions(outputs_test)

            accuracy = round(
                accuracy_score(data_test_device.y.cpu(), predictions_test.cpu()), 4
            )
            precision = round(
                precision_score(data_test_device.y.cpu(), predictions_test.cpu()), 4
            )
            recall = round(
                recall_score(data_test_device.y.cpu(), predictions_test.cpu()), 4
            )
            f1 = round(f1_score(data_test_device.y.cpu(), predictions_test.cpu()), 4)

            try:
                probs = torch.softmax(outputs_test, dim=1)[:, 1].cpu().numpy()
                y_true = data_test_device.y.cpu().numpy()
                auc = round(roc_auc_score(y_true, probs), 4)

                # 保存 ROC 点（每次覆盖）
                fpr, tpr, _ = roc_curve(y_true, probs)
                pd.DataFrame({"FPR": fpr, "TPR": tpr}).to_csv(
                    "roc_points.csv", index=False
                )
            except Exception as e:
                print(f"[WARN] ROC-AUC skipped: {e}")
                auc = None

            return accuracy, precision, recall, f1, auc

    def compute_loss(self, outputs, labels):
        return criterion(outputs, labels.squeeze().long())

    def to_predictions(self, outputs):
        return outputs.argmax(dim=1)

    # def train_svm(self, gnn_output, labels):
    #     svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    #     svm_model.fit(gnn_output, labels)
    #     return svm_model

    def check_and_aggregate_models(self):
        use_reputation = self.use_reputation
        # 根据实际的节点数量调整

        # ✅ 在此处定义当前实验的基础信息
        algorithm_map = {
            "A": "Reputation-Based Aggregation",
            "B": "Simple Average Aggregation",
            "C": "Fed-MAE Aggregation",
            "D": "median_aggregation",
            "E": "trimmed_mean_aggregation",
            "F": "krum_aggregation",
            "G": "multi_krum_aggregation",
            "H": "bulyan_aggregation",
        }

        self.current_config = {
            "Aggregation Algorithm": algorithm_map.get(use_reputation, "Unknown"),
            "Total Nodes": num_nodes,  # 如果你动态设定了 num_nodes，请用变量
            "Malicious Nodes": 0,  # ⚠️ 需要你手动或程序中指定当前的恶意节点数
        }

        torch.manual_seed(0)
        np.random.seed(0)
        print("LOGGER-INFO: check_and_aggregate_models() is called")
        aggregation_start_time = time.time()  # 记录聚合开始时间

        # 假设有 3 个节点，为每个节点的准确率添加初始值 0.6

        all_individual_accuracies = [[0.5] for _ in range(num_nodes)]

        while True:
            self.new_model_event.wait()
            if self.local_models.qsize() >= num_nodes:
                round_start_time = time.time()
                models_to_aggregate = []

                self.lock.acquire()
                try:
                    if use_reputation == "A":  # Weighted Average Aggregation
                        all_models = [self.local_models.get() for _ in range(num_nodes)]
                        all_reputations = [
                            self.reputation[drone_id]
                            for model_dict in all_models
                            for drone_id in model_dict
                        ]
                        sorted_indices = sorted(
                            range(len(all_reputations)),
                            key=lambda i: all_reputations[i],
                            reverse=True,
                        )
                        models_to_aggregate = [
                            all_models[i] for i in sorted_indices[:num_nodes]
                        ]
                        aggregated_node_ids = [
                            list(model_dict.keys())[0]
                            for model_dict in models_to_aggregate
                        ]
                        print(
                            f"Nodes participating in weighted average aggregation: {aggregated_node_ids}"
                        )
                    else:
                        # ✅ B/C/D/E/F/G/H 统 一 逻 辑 ：直接取 num_nodes 个
                        models_to_aggregate = [
                            self.local_models.get() for _ in range(num_nodes)
                        ]
                        aggregated_node_ids = [
                            list(m.keys())[0] for m in models_to_aggregate
                        ]
                        print(
                            f"Nodes participating in {use_reputation} aggregation: {aggregated_node_ids}"
                        )

                finally:
                    self.lock.release()

                individual_accuracies = []
                for model_dict in models_to_aggregate:
                    for drone_id, model in model_dict.items():
                        accuracy, precision, recall, f1, auc = self.fed_evaluate(
                            model, data_test_device
                        )
                        individual_accuracies.append(accuracy)

                # 添加本次聚合的节点准确率
                for i in range(len(individual_accuracies)):
                    all_individual_accuracies[i].append(individual_accuracies[i])

                proc = psutil.Process(os.getpid())
                rss_before = proc.memory_info().rss / 1024  # MB
                t0 = time.perf_counter()
                self.lock.acquire()
                try:
                    if use_reputation == "A":
                        self.aggregate_models(models_to_aggregate, use_reputation="A")
                    elif use_reputation == "B":
                        self.aggregate_models(models_to_aggregate, use_reputation="B")
                    elif use_reputation == "C":
                        self.aggregate_models(models_to_aggregate, use_reputation="C")

                    elif use_reputation == "D":
                        self.aggregate_models(models_to_aggregate, use_reputation="D")

                    elif use_reputation == "E":
                        self.aggregate_models(models_to_aggregate, use_reputation="E")

                    elif use_reputation == "F":
                        self.aggregate_models(models_to_aggregate, use_reputation="F")

                    elif use_reputation == "G":
                        self.aggregate_models(models_to_aggregate, use_reputation="G")

                    elif use_reputation == "H":
                        self.aggregate_models(models_to_aggregate, use_reputation="H")

                finally:
                    self.lock.release()

                t1 = time.perf_counter()
                rss_after = proc.memory_info().rss / 1024  # MB

                self.agg_cost_log.append(
                    {
                        "Round": self.num_aggregations,
                        "Aggregation": use_reputation,
                        "Num_Nodes": len(models_to_aggregate),
                        "t_aggregate_s": round(t1 - t0, 4),
                        "RSS_Before_KB": round(rss_before, 2),
                        "RSS_After_KB": round(rss_after, 2),
                    }
                )
                print(
                    f"💡 Aggregation {use_reputation} took {t1 - t0:.4f}s, mem {rss_after - rss_before:+.2f}MB"
                )

                # Evaluate the aggregated global model
                accuracy, precision, recall, f1, auc = self.fed_evaluate(
                    self.aggregated_global_model, data_test_device
                )

                print(f"Aggregated model accuracy after aggregation: {accuracy:.4f}")
                self.aggregation_accuracies.append(accuracy)
                self.num_aggregations += 1

                print("Aggregation accuracies so far: ", self.aggregation_accuracies)

                round_end_time = time.time()  # 记录每轮聚合的结束时间
                round_duration = round_end_time - round_start_time
                self.aggregation_times.append(round_duration)
                # print(f"Time for this aggregation round: {round_duration:.4f} seconds")

                for drone_id, ip in self.drone_nodes.items():
                    self.send_model_thread(ip, "aggregated_global_model")

            if self.num_aggregations == 10:

                aggregation_end_time = time.time()  # 记录聚合结束时间
                total_aggregation_time = aggregation_end_time - aggregation_start_time
                # print(f"Aggregation times for each round: {self.aggregation_times}")
                print(
                    f"Total time for aggregation: {total_aggregation_time:.4f} seconds"
                )

                plot_accuracy_vs_epoch(
                    self.aggregation_accuracies,
                    all_individual_accuracies,
                    self.num_aggregations,
                    learning_rate=0.02,
                )
                final_accuracy, final_precision, final_recall, final_f1, final_auc = (
                    self.fed_evaluate(self.aggregated_global_model, data_test_device)
                )

                try:
                    import matplotlib

                    matplotlib.use("Agg")  # ✅ 不依赖图形界面，后台渲染
                    import matplotlib.pyplot as plt
                    import pandas as pd

                    roc_df = pd.read_csv("roc_points.csv")
                    plt.figure(figsize=(6, 5))
                    plt.plot(
                        roc_df["FPR"],
                        roc_df["TPR"],
                        label=f"AUC={final_auc:.4f}",
                        color="blue",
                        linewidth=2,
                    )
                    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve (Global Model)")
                    plt.legend(loc="lower right")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    plt.savefig("roc_curve.png", dpi=300)
                    plt.close()
                    print("✅ ROC curve saved as roc_curve.png")
                except Exception as e:
                    print(f"[WARN] ROC plotting failed: {e}")
                # === 构造性能表（单行） ===
                df_perf = pd.DataFrame(
                    [
                        {
                            **self.current_config,
                            "Acc": round(final_accuracy, 4),
                            "F1": round(final_f1, 4),
                            "AUC": (
                                round(final_auc, 4) if final_auc is not None else "N/A"
                            ),
                            "Total_Time_s": round(total_aggregation_time, 4),
                        }
                    ]
                )

                # === 训练/通信日志：统一到 KB，再按 Round 求平均 ===
                if hasattr(self, "train_comm_log") and len(self.train_comm_log) > 0:
                    df_train = pd.DataFrame(self.train_comm_log)
                    if (
                        "Comm_Size_KB" not in df_train.columns
                        and "Comm_Size_MB" in df_train.columns
                    ):
                        df_train["Comm_Size_KB"] = df_train["Comm_Size_MB"] * 1024
                    df_train_mean = df_train.groupby("Round", as_index=False).agg(
                        Avg_Train_Time_s=("Train_Time_s", "mean"),
                        Avg_Comm_KB=("Comm_Size_KB", "mean"),
                    )
                    avg_train_time = df_train_mean["Avg_Train_Time_s"].mean()
                    avg_comm_kb = df_train_mean["Avg_Comm_KB"].mean()
                else:
                    avg_train_time = np.nan
                    avg_comm_kb = np.nan

                    # === 聚合开销：按算法代号分组（A/B/...）求平均，再把代号->名称 ===
                if hasattr(self, "agg_cost_log") and len(self.agg_cost_log) > 0:
                    df_agg = pd.DataFrame(self.agg_cost_log)
                    df_agg_mean = df_agg.groupby("Aggregation", as_index=False).agg(
                        Avg_Agg_Time_s=("t_aggregate_s", "mean"),
                        RSS_KB=("RSS_After_KB", "max"),
                    )
                    # 代号 -> 名称
                    algorithm_map = {
                        "A": "Reputation-Based Aggregation",
                        "B": "Simple Average Aggregation",
                        "C": "Fed-MAE Aggregation",
                        "D": "median_aggregation",
                        "E": "trimmed_mean_aggregation",
                        "F": "krum_aggregation",
                        "G": "multi_krum_aggregation",
                        "H": "bulyan_aggregation",
                    }
                    df_agg_mean["Aggregation Algorithm"] = df_agg_mean[
                        "Aggregation"
                    ].map(algorithm_map)
                else:
                    df_agg_mean = pd.DataFrame()

                # === 合并到一个总表（用“算法名称”作为键）===
                df_final = df_perf.copy()
                df_final["Avg_Train_Time_s"] = avg_train_time
                df_final["Avg_Comm_KB"] = avg_comm_kb
                if not df_agg_mean.empty:
                    df_final = df_final.merge(
                        df_agg_mean.drop(columns=["Aggregation"]),
                        how="left",
                        on="Aggregation Algorithm",
                    )

                df_final.drop(
                    columns=["Avg_Train_Time_s", "Avg_Comm_KB"],
                    errors="ignore",
                    inplace=True,
                )
                df_final = df_final.round(
                    {
                        "Acc": 4,
                        "F1": 4,
                        "AUC": 4,
                        "Total_Time_s": 4,
                        # "Avg_Train_Time_s": 4,
                        # "Avg_Comm_KB": 4,
                        "Avg_Agg_Time_s": 4,
                        "RSS_KB": 2,  # 内存保留两位即可
                    }
                )
                csv_path = "final_experiment_summary.csv"
                if os.path.exists(csv_path):
                    old = pd.read_csv(csv_path)
                    combined = pd.concat([old, df_final], ignore_index=True)
                    combined.to_csv(csv_path, index=False)
                else:
                    df_final.to_csv(csv_path, index=False)
                print("✅ Unified results saved to final_experiment_summary.csv")

                print("Program is about to terminate")
                sys.exit()
            self.new_model_event.clear()

    def update_reputation(self, drone_id, new_reputation):
        self.reputation[drone_id] = new_reputation

    def aggregate_models(self, models_to_aggregate, use_reputation):
        if use_reputation == "A":
            aggregated_model = weighted_average_aggregation(
                models_to_aggregate, self.reputation
            )
        elif use_reputation == "B":
            aggregated_model = average_aggregation(models_to_aggregate)

        elif use_reputation == "C":
            aggregated_model = fed_mae_aggregation(models_to_aggregate)

        elif use_reputation == "D":
            aggregated_model = median_aggregation(models_to_aggregate)

        elif use_reputation == "E":
            aggregated_model = trimmed_mean_aggregation(models_to_aggregate)
        elif use_reputation == "F":
            aggregated_model = krum_aggregation(models_to_aggregate)
        elif use_reputation == "G":
            aggregated_model = multi_krum_aggregation(models_to_aggregate)
        elif use_reputation == "H":
            aggregated_model = bulyan_aggregation(models_to_aggregate)

        self.aggregated_global_model = Net18(num_output_features).to(device)
        self.aggregated_global_model.load_state_dict(aggregated_model)
        torch.save(
            self.aggregated_global_model.state_dict(), "aggregated_global_model.pt"
        )

    def send_model(self, ip, model_type="global_model"):
        if model_type == "aggregated_global_model":
            buffer = io.BytesIO()
            torch.save(self.aggregated_global_model.state_dict(), buffer)
            global_model_serialized = buffer.getvalue()
            global_model_serialized_base64 = base64.b64encode(
                global_model_serialized
            ).decode()
            # **NEW: 记录最近一次广播**
            import copy

            self.last_broadcast_state = copy.deepcopy(
                self.aggregated_global_model.state_dict()
            )
            response = requests.post(
                f"http://{ip}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            return json.dumps({"status": "success"})
        else:
            model_path = "global_model.pt"
            os.path.isfile(model_path)
            self.global_model = Net18(num_output_features).to(device)
            self.global_model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            print(f"本地存在 {model_type}，发送中…… ")
            # else:
            #     print(f"本地不存在 {model_type}，训练中……")
            #     self.initialize_global_model()
            # **NEW: 记录最近一次广播**
            import copy

            self.last_broadcast_state = copy.deepcopy(self.global_model.state_dict())
            buffer = io.BytesIO()
            torch.save(self.global_model.state_dict(), buffer)
            global_model_serialized = buffer.getvalue()
            global_model_serialized_base64 = base64.b64encode(
                global_model_serialized
            ).decode()

            response = requests.post(
                f"http://{ip}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            return json.dumps({"status": "success"})

    def compute_r1_model_similarity(self, local_model, aggregated_global_model):
        """
        通过比较模型的权重（state_dict）计算一致性（R1）
        """
        # 获取 local_model 和 aggregated_global_model 的 state_dict
        local_model_state_dict = local_model.state_dict()
        aggregated_global_model_state_dict = aggregated_global_model.state_dict()

        # 计算两者之间的差异
        diff = 0.0
        count = 0
        for param_name in local_model_state_dict:
            if param_name in aggregated_global_model_state_dict:
                # 计算该参数的差异
                param_diff = torch.mean(
                    torch.abs(
                        local_model_state_dict[param_name]
                        - aggregated_global_model_state_dict[param_name]
                    )
                )
                diff += param_diff.item()  # 累加所有参数的差异
                count += 1  # 计数

        # 计算 R1 一致性：1 / (1 + diff / count)，越接近越好
        if count > 0:
            return 1.0 / (1.0 + diff / count)
        else:
            return 0.0  # 如果没有匹配的参数，返回0

    # # 计算性能惩罚因子
    # def compute_penalty_factor(self, performance, performance_threshold):
    #     # 如果性能低于阈值，计算基于差异的惩罚因子
    #     if performance < performance_threshold:
    #         penalty_factor = 1 / (1 + np.exp(performance - performance_threshold))
    #         return penalty_factor  # 惩罚因子越小，节点贡献越小
    #     else:
    #         # 高性能节点不给与惩罚
    #         return 1

    # def compute_reputation(
    #     self,
    #     drone_id,
    #     local_model,
    #     aggregated_global_model,
    #     performance,
    #     data_age,
    #     performance_threshold=0.67,
    # ):

    #     #  Grpah structure similarity
    #     # 动态计算 threshold，可以用全局平均 performance 或当前最差 performance
    #     if performance_threshold is None:
    #         performance_threshold = self.get_dynamic_threshold()

    #     R1 = self.compute_r1_model_similarity(local_model, aggregated_global_model)
    #     R1 = 0.1 if R1 < 0.6 else 0.99
    #     # performance_contribution
    #     R2 = sigmoid(performance)
    #     # data_age_contribution
    #     R3 = exponential_decay(data_age)
    #     # large model output
    #     # R4 = self.large_model_output

    #     if performance < performance_threshold:
    #         self.low_performance_counts[drone_id] = (
    #             self.low_performance_counts.get(drone_id, 0) + 1
    #         )
    #         print("low performance node,", drone_id)
    #         R2 *= 0.1
    #     else:
    #         penalty_factor = 1 / (np.exp(self.low_performance_counts.get(drone_id, 0)))
    #         self.low_performance_counts[drone_id] = max(
    #             0, self.low_performance_counts.get(drone_id, 0) - 1
    #         )
    #         print("use penalty factor")
    #         R2 *= penalty_factor

    #     reputation = 0.4 * R2 + 0.5 * R1 + 0.1 * R3
    #     # 将 R1 存入 CSV 文件
    #     result_data1 = {
    #         "drone_id": drone_id,
    #         "R1": R1,
    #         "performance": performance,
    #         "data_age": data_age,
    #     }

    #     result_data2 = {
    #         "drone_id": drone_id,
    #         "R2": R2,
    #         "performance": performance,
    #         "data_age": data_age,
    #     }

    #     # df1 = pd.DataFrame([result_data1])
    #     # df1.to_csv("r1_values.csv", mode="a", header=False, index=False)
    #     # df2 = pd.DataFrame([result_data2])
    #     # df2.to_csv("r2_values.csv", mode="a", header=False, index=False)

    #     if self.low_performance_counts.get(drone_id, 0) > 2:
    #         print(f"🛑 节点 {drone_id} 连续低性能超过 2 次，声誉设为 0")
    #         reputation = 0.0
    #     result_data3 = {
    #         "drone_id": drone_id,
    #         "reputation": reputation,
    #         "performance": performance,
    #         "data_age": data_age,
    #     }
    #     # df3 = pd.DataFrame([result_data3])
    #     # df3.to_csv("rep_values.csv", mode="a", header=False, index=False)

    #     return reputation
    def compute_reputation(
        self,
        drone_id,
        local_model,
        aggregated_global_model,
        performance,
        data_age,
        performance_threshold=0.67,
    ):
        # **CHANGED: R1（参数一致性）不再二值化，改为平滑 Sigmoid 映射**
        R1_raw = self.compute_r1_model_similarity(local_model, aggregated_global_model)
        alpha_R1 = 10.0  # 陡峭度（可调）
        R1_w = 1.0 / (1.0 + np.exp(-alpha_R1 * (R1_raw - 0.6)))
        R1 = 0.1 + (0.99 - 0.1) * R1_w  # 缩放到 [0.1, 0.99]

        # **CHANGED: R2（性能）仍用 sigmoid，但对“低性能”改为柔和衰减**
        R2 = sigmoid(performance)
        if performance_threshold is None:
            # **NEW: 动态阈值（中位数），需要 perf_history 支持**
            if len(self.perf_history) >= 10:
                perf_vals = [p for _, p in self.perf_history[-50:]]
                performance_threshold = float(np.median(perf_vals))
            else:
                performance_threshold = 0.67

        if performance < performance_threshold:
            cnt = self.low_performance_counts.get(drone_id, 0) + 1
            self.low_performance_counts[drone_id] = cnt
            print("low performance node,", drone_id)
            R2 *= 0.5**cnt  # 次数越多，衰减越强，但不是一次性砍死
        else:
            cnt = self.low_performance_counts.get(drone_id, 0)
            penalty_factor = 1.0 / (1.0 + 0.5 * cnt)
            self.low_performance_counts[drone_id] = max(0, cnt - 1)
            print("use penalty factor")
            R2 *= penalty_factor

        # **NEW: R4（更新方向相似度）+ R5（范数相对大小惩罚）**
        try:
            # 基于最近一次广播的模型，构造“更新向量”
            ref_state = (
                self.last_broadcast_state or aggregated_global_model.state_dict()
            )
            vec_ref0 = self._state_to_vec(ref_state)
            vec_loc = self._state_to_vec(local_model.state_dict())
            vec_agg = self._state_to_vec(aggregated_global_model.state_dict())

            u_local = vec_loc - vec_ref0
            u_ref = vec_agg - vec_ref0

            # 方向一致性（余弦相似度 -> [0,1]）
            R4 = self._cos01(u_local, u_ref)

            # 范数相对大小（>1 说明更离谱，指数惩罚）
            ratio = (u_local.norm().item() + 1e-8) / (u_ref.norm().item() + 1e-8)
            beta = 1.0
            R5 = float(
                np.exp(-beta * max(0.0, ratio - 1.0))
            )  # ratio<=1 ~1.0；超出逐步惩罚
        except Exception as e:
            print(f"[WARN] update-sim features failed: {e}")
            R4, R5 = 0.5, 1.0

        # **原有的 R3（数据时效）**
        R3 = exponential_decay(data_age)

        # **CHANGED: 调整权重，可扫参优化**
        w_R1, w_R2, w_R3, w_R4, w_R5 = 0.40, 0.30, 0.10, 0.15, 0.05
        rep_now = w_R1 * R1 + w_R2 * R2 + w_R3 * R3 + w_R4 * R4 + w_R5 * R5

        # **NEW: EWMA 平滑声誉（对抗瞬时波动/作恶后短暂“洗白”）**
        ema_alpha = 0.3
        rep_prev = self.reputation_ema.get(drone_id, rep_now)
        rep_ewma = (1 - ema_alpha) * rep_prev + ema_alpha * rep_now
        self.reputation_ema[drone_id] = rep_ewma
        reputation = rep_ewma

        # **保留极端保护：连续低性能超 2 轮 -> 直接清零**
        if self.low_performance_counts.get(drone_id, 0) > 2:
            print(f"🛑 节点 {drone_id} 连续低性能超过 2 次，声誉设为 0")
            reputation = 0.0

        return reputation

    def send_model_thread(self, ip, model_type="aggregated_global_model"):
        threading.Thread(target=self.send_model, args=(ip, model_type)).start()

    def run(self, port=5000):
        torch.manual_seed(0)
        np.random.seed(0)
        app = Flask(__name__)

        @app.route("/health_check", methods=["GET"])
        def health_check():
            return jsonify({"status": "OK"})

        @app.route("/getDroneNodeIDS", methods=["GET"])
        def getDroneNodeIDS():
            return jsonify({"number": list(self.drone_nodes.keys())})

        @app.route("/register", methods=["POST"])
        def register():
            drone_id = request.form["drone_id"]
            ip = request.form["ip"]
            print(f"接收到新节点, id: {drone_id}, ip: {ip}")
            self.drone_nodes[drone_id] = ip

            print(f"发送全局模型-执行中---> {ip}")
            self.send_model(ip, "global_model")
            return jsonify({"status": "success"})

        @app.route("/upload_model", methods=["POST"])
        def upload_model():
            drone_id = request.form["drone_id"]
            train_time = float(request.form.get("train_time", 0))
            comm_size = float(request.form.get("comm_size", 0))
            self.global_model = Net18(num_output_features).to(device)

            # 加载 global_model 的权重
            self.global_model.load_state_dict(
                torch.load("global_model.pt", map_location="cpu")
            )

            self.train_comm_log.append(
                {
                    "Round": self.num_aggregations,
                    "Drone_ID": drone_id,
                    "Train_Time_s": train_time,
                    "Comm_Size_KB": comm_size,
                }
            )

            local_model_serialized = request.form["local_model"]
            local_model = pickle.loads(base64.b64decode(local_model_serialized))
            performance = float(request.form["performance"])

            # **NEW: 性能历史（保留最近 200 条即可）**
            self.perf_history.append((drone_id, performance))
            if len(self.perf_history) > 200:
                self.perf_history = self.perf_history[-200:]
            if drone_id in self.data_age:
                self.data_age[drone_id] += 1
            else:
                self.data_age[drone_id] = 1

            # 检查是否为第一次聚合
            if self.aggregated_global_model is None:
                # 如果第一次聚合，使用 global_model
                print(
                    "❌ Aggregated global model is None. Using global model for the first aggregation."
                )
                aggregated_global_model = self.global_model
            else:
                # 如果已经有 aggregated_global_model，使用它
                aggregated_global_model = self.aggregated_global_model

            reputation = self.compute_reputation(
                drone_id,
                local_model,
                aggregated_global_model,
                performance,  # 传递 performance 参数
                self.data_age[drone_id],  # 传递 data_age 参数
            )

            self.update_reputation(drone_id, reputation)

            self.local_models.put({drone_id: local_model})

            self.new_model_event.set()

            return jsonify({"status": "success"})

        app.run(host="localhost", port=port)


if __name__ == "__main__":
    import sys

    # 读取第一个参数 A/B/C，默认 C
    mode = sys.argv[1].upper() if len(sys.argv) > 1 else "C"
    if mode not in ("A", "B", "C", "D", "E", "F", "G", "H"):
        print("Usage: python Central_Server.py [A|B|C|D|E|F|G|H]")
        sys.exit(1)

    # 根据 mode 传入不同的聚合算法
    central_server = CentralServer(use_reputation=mode)
    central_server.run()
