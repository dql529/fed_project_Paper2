import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # 正确的环境变量名
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Net54, Net18, GAT18, GAT54
import pickle
import io
import base64
from flask import Flask, request, jsonify
from multiprocessing import *
import numpy as np
import random
import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import copy
from wifi_traffic_dataset.Dataset_18 import n_nodes
from tools import plot_accuracy_vs_epoch

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
num_output_features = 2


class MaliciousDroneNode:
    def __init__(self, drone_id):
        self.port = 5001
        self.central_server_ip = "localhost:5000"
        self.drone_id = drone_id
        self.local_data = None
        self.local_model = None
        self.performance = None
        # **NEW: 攻击相位与强度**
        self.round_idx = 0
        self.warmup_rounds = 2  # 前几轮“装正常”，骗取声誉
        self.max_amp = 0.20  # 方向放大极限（可调，小于 0.3 更隐匿）
        self.amp_step = 0.04  # 每轮增长步长
        self.noise_base = 0.00  # 高斯噪声起点（可设 0）
        self.noise_step = 0.01  # 每轮噪声递增（小而隐匿）

    def receive_global_model(self, global_model):
        self.global_model = global_model
        self.local_model = Net18(num_output_features).to(device)
        self.local_model.load_state_dict(global_model.state_dict())

        # # 使用 GNN 输出特征训练 SVM 模型
        # gnn_output = self.local_model(self.data_device).detach().cpu().numpy()
        # labels = self.data_device.y.cpu().numpy()
        # self.svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        # self.svm_model.fit(gnn_output, labels)

    def train_local_model(self):
        start_time = time.time()
        if self.local_model is None:
            print("Error: No local model is available for training.")
            return

        num_epochs = 1
        learning_rate = 0.01
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)

        if self.local_model.num_output_features == 1:
            criterion = nn.BCEWithLogitsLoss()
        elif self.local_model.num_output_features == 2:
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    self.local_model.num_output_features
                )
            )

        if device.type == "cuda":
            print(
                f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}"
            )
        else:
            print(f"Using device: {device}")

        def compute_loss(outputs, labels):
            if self.local_model.num_output_features == 1:
                return criterion(outputs, labels)
            elif self.local_model.num_output_features == 2:
                return criterion(outputs, labels.squeeze().long())
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        self.local_model.num_output_features
                    )
                )

        def to_predictions(outputs):
            if self.local_model.num_output_features == 1:
                return (torch.sigmoid(outputs) > 0.5).float()
            elif self.local_model.num_output_features == 2:
                return outputs.argmax(dim=1)
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        self.local_model.num_output_features
                    )
                )

        def evaluate(data_test_device):
            self.local_model.eval()
            with torch.no_grad():
                outputs_test = self.local_model(data_test_device)
                predictions_test = to_predictions(outputs_test)

                accuracy = accuracy_score(
                    data_test_device.y.cpu(), predictions_test.cpu()
                )
                precision = precision_score(
                    data_test_device.y.cpu(), predictions_test.cpu()
                )
                recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
                f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
                return accuracy, precision, recall, f1

        accuracies = []
        best_accuracy = 0.0
        best_model_state_dict = None
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        for epoch in range(num_epochs):
            self.local_model.train()
            outputs = self.local_model(self.data_device)
            loss = compute_loss(outputs, self.data_device.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                amp = 0.0
                noise_std = 0.0
                if self.round_idx > self.warmup_rounds:
                    amp = min(
                        self.max_amp,
                        (self.round_idx - self.warmup_rounds) * self.amp_step,
                    )
                    noise_std = (
                        self.noise_base
                        + (self.round_idx - self.warmup_rounds) * self.noise_step
                    )
                # 方向：相对于“参考全局模型”的更新方向，轻微往“远离”方向推进
                if hasattr(self, "ref_model_for_attack"):
                    ref_sd = self.ref_model_for_attack.state_dict()
                    for name, p in self.local_model.named_parameters():
                        if name in ref_sd and torch.is_floating_point(p):
                            ref = ref_sd[name].to(p.device).data
                            delta = p.data - ref
                            # **方向漂移 + 小噪声（隐匿）**
                            p.add_(amp * delta)
                            if noise_std > 0:
                                p.add_(torch.randn_like(p) * noise_std)

            print("Malicious behavior: Model quality has been intentionally degraded.")

            accuracy, precision, recall, f1 = evaluate(self.data_test_device)
            accuracies.append(accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Loss: {loss.item()}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1 = f1

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state_dict = copy.deepcopy(self.local_model.state_dict())
                best_precision = precision
                best_recall = recall
                best_f1 = f1

        max_accuracy = max(accuracies)
        max_epoch = accuracies.index(max_accuracy) + 1

        print(
            f"learning rate {learning_rate}, epoch {num_epochs} and dimension {num_output_features}, Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
        )
        self.train_time = round(time.time() - start_time, 2)
        self.local_model.load_state_dict(best_model_state_dict)
        self.accuracy = round(best_accuracy, 4)
        self.precision = round(best_precision, 4)
        self.recall = round(best_recall, 4)
        self.f1 = round(best_f1, 4)

    def upload_local_model(self, central_server_ip):
        local_model_serialized = pickle.dumps(self.local_model)
        local_model_serialized_base64 = base64.b64encode(
            local_model_serialized
        ).decode()
        # ✅ 新增：通信开销（KB）
        self.comm_size = round(len(local_model_serialized_base64) / 1024, 2)
        print(f"📡 Malicious Drone {self.drone_id} upload size: {self.comm_size} KB")
        performance = self.accuracy, self.precision, self.recall, self.f1
        print(self.accuracy)

        response = requests.post(
            f"http://{central_server_ip}/upload_model",
            data={
                "drone_id": self.drone_id,
                "local_model": local_model_serialized_base64,
                "performance": self.accuracy,
                "train_time": str(self.train_time),  # **NEW: 上报训练时长**
                "comm_size": str(self.comm_size),  # **NEW: 上报通信开销**
            },
        )

        if response.json()["status"] == "success":
            print(f"Drone {self.drone_id}: Model uploaded successfully.")
        else:
            print(f"Drone {self.drone_id}: Model upload failed.")

    def registerToMaster(self):
        print("连接到主节点……,本节点端口：" + str(self.port) + "\n")
        response = requests.post(
            f"http://{self.central_server_ip}/register",
            data={"drone_id": str(self.drone_id), "ip": "localhost:" + str(self.port)},
        )

    def config(self, drone_id, local_data):
        self.drone_id = drone_id
        self.local_data = local_data

    def run(self):
        app = Flask(__name__)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.data_device = torch.load(
            f"./wifi_traffic_dataset/Mdata_object/node_train_{self.drone_id}.pt",
            map_location="cpu",
        ).to(device)
        self.data_test_device = torch.load(
            f"./wifi_traffic_dataset/Mdata_object/node_test_{self.drone_id}.pt",
            map_location="cpu",
        ).to(device)

        @app.route("/health_check", methods=["POST"])
        def health_check():
            return jsonify({"status": "OK"})

        @app.route("/receive_model", methods=["POST"])
        def receiveModel():
            model_serialized_base64 = request.form["model"]
            model_serialized = base64.b64decode(model_serialized_base64)
            buffer = io.BytesIO(model_serialized)
            state_dict = torch.load(buffer)

            model = Net18(num_output_features).to(device)
            model.load_state_dict(state_dict)

            self.receive_global_model(model)
            print("LOGGER-INFO: global model received")
            # **NEW: 每轮 +1；保留收到的“参考模型”**
            self.round_idx += 1
            self.ref_model_for_attack = copy.deepcopy(self.global_model).to(device)

            def to_predictions(outputs):
                if self.local_model.num_output_features == 1:
                    return (torch.sigmoid(outputs) > 0.5).float()
                elif self.local_model.num_output_features == 2:
                    return outputs.argmax(dim=1)
                else:
                    raise ValueError(
                        "Invalid number of output features: {}".format(
                            self.local_model.num_output_features
                        )
                    )

            self.local_model.eval()
            with torch.no_grad():
                outputs_test = self.local_model(self.data_test_device)
                predictions_test = to_predictions(outputs_test)

                accuracy = accuracy_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                precision = precision_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                recall = recall_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                f1 = f1_score(self.data_test_device.y.cpu(), predictions_test.cpu())
                print(f"Accuracy of received model: {accuracy}")
                print(f"Precision of received model: {precision}")
                print(f"Recall of received model: {recall}")
                print(f"F1 Score of received model: {f1}")

            print("接收到全局模型，训练中")
            self.train_local_model()
            print("发送本地训练结果至主节点……")
            self.upload_local_model(self.central_server_ip)
            print("发送完毕……")

            return jsonify({"status": "OK"})

        @app.route("/uploadToMaster", methods=["POST"])
        def uploadToMaster(ip=self.central_server_ip):
            ip = request.json["ip"]
            self.upload_local_model(ip)
            return jsonify({"status": "upload to master succeed"})

        app.run(host="localhost", port=self.port)


import threading, time, sys

if __name__ == "__main__":
    drone_id = int(sys.argv[2])
    drone_node_instance = MaliciousDroneNode(drone_id)
    drone_node_instance.port = sys.argv[1]
    drone_node_instance.drone_id = drone_id
    # 1) 先启动 Flask（后台线程，不阻塞主线程）
    flask_thread = threading.Thread(target=drone_node_instance.run, daemon=True)
    flask_thread.start()

    # 3) 再去注册，Server 此时能连接 /receive_model
    drone_node_instance.registerToMaster()

    # 4) 10秒后自动结束进程
    time.sleep(3)
    print("--- Malicious Node timeout reached, exiting. ---")
    os._exit(0)
