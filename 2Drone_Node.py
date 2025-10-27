import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # 正确的环境变量名
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Net54, Net18, GAT54, GAT18
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


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from wifi_traffic_dataset.Dataset_18 import n_nodes
from tools import plot_accuracy_vs_epoch
import copy

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2
num_epochs = 3
learning_rate = 0.014


class DroneNode:
    def __init__(self, drone_id):
        self.port = 5001
        self.central_server_ip = "localhost:5000"
        self.drone_id = drone_id
        self.local_data = None
        self.local_model = None
        self.performance = None
        # self.model1 = None  # 新增的model1属性

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
        if self.local_model is None:
            print("Error: No local model is available for training.")
            return

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
            self.local_model.eval()  # Set the self.local_model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
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
            self.local_model.train()  # Set the model to training mode
            outputs = self.local_model(self.data_device)
            loss = compute_loss(outputs, self.data_device.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        self.local_model.load_state_dict(best_model_state_dict)
        self.accuracy = round(best_accuracy, 4)
        self.precision = round(best_precision, 4)
        self.recall = round(best_recall, 4)
        self.f1 = round(best_f1, 4)

    # 保存最好的模型状态到 model1
    # self.model1 = Net18(num_output_features).to(device)
    # self.model1.load_state_dict(best_model_state_dict)

    def upload_local_model(self, central_server_ip):
        local_model_serialized = pickle.dumps(self.local_model)
        local_model_serialized_base64 = base64.b64encode(
            local_model_serialized
        ).decode()

        performance = self.accuracy, self.precision, self.recall, self.f1
        print(self.accuracy)

        response = requests.post(
            f"http://{central_server_ip}/upload_model",
            data={
                "drone_id": self.drone_id,
                "local_model": local_model_serialized_base64,
                "performance": self.accuracy,
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
            f"./wifi_traffic_dataset/data_object/node_train_{self.drone_id}.pt"
        ).to(device)
        self.data_test_device = torch.load(
            f"./wifi_traffic_dataset/data_object/node_test_{self.drone_id}.pt"
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
            print(model)
            print("LOGGER-INFO: global model received")

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

            self.local_model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
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


if __name__ == "__main__":
    drone_id = int(sys.argv[2])

    drone_node_instance = DroneNode(drone_id)
    drone_node_instance.port = sys.argv[1]

    drone_node_instance.drone_id = drone_id
    p1 = Process(target=drone_node_instance.registerToMaster)
    p1.start()
    drone_node_instance.run()
