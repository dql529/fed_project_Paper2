import numpy as np
from flask import Flask, request, jsonify
from Model import Net18
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
from time import sleep
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

os.chdir("./wifi_traffic_dataset")
torch.manual_seed(0)

n_nodes = 8
node_train = []
node_test = []
for i in range(1, n_nodes + 1):
    node_train.append(torch.load(f"data_object/node_train_{i}.pt"))
    node_test.append(torch.load(f"data_object/node_test_{i}.pt"))


server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_device = server_train.to(device)
data_test_device = node_test[3].to(device)


# 定义输出维度
num_output_features = 2
num_epochs = 100
learning_rate = 0.02
model = Net18(num_output_features).to(device)
model.load_state_dict(torch.load("global_model.pt"))


# 训练参数
# 定义损失函数和优化器


if model.num_output_features == 1:
    criterion = nn.BCEWithLogitsLoss()
elif model.num_output_features == 2:
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError(
        "Invalid number of output features: {}".format(model.num_output_features)
    )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


if device.type == "cuda":
    print(
        f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}"
    )
else:
    print(f"Using device: {device}")


def compute_loss(outputs, labels):
    if model.num_output_features == 1:
        return criterion(outputs, labels)
    elif model.num_output_features == 2:
        return criterion(outputs, labels.squeeze().long())
    else:
        raise ValueError(
            "Invalid number of output features: {}".format(model.num_output_features)
        )


# Convert the model's output probabilities to binary predictions
def to_predictions(outputs):
    if model.num_output_features == 1:
        return (torch.sigmoid(outputs) > 0.5).float()
    elif model.num_output_features == 2:
        return outputs.argmax(dim=1)
    else:
        raise ValueError(
            "Invalid number of output features: {}".format(model.num_output_features)
        )


model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Do not calculate gradients to save memory
    outputs_test = model(data_test_device)

    predictions_test = to_predictions(outputs_test)

    # Calculate metrics
    accuracy = accuracy_score(data_test_device.y.cpu(), predictions_test.cpu())
    precision = precision_score(data_test_device.y.cpu(), predictions_test.cpu())
    recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
    f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
    print(f"Accuracy of received model: {accuracy}")
    print(f"Precision of received model: {precision}")
    print(f"Recall of received model: {recall}")
    print(f"F1 Score of received model: {f1}")
