import numpy as np
from flask import Flask, request, jsonify
import sys
import os
import random
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
from time import sleep
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model import Net18, GAT18
from sklearn.preprocessing import StandardScaler

# Fixing random seeds for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # to ensure reproducibility
    torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define output dimension
num_output_features = 2
num_epochs = 30
learning_rate = 0.01

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(seed)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

model = GAT18(num_output_features).to(device)
model.apply(weight_init) 
criterion = nn.CrossEntropyLoss()


# 读取训练集和测试集数据
df_train = pd.read_csv("./wifi_traffic_dataset/train_test_file/train.csv", sep=" ")
df_test = pd.read_csv("./wifi_traffic_dataset/train_test_file/test.csv", sep=" ")

# 提取训练集的特征和标签
train_features = df_train.iloc[:, :18]
train_labels = df_train.iloc[:, 18]
test_features = df_test.iloc[:, :18]
test_labels = df_test.iloc[:, 18]


# # 标准化处理训练集特征
# scaler = StandardScaler()
# train_features_scaled = scaler.fit_transform(train_features)
# # 将标准化后的训练集特征和标签重新组合成一个DataFrame
# df_train_scaled = pd.DataFrame(train_features_scaled, columns=df_train.columns[:18])
# df_train_scaled['label'] = train_labels.values

# # 使用训练集的标准化器对测试集进行标准化
# test_features_scaled = scaler.transform(test_features)
# # 将标准化后的测试集特征和标签重新组合成一个DataFrame
# df_test_scaled = pd.DataFrame(test_features_scaled, columns=df_test.columns[:18])
# df_test_scaled['label'] = test_labels.values

# train_features = df_train_scaled.iloc[:, :18]
# train_labels = df_train_scaled.iloc[:, 18]
# test_features = df_test_scaled.iloc[:, :18]
# test_labels = df_test_scaled.iloc[:, 18]

# Define the adjacency matrix for the feature computational dependencies
adjacency_matrix = torch.tensor(
    [
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.float,
)

# Convert the adjacency matrix to edge index format
edges = adjacency_matrix.nonzero(as_tuple=False).t()

data = Data(
    x=torch.tensor(train_features.values.reshape(-1, 18), dtype=torch.float32),
    edge_index=edges,
    y=torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32),
)

data_test = Data(
    x=torch.tensor(test_features.values.reshape(-1, 18), dtype=torch.float32),
    edge_index=edges,
    y=torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32),
)

data_device = data.to(device)
data_test_device = data_test.to(device)

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
    print(f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}")
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

def to_predictions(outputs):
    if model.num_output_features == 1:
        return (torch.sigmoid(outputs) > 0.5).float()
    elif model.num_output_features == 2:
        return outputs.argmax(dim=1)
    else:
        raise ValueError(
            "Invalid number of output features: {}".format(model.num_output_features)
        )

def evaluate(data_test_device):
    model.eval()
    with torch.no_grad():
        outputs_test = model(data_test_device)
        predictions_test = to_predictions(outputs_test)
        accuracy = accuracy_score(data_test_device.y.cpu(), predictions_test.cpu())
        precision = precision_score(data_test_device.y.cpu(), predictions_test.cpu())
        recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
        f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
        return accuracy, precision, recall, f1

# Train the model and evaluate at the end of each epoch
accuracies = []
best_accuracy = 0.0
best_model_state_dict = None

# Logging the initial state of the model
initial_model_state = copy.deepcopy(model.state_dict())
print("Initial model state:", initial_model_state)

for epoch in range(num_epochs):
    model.train()
    outputs = model(data_device)
    loss = compute_loss(outputs, data_device.y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy, precision, recall, f1 = evaluate(data_test_device)
    accuracies.append(accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Loss: {loss.item()}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = copy.deepcopy(model.state_dict())

# Logging the final state of the model
final_model_state = copy.deepcopy(model.state_dict())
print("Final model state:", final_model_state)

model.load_state_dict(best_model_state_dict)

# Save the best model to a file
torch.save(model.state_dict(), "global_model.pt")

# Find the maximum accuracy and its corresponding epoch
max_accuracy = max(accuracies)
max_epoch = accuracies.index(max_accuracy) + 1
print(f"learning rate {learning_rate}, epoch {num_epochs} and dimension {model.num_output_features}, Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}")


def to_percent(y, position):
    return f"{100*y:.2f}%"

formatter = FuncFormatter(to_percent)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), accuracies, marker="o", linestyle="-", color="b", label="Accuracy")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("Accuracy vs. Epoch", fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlim([1, num_epochs])
plt.ylim([0, 1])

# Annotate the maximum point
plt.annotate(
    f"Max Accuracy: {100*max_accuracy:.2f}%",
    xy=(max_epoch, max_accuracy),
    xytext=(max_epoch + 5, max_accuracy - 0.1),
    arrowprops=dict(facecolor="red", shrink=0.05),
)

plt.show()

# 按照学习率，维度，epoch给模型命名
model_name = f"{learning_rate}_{model.num_output_features}_{num_epochs}_{100*max_accuracy:.2f}.pt"

