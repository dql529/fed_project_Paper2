import numpy as np
from flask import Flask, request, jsonify
from Model import Net18
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import random
from time import sleep
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data
import pandas as pd
import copy
import sys
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# 定义输出维度
num_output_features = 2
num_epochs = 1000
learning_rate = 0.02
model = Net18(num_output_features).to(device)
num_output_features = 2
criterion = nn.CrossEntropyLoss()

server_train = torch.load("data_object/server_train.pt")
server_test = torch.load("data_object/server_test.pt")
# 写法和drone node.py有区别   Drone node 中定义在clss中，根据self来调用，此处为了方便，直接定义在函数中
data_device = server_train.to(device)
data_test_device = server_test.to(device)
# 训练参数
# 定义损失函数和优化器


criterion = nn.CrossEntropyLoss()

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


# Evaluate the model on the test data
def evaluate(data_test_device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradients to save memory
        outputs_test = model(data_test_device)

        predictions_test = to_predictions(outputs_test)

        # Calculate metrics
        accuracy = accuracy_score(data_test_device.y.cpu(), predictions_test.cpu())
        precision = precision_score(data_test_device.y.cpu(), predictions_test.cpu())
        recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
        f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
        return accuracy, precision, recall, f1


# Train the model and evaluate at the end of each epoch
import matplotlib.pyplot as plt

# 训练模型并记录每个epoch的准确率
accuracies = []
best_accuracy = 0.0
best_model_state_dict = None
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    outputs = model(data_device)

    loss = compute_loss(outputs, data_device.y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.5)  # 添加噪声

    print("Malicious behavior: Model quality has been intentionally degraded.")

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
# After training, load the best model weights
model.load_state_dict(best_model_state_dict)
# Save the best model to a file
# torch.save(model.state_dict(), "global_model.pt")
# Find the maximum accuracy and its corresponding epoch
max_accuracy = max(accuracies)
max_epoch = accuracies.index(max_accuracy) + 1
# Print the coordinates of the maximum point
print(
    f"learning rate {learning_rate}, epoch {num_epochs} and dimension {model.num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
)

# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter


# # 定义一个函数，将y轴刻度转换为百分比格式，保留两位小数
# def to_percent(y, position):
#     return f"{100*y:.2f}%"


# formatter = FuncFormatter(to_percent)

# plt.figure(figsize=(10, 6))  # Set the figure size
# plt.plot(
#     range(1, num_epochs + 1),
#     accuracies,
#     marker="o",
#     linestyle="-",
#     color="b",
#     label="Accuracy",
# )  # Plot accuracy
# plt.xlabel("Epoch", fontsize=14)  # Set the label for the x-axis
# plt.ylabel("Accuracy", fontsize=14)  # Set the label for the y-axis
# plt.title("Accuracy vs. Epoch", fontsize=16)  # Set the title
# plt.grid(True)  # Add grid lines
# plt.legend(fontsize=12)  # Add a legend
# plt.xticks(fontsize=12)  # Set the size of the x-axis ticks
# plt.yticks(fontsize=12)  # Set the size of the y-axis ticks
# plt.gca().yaxis.set_major_formatter(formatter)  # Set the formatter for the y-axis
# plt.xlim([1, num_epochs])  # Set the range of the x-axis
# plt.ylim([0, 1])  # Set the range of the y-axis

# # Find the maximum accuracy and its corresponding epoch
# max_accuracy = max(accuracies)
# max_epoch = accuracies.index(max_accuracy) + 1

# # Print the coordinates of the maximum point
# print(
#     f"learning rate {learning_rate}, epoch {num_epochs} and dimension {model.num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
# )


# # Annotate the maximum point
# plt.annotate(
#     f"Max Accuracy: {100*max_accuracy:.2f}%",
#     xy=(max_epoch, max_accuracy),
#     xytext=(max_epoch + 5, max_accuracy - 0.1),
#     arrowprops=dict(facecolor="red", shrink=0.05),
# )

# plt.show()


# import os

# # 按照学习率，维度，epoch给模型命名
# model_name = f"{learning_rate}_{model.num_output_features}_{num_epochs}_{100*max_accuracy:.2f}.pt"
