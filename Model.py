import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # 正确的环境变量名
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import sys
from wifi_traffic_dataset.Dataset_18 import n_nodes
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""

在这个模型中,16是第一个图卷积层(GCNConv)的输出特征数量。在图神经网络中,每一层都会对节点的特征进行转换,
输出新的特征表示。这个数字决定了转换后的特征数量。

在你的模型中,self.conv1 = GCNConv(18, 16)表示第一个图卷积层接收每个节点的18个特征,并输出16个新的特征。
这16个特征然后被用作下一层(self.conv')的输入。

这个数字可以根据你的具体任务和数据进行调整。更多的特征可能会帮助模型捕捉更复杂的模式,但也可能导致过拟合。
同样,更少的特征可能使模型更简单,但可能无法捕捉到所有重要的信息。这是一个需要通过实验来找到最优值的超参数
"""


# Define a simple GNN with GCN layers
class Net18(torch.nn.Module):
    def __init__(self, num_output_features):
        super(Net18, self).__init__()
        self.conv1 = GCNConv(18, 16)
        self.conv2 = GCNConv(16, num_output_features)  # Binary classification
        self.num_output_features = num_output_features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # print("After conv1:", x)  # 打印特征值
        x = torch.nn.functional.relu(x)
        # print("After ReLU:", x)  # 打印特征值
        x = torch.nn.functional.dropout(x, training=self.training)
        # print("After Dropout:", x)  # 打印特征值
        x = self.conv2(x, edge_index)
        # print("After conv2:", x)  # 打印特征值

        return x


class Net54(torch.nn.Module):
    def __init__(self, num_output_features):
        super(Net54, self).__init__()
        self.conv1 = GCNConv(54, 16)
        self.conv2 = GCNConv(16, num_output_features)  # Binary classification
        self.num_output_features = num_output_features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        # print("After conv1:", x)  # 打印特征值
        x = F.relu(x)
        # print("After ReLU:", x)  # 打印特征值
        x = F.dropout(x, training=self.training)
        # print("After Dropout:", x)  # 打印特征值
        x = self.conv2(x, edge_index)
        # print("After conv2:", x)  # 打印特征值

        return x


class GAT18(torch.nn.Module):
    def __init__(self, num_output_features):
        super(GAT18, self).__init__()
        self.conv1 = GATConv(18, 16, heads=1)
        self.conv2 = GATConv(16, num_output_features, heads=1)  # Binary classification
        self.num_output_features = num_output_features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        print("After conv1:", x)  # 打印特征值
        x = torch.nn.functional.elu(x)
        print("After eLU:", x)  # 打印特征值
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        print("After Dropout:", x)  # 打印特征值
        x = self.conv2(x, edge_index)
        print("After conv2:", x)  # 打印特征值

        return x


class GAT54(torch.nn.Module):
    def __init__(self, num_output_features):
        super(GAT54, self).__init__()
        self.conv1 = GATConv(54, 16, heads=1)
        self.conv2 = GATConv(16, num_output_features, heads=1)  # Binary classification
        self.num_output_features = num_output_features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        print("After conv1:", x)  # 打印特征值
        x = F.elu(x)
        print("After eLU:", x)  # 打印特征值
        x = F.dropout(x, p=0.5, training=self.training)
        print("After Dropout:", x)  # 打印特征值
        x = self.conv2(x, edge_index)
        print("After conv2:", x)  # 打印特征值

        return x


class CNN18(nn.Module):
    def __init__(self, num_output_features):
        super(CNN18, self).__init__()
        self.num_output_features = num_output_features
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 1 * 3, 128)  # Adjusted dimensions
        self.fc2 = nn.Linear(128, num_output_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [batch_size, 16, 3, 6]
        print(f"After conv1: {x.shape}")
        x = self.pool(x)  # [batch_size, 16, 1, 3]
        print(f"After pool1: {x.shape}")
        x = F.relu(self.conv2(x))  # [batch_size, 32, 1, 3]
        print(f"After conv2: {x.shape}")
        # Removing the second pooling layer
        x = x.view(-1, 32 * 1 * 3)  # Adjusted dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN54(nn.Module):
    def __init__(self, num_output_features):
        super(CNN54, self).__init__()
        self.num_output_features = num_output_features
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 1 * 9, 128)  # Adjusted dimensions
        self.fc2 = nn.Linear(128, num_output_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [batch_size, 16, 3, 18]
        print(f"After conv1: {x.shape}")
        x = self.pool(x)  # [batch_size, 16, 1, 9]
        print(f"After pool1: {x.shape}")
        x = F.relu(self.conv2(x))  # [batch_size, 32, 1, 9]
        print(f"After conv2: {x.shape}")
        # Removing the second pooling layer
        x = x.view(-1, 32 * 1 * 9)  # Adjusted dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
