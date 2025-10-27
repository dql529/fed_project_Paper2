# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# import os
# from sklearn.preprocessing import StandardScaler

# torch.manual_seed(0)
# df = pd.read_csv("wifi_traffic_dataset/data456.csv", sep=" ")

# # 提取特征和标签
# features = df.iloc[:, :18]
# labels = df.iloc[:, 18]

# # 标准化处理特征
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# # 将标准化后的特征和标签重新组合成一个DataFrame
# df_scaled = pd.DataFrame(features_scaled, columns=df.columns[:18])
# df_scaled['label'] = labels.values
# df=df_scaled
# # 查看处理后的数据
# print(df_scaled.head())

# # 按9：1 划分训练，测试集

# train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)
# # 写入train_df文件到train_test_file_54文件夹中
# train_df.to_csv("wifi_traffic_dataset/train_test_file/train.csv", sep=" ", index=False)

# # 写入test_df文件
# test_df.to_csv(
#     "wifi_traffic_dataset/train_test_file/test.csv", sep=" ", index=False
# )  # index=False表示不将索引写入文件中



# # 检查有没有 train_test_file_54 文件夹，如果没有就创建一个
# if not os.path.exists("wifi_traffic_dataset/train_test_file"):
#     os.mkdir("wifi_traffic_dataset/train_test_file")

# # # 假设你有一个服务器和n个节点
n_nodes = 8

# # 首先，我们将数据划分为服务器的数据和节点的数据
# server_data, nodes_data = train_test_split(df, test_size=0.6)

# # 然后，我们将节点的数据进一步划分为n个部分
# nodes_data_splits = np.array_split(nodes_data, n_nodes)

# # 保存服务器的数据
# server_data.to_csv("wifi_traffic_dataset/train_test_file/server_data.csv", index=False)

# # 保存每个节点的数据
# for i, node_data in enumerate(nodes_data_splits):
#     node_data.to_csv(f"wifi_traffic_dataset/train_test_file/node_data_{i+1}.csv", index=False)


# adjacency_matrix = torch.tensor(
#     [
#         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ],
#     dtype=torch.float,
# )
# # Convert the adjacency matrix to edge index format
# edges = adjacency_matrix.nonzero(as_tuple=False).t()


# # 读取服务器的数据
# server_data = pd.read_csv("wifi_traffic_dataset/train_test_file/server_data.csv")

# # 划分服务器的训练集和测试集
# server_train, server_test = train_test_split(server_data, test_size=0.2)

# # 划分服务器的特征和标签
# server_train_features = server_train.iloc[:, :18]
# server_train_labels = server_train.iloc[:, 18]
# server_test_features = server_test.iloc[:, :18]
# server_test_labels = server_test.iloc[:, 18]
# server_train = Data(
#     x=torch.tensor(server_train_features.values.reshape(-1, 18), dtype=torch.float32),
#     edge_index=edges,
#     y=torch.tensor(server_train_labels.values.reshape(-1, 1), dtype=torch.float32),
# )

# server_test = Data(
#     x=torch.tensor(server_test_features.values.reshape(-1, 18), dtype=torch.float32),
#     edge_index=edges,
#     y=torch.tensor(server_test_labels.values.reshape(-1, 1), dtype=torch.float32),
# )

# # 检查有没有 data_object_54 文件夹，如果没有就创建一个
# if not os.path.exists("wifi_traffic_dataset/data_object"):
#     os.mkdir("wifi_traffic_dataset/data_object")

# torch.save(server_train, "wifi_traffic_dataset/data_object/server_train.pt")
# torch.save(server_test, "wifi_traffic_dataset/data_object/server_test.pt")


# # 对于每个节点，进行同样的操作
# for i in range(1, n_nodes + 1):
#     # 读取节点的数据
#     node_data = pd.read_csv(f"wifi_traffic_dataset/train_test_file/node_data_{i}.csv")

#     # 划分节点的训练集和测试集
#     node_train, node_test = train_test_split(node_data, test_size=0.2)

#     # 划分节点的特征和标签
#     node_train_features = node_train.iloc[:, :18]
#     node_train_labels = node_train.iloc[:, 18]
#     node_test_features = node_test.iloc[:, :18]
#     node_test_labels = node_test.iloc[:, 18]

#     node_train_i = Data(
#         x=torch.tensor(node_train_features.values.reshape(-1, 18), dtype=torch.float32),
#         edge_index=edges,
#         y=torch.tensor(node_train_labels.values.reshape(-1, 1), dtype=torch.float32),
#     )
#     node_test_i = Data(
#         x=torch.tensor(node_test_features.values.reshape(-1, 18), dtype=torch.float32),
#         edge_index=edges,
#         y=torch.tensor(node_test_labels.values.reshape(-1, 1), dtype=torch.float32),
#     )
#     torch.save(node_train_i, f"wifi_traffic_dataset/data_object/node_train_{i}.pt")
#     torch.save(node_test_i, f"wifi_traffic_dataset/data_object/node_test_{i}.pt")
