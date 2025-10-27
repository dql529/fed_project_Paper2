import pandas as pd
import os

# 文件路径列表（相对路径）
file_paths = {
    "Benign_Cyber": "Benign_Cyber.csv",
    "Benign_Physical": "Benign_Physical.csv",
    "DoS_Attack_Cyber": "DoS_Attack_Cyber.csv",
    "DoS_Attack_Physical": "DoS_Attack_Physical.csv",
    "Evil_Twin_Cyber": "Evil_Twin_Cyber.csv",
    "Evil_Twin_Physical": "Evil_Twin_Physical.csv",
    "FDI_Cyber": "FDI_Cyber.csv",
    "FDI_Physical": "FDI_Physical.csv",
    "Replay_Attack_Cyber": "Replay_Attack_Cyber.csv",
    "Replay_Attack_Physical": "Replay_Attack_Physical.csv",
}

# 创建字典存储各子数据集的特征和标签
sub_data_features = {}
labels = []

# 读取各个子数据集，提取特征和标签
for name, path in file_paths.items():
    df = pd.read_csv(path)
    # 提取特征
    if "timestamp_c" in df.columns:
        df = df.drop(columns=["timestamp_c"])
    sub_data_features[name] = df

    # 提取标签
    label_length = len(df)
    if "Benign" in name:
        labels.extend([0] * label_length)
    elif "DoS_Attack" in name:
        labels.extend([1] * label_length)
    elif "Evil_Twin" in name:
        labels.extend([2] * label_length)
    elif "FDI" in name:
        labels.extend([3] * label_length)
    elif "Replay_Attack" in name:
        labels.extend([4] * label_length)

# 将各子数据集的特征分别保存到CSV文件
for name, features in sub_data_features.items():
    features.to_csv(f"{name}_features.csv", index=False)

# 将所有标签保存到一个CSV文件
labels_df = pd.DataFrame(labels, columns=["label"])
labels_df.to_csv("labels.csv", index=False)

# 输出检查
for name, features in sub_data_features.items():
    print(f"{name} features:")
    print(features.head())
print("Labels:")
print(labels_df.head())
