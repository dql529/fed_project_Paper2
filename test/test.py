import pandas as pd
import numpy as np

# 文件路径列表（相对路径）
file_paths = {
    "Benign_Cyber": "Benign_Cyber.csv",
    "DoS_Attack_Cyber": "DoS_Attack_Cyber.csv",
    "Evil_Twin_Cyber": "Evil_Twin_Cyber.csv",
    "FDI_Cyber": "FDI_Cyber.csv",
    "Replay_Attack_Cyber": "Replay_Attack_Cyber.csv",
}

# 初始化字典存储数据
cyber_data = {}

# 读取数据
for name, path in file_paths.items():
    df = pd.read_csv(path)
    cyber_data[name] = df


# 同步和融合网络数据
def combine_cyber_datasets(dfs, timestamp_col="timestamp_c"):
    combined = []
    labels = []

    # 获取所有唯一的时间戳
    all_timestamps = np.unique(
        np.concatenate([df[timestamp_col].values for df in dfs.values()])
    )

    # 初始化最后一个数据点
    last_rows = {name: df.iloc[0] for name, df in dfs.items()}

    for timestamp in all_timestamps:
        combined_row = {}
        for name, df in dfs.items():
            if timestamp in df[timestamp_col].values:
                last_rows[name] = df[df[timestamp_col] == timestamp].iloc[-1]
            combined_row.update(last_rows[name].to_dict())

        combined.append(combined_row)

        # 假设 'class' 列表示标签
        class_label = next(
            (
                last_rows[name]["class"]
                for name in last_rows
                if "class" in last_rows[name]
            ),
            None,
        )
        labels.append(class_label)

    combined_df = pd.DataFrame(combined).reset_index(drop=True)
    return combined_df, labels


# 合并所有网络数据集
combined_cyber_features, combined_cyber_labels = combine_cyber_datasets(cyber_data)

# 转换标签为Series
combined_cyber_labels = pd.Series(combined_cyber_labels)

# 输出检查
print("Combined Cyber Features:")
print(combined_cyber_features.head())
print("Combined Cyber Labels:")
print(combined_cyber_labels.head())
