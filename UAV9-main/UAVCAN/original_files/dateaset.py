import pandas as pd
import os

# 设置文件夹路径
input_directory = "C:/Users/asus/Documents/GitHub/fed_project1/UAVCAN/original_files"
output_directory = "C:/Users/asus/Documents/GitHub/fed_project1/UAVCAN/csv_files"

# 列出所有需要处理的文件名
filenames = [
    "type1_label.bin",
    "type2_label.bin",
    "type3_label.bin",
    "type4_label.bin",
    "type5_label.bin",
    "type6_label.bin",
    "type7_label.bin",
    "type8_label.bin",
    "type9_label.bin",
    "type10_label.bin",
]


# 重用之前定义的处理函数
def process_can_log(lines):
    processed_data = []
    for line in lines:
        parts = line.split()
        message_type = 0 if parts[0] == "Normal" else 1
        timestamp = float(parts[1].strip("()"))
        can_id = parts[3]
        data_length = int(parts[4].strip("[]"))
        data_content = parts[5:]
        if len(data_content) < 8:
            data_content += ["00"] * (8 - len(data_content))
        elif len(data_content) > 8:
            data_content = data_content[:8]
        processed_data.append([message_type, timestamp, can_id] + data_content)
    columns = ["Type", "Timestamp", "CAN_ID"] + [f"Data_{i+1}" for i in range(8)]
    return pd.DataFrame(processed_data, columns=columns)


# 处理每个文件
for filename in filenames:
    file_path = os.path.join(input_directory, filename)
    with open(file_path, "r") as file:
        lines = file.readlines()
    df_can_logs = process_can_log(lines)
    output_csv_path = os.path.join(output_directory, filename.replace(".bin", ".csv"))
    df_can_logs.to_csv(output_csv_path, index=False)
    print(f"Processed {filename} and saved to {output_csv_path}")
