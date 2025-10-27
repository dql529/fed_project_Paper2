import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
R1_list = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 比较模型权重
def compare_model_weights(model1, model2, debug=False):
    # if not hasattr( 'scaler'):
    #     scaler = MinMaxScaler(feature_range=(0, 1))   
    model1_weights = model1.state_dict()
    model2_weights = model2.state_dict()
    
    weight_diffs = {}
    for key in model1_weights.keys():
        if key in model2_weights:
            diff = torch.sum((model1_weights[key] - model2_weights[key]) ** 2).item()
            weight_diffs[key] = diff
            if debug:
                print(f"Difference in {key}: {diff}")
        else:
            if debug:
                print(f"Key {key} not found in model2.")
    
    # 查找 conv2.lin.weight 的差异
    key = 'conv2.lin.weight'
    if key in weight_diffs:
        diff_value = weight_diffs[key]
        # R1=diff_value
        # 计算倒数
        if diff_value != 0:          
            R1 = sigmoid(1 / diff_value)
        else:
            R1 = float('inf')  # 处理diff_value为0的情况，避免除以零
        if debug:
            print(f"R1: {R1}")
    else:
        R1 = 0  # 如果没有找到 conv2.lin.weight 的差异，R1 设置为 0
        if debug:
            print(f"Key {key} not found in weight differences.")
    R1_list.append(R1)
    # StandardScaler process
    R1_scaled = scaler.fit_transform(np.array(R1_list).reshape(-1, 1))[-1, 0]  # Only take the last scaled R1 value
    return R1_scaled
    # return R1
