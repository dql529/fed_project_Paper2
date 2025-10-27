import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import numpy as np

torch.manual_seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exponential_decay(x, a=0.8):
    return np.exp(-a * x)


def plot_accuracy_vs_epoch(
    accuracies, individual_accuracies, num_epochs, learning_rate
):
    def to_percent(y, position):
        return f"{100*y:.2f}%"

    formatter = FuncFormatter(to_percent)

    plt.figure(figsize=(10, 6))

    # 确保 individual_accuracies 中的每个元素都是列表
    if isinstance(individual_accuracies[0], float):
        individual_accuracies = [[acc] for acc in individual_accuracies]

    # 在这里确保不重复添加初始 0.5
    individual_accuracies_with_initial = individual_accuracies

    # 调整 x 轴，从 1 开始，包含初始的 0.5
    epochs = range(1, len(accuracies) + 1)  # 确保 x 轴与 accuracies 一致

    # 确保 x 和 y 的维度一致
    min_length = min(len(epochs), len(accuracies))
    epochs = list(epochs)[:min_length]
    accuracies = accuracies[:min_length]

    plt.plot(
        epochs,
        accuracies,  # y 轴数据现在也从 1 开始
        marker="o",
        linestyle="-",
        color="blue",
        linewidth=2,
        label="Server model based on reputation aggregation",
    )

    # 绘制每个节点的模型准确率
    colors = ["red", "green", "orange", "purple", "cyan", "brown", "pink", "grey"]
    markers = ["o", "s", "^", "v", "p", "*", "+", "x"]

    for i, acc_list in enumerate(individual_accuracies_with_initial):
        # 确保节点准确率的长度和 x 轴一致
        acc_list = acc_list[:min_length]
        plt.plot(
            epochs,
            acc_list,
            marker=markers[i % len(markers)],
            linestyle="--",
            color=colors[i % len(colors)],
            linewidth=1,
            label=f"Local GNN model {i + 1}",
            alpha=0.5,
        )

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Model Aggregation", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlim([1, len(epochs)])  # 确保 x 轴范围从 1 到最后一个聚合次数
    plt.ylim([0, 1])

    # Annotate the maximum accuracy point
    max_accuracy = max(accuracies)
    max_epoch = accuracies.index(max_accuracy) + 1  # x 轴从 1 开始，所以加 1

    plt.annotate(
        f"Max Accuracy: {100*max_accuracy:.2f}%",
        xy=(max_epoch, max_accuracy),
        xytext=(max_epoch + 0.5, max_accuracy - 0.05),
        arrowprops=dict(facecolor="red", shrink=0.05),
    )

    plt.tight_layout()
    plt.savefig("plot.png", format="png")
