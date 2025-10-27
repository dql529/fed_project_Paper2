import matplotlib.pyplot as plt

# 替换为你的实际实验结果
rounds = list(range(1, 11))
accuracy_avg = [0.5, 0.65, 0.7, 0.74, 0.78, 0.8, 0.82, 0.83, 0.84, 0.85]
accuracy_mae = [0.5, 0.7, 0.76, 0.79, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89]
accuracy_rep = [0.5, 0.68, 0.74, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88]
accuracy_med = [0.5, 0.69, 0.75, 0.8, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9]  # 示例
accuracy_trim = [0.5, 0.66, 0.73, 0.77, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87]  # 示例

plt.figure()
plt.plot(rounds, accuracy_avg, marker="o", label="Simple Average")
plt.plot(rounds, accuracy_mae, marker="s", label="Fed-MAE")
plt.plot(rounds, accuracy_rep, marker="^", label="Reputation-Hybrid")
plt.plot(rounds, accuracy_med, marker="D", label="Coordinate Median")
plt.plot(rounds, accuracy_trim, marker="v", label="Trimmed Mean")

plt.xlabel("Aggregation Round")
plt.ylabel("Accuracy")
plt.title("Comparison of Aggregation Methods")
plt.legend()
plt.grid(True)
plt.show()
