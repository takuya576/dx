import csv
import os

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_1case = pd.read_csv(
    "~/dx/result/data6/2024-01-26_18-40-38/history.csv", header=None
)
df_sig = pd.read_csv(
    "~/dx/result/data6/2024-01-27_15-32-03/history.csv", header=None
)
df_10 = pd.read_csv(
    "/result/data6/2024-01-27_15-01-20/history.csv", header=None
)
df_20 = pd.read_csv(
    "~/dx/result/data6/2024-01-27_14-31-09/history.csv", header=None
)
df_50 = pd.read_csv(
    "~/dx/result/data6/2024-01-27_13-46-44/history.csv", header=None
)

num_epochs = len(df_1case)
if num_epochs < 10:
    unit = 1
else:
    unit = num_epochs / 10

# print(df_1case.iloc[:, 0])
# 学習曲線の表示 (精度)
plt.figure(figsize=(9, 8))
plt.plot(df_1case.iloc[:, 0], df_1case.iloc[:, 8], "k", label="SLC")
plt.plot(df_sig.iloc[:, 0], df_sig.iloc[:, 8], "b", label="GMLC(linear)")
plt.plot(df_10.iloc[:, 0], df_10.iloc[:, 8], "c", label="GMLC(sigmoid, a=10)")
plt.plot(df_20.iloc[:, 0], df_20.iloc[:, 8], "y", label="GMLC(sigmoid, a=20)")
plt.plot(df_50.iloc[:, 0], df_50.iloc[:, 8], "r", label="GMLC(sigmoid, a=50)")

plt.xticks(np.arange(0, num_epochs + 1, unit), fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(-0.04, 1.04)
# グリッドラインを追加
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
# plt.title("Learning Curve(Accuracy)", fontsize=24)
plt.legend(fontsize=18, loc="lower right")
plt.tight_layout()
plt.savefig("merged_acc.png")
plt.show()
