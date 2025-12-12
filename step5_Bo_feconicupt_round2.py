# -*- coding: utf-8 -*-
"""
基于第 1 轮 Fe–Co–Ni–Cu–Pt (Pt=20%) 20 组实验数据，
使用高斯过程 + Expected Improvement，从离散候选池中
选出下一轮 20 组组合。

使用前：
1. 准备 round1_results.csv，包含列：
   Fe,Co,Ni,Cu,Pt,overpotential_mV
2. pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from itertools import product

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

# ===== 1. 读入第 1 轮实验结果 =====
round1_path = r"E:\AMsystem\Project\QCH-RAG\data\round1_results.csv"
df = pd.read_csv(round1_path, encoding="utf-8-sig")

# 只用 Fe,Co,Ni,Cu 作为输入特征（Pt 固定 20）
X_train = df[["Fe", "Co", "Ni", "Cu"]].values.astype(float)

# 目标：overpotential_mV，越小越好
y_train = df["overpotential_mV"].values.astype(float)

# ===== 2. 构建候选成分池（5% 步进） =====
# Fe,Co,Ni,Cu ∈ {0,1,2,...,80} 且总和 = 80
# 总组合数约 88521，依然可接受，但会比 5% 步进慢一些
candidates = []
total_target = 80

for fe in range(0, total_target + 1):
    for co in range(0, total_target - fe + 1):
        for ni in range(0, total_target - fe - co + 1):
            cu = total_target - fe - co - ni
            candidates.append((fe, co, ni, cu))

cand_array = np.array(candidates, dtype=float)

print(f"1% 步进候选池数量：{len(cand_array)}")

# 把第 1 轮已经做过的点从候选池中剔除
# （如果你的 round1 是 5% 步进生成的，这个匹配是严格的）
done_set = set(
    tuple(row)
    for row in X_train
)
mask = np.array(
    [tuple(row) not in done_set for row in cand_array],
    dtype=bool
)
cand_array = cand_array[mask]

print(f"候选池总数：{len(candidates)}，未做过的候选点数：{len(cand_array)}")

# ===== 3. 训练高斯过程代理模型 =====
# 简单的 Matern 核 + 噪声
kernel = ConstantKernel(1.0, (0.1, 10.0)) * \
         Matern(length_scale=20.0, length_scale_bounds=(1.0, 100.0), nu=2.5) + \
         WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=0.0,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=42,
)

gp.fit(X_train, y_train)

# 当前最优过电位（越小越好）
y_best = np.min(y_train)
print("当前最小过电位 (mV)：", y_best)


# ===== 4. 定义 Expected Improvement (EI) =====
def expected_improvement(X_cand, model, y_best, xi=0.01):
    """
    X_cand: (N, d) 候选点
    model: 已训练的 GP
    y_best: 当前最优（最小）目标值
    xi: 探索参数（越大越“敢冒险”）
    返回：每个候选点对应的 EI 值（越大越好）
    """
    mu, sigma = model.predict(X_cand, return_std=True)
    sigma = sigma.reshape(-1, 1)

    # 防止除 0
    sigma = np.maximum(sigma, 1e-9)

    # 我们在最小化 y -> so improvement = y_best - y
    imp = y_best - mu.reshape(-1, 1) - xi
    Z = imp / sigma

    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    # 负值设为 0
    ei[sigma == 0.0] = 0.0

    return ei.ravel()


# ===== 5. 对候选池计算 EI，选出 top-20 作为下一轮 =====
ei_values = expected_improvement(cand_array, gp, y_best=y_best, xi=0.01)

# 根据 EI 从大到小排序
idx_sorted = np.argsort(-ei_values)
top_k = 20
top_idx = idx_sorted[:top_k]

next_round = cand_array[top_idx]

# 组装成 DataFrame，补上 Pt=20
df_next = pd.DataFrame(next_round, columns=["Fe", "Co", "Ni", "Cu"])
df_next["Pt"] = 20.0

print("\n推荐的下一轮 20 组 FeCoNiCuPt 成分（原子百分数）：")
print(df_next)

# 保存到 CSV 方便你做实验
save_path = r"E:\AMsystem\Project\QCH-RAG\data\round2_candidates_bo.csv"
df_next.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n已保存至：{save_path}")
