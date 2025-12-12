# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

round1_path = your file path
df = pd.read_csv(round1_path, encoding="utf-8-sig")

X_train = df[["Fe", "Co", "Ni", "Cu"]].values.astype(float)
y_train = df["overpotential_mV"].values.astype(float)

candidates = []
total_target = 80

for fe in range(0, total_target + 1):
    for co in range(0, total_target - fe + 1):
        for ni in range(0, total_target - fe - co + 1):
            cu = total_target - fe - co - ni
            candidates.append((fe, co, ni, cu))

cand_array = np.array(candidates, dtype=float)

print(f"1% 步进候选池数量：{len(cand_array)}")

done_set = set(
    tuple(row)
    for row in X_train
)
mask = np.array(
    [tuple(row) not in done_set for row in cand_array],
    dtype=bool
)
cand_array = cand_array[mask]


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

y_best = np.min(y_train)
print("当前最小过电位 (mV)：", y_best)



def expected_improvement(X_cand, model, y_best, xi=0.01):

    mu, sigma = model.predict(X_cand, return_std=True)
    sigma = sigma.reshape(-1, 1)

    sigma = np.maximum(sigma, 1e-9)

    imp = y_best - mu.reshape(-1, 1) - xi
    Z = imp / sigma

    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0

    return ei.ravel()


ei_values = expected_improvement(cand_array, gp, y_best=y_best, xi=0.01)

idx_sorted = np.argsort(-ei_values)
top_k = 20
top_idx = idx_sorted[:top_k]

next_round = cand_array[top_idx]


df_next = pd.DataFrame(next_round, columns=["Fe", "Co", "Ni", "Cu"])
df_next["Pt"] = 20.0

print("\n推荐的下一轮 20 组 FeCoNiCuPt 成分（原子百分数）：")
print(df_next)

save_path = save path
df_next.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n已保存至：{save_path}")

