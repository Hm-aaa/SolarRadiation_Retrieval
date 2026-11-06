import numpy as np
import matplotlib
matplotlib.use("Agg")  # 服务器无图形界面时使用
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. 读取数据
# ============================================================
path = "/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04/coords_04_train_v2.0.npz"

with np.load(path, allow_pickle=False) as data:
    x = data["X"][:, 1200, -2]

# 转为 float32 一维数组，并去除 NaN / Inf
x = np.asarray(x, dtype=np.float32).reshape(-1)
x = x[np.isfinite(x)]

print("样本数:", x.size)
print("最小值:", float(np.min(x)))
print("最大值:", float(np.max(x)))
print("均值:", float(np.mean(x)))
print("标准差:", float(np.std(x)))
print("中位数:", float(np.median(x)))
print("95% 分位数:", float(np.quantile(x, 0.95)))
print("99% 分位数:", float(np.quantile(x, 0.99)))
print("零值比例:", float(np.mean(x == 0.0)))

# ============================================================
# 2. 绘制分布
# ============================================================
nonzero = x[x > 0.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 全部样本
axes[0].hist(
    x,
    bins=200,
    density=True,
    color="steelblue",
    edgecolor="none",
    alpha=0.8,
)
axes[0].set_title("All normalized radiation samples")
axes[0].set_xlabel("Normalized GHI")
axes[0].set_ylabel("Density")
axes[0].set_yscale("log")
axes[0].grid(alpha=0.25)

# 非零样本，也就是白天部分
if nonzero.size > 0:
    axes[1].hist(
        nonzero,
        bins=200,
        density=True,
        color="darkorange",
        edgecolor="none",
        alpha=0.8,
    )
    axes[1].set_title("Daytime non-zero normalized radiation")
    axes[1].set_xlabel("Normalized GHI")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.25)
else:
    axes[1].text(
        0.5,
        0.5,
        "No non-zero samples",
        ha="center",
        va="center",
        transform=axes[1].transAxes,
    )

plt.tight_layout()

out_path = Path(path).parent / "coords_04_ghi_norm_distribution.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print("图像已保存:", out_path)