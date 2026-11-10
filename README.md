# Solar Radiation Retrieval

![](https://img.shields.io/badge/python-3.10.14-red) ![](https://img.shields.io/badge/Pytorch-cuda11.8-red)

基于多源气象、卫星与地表观测数据的短波辐射反演与预测项目。项目以 **SMACNP** 为核心，通过上下文站点观测和目标位置的气象/几何特征，对目标位置的短波辐射进行条件预测。

> **说明**
>
> 本仓库包含项目源代码、配置、示例坐标、部分日志和模型 checkpoint。
> 原始 ERA5/Himawari 数据、DEM 数据、完整站点观测以及大型 `.npz` 训练集未包含在仓库中，需要按本文档的数据准备流程自行生成或放置。

---

## 1. 项目简介

短波辐射受太阳高度角、云量、气溶胶、地形和天气过程共同影响，具有明显的时空非平稳性和高值样本稀缺问题。本项目使用条件神经过程式建模：

- 将同一时刻的部分坐标作为 **context set**；
- 将剩余坐标作为 **target set**；
- 利用 context 坐标的辐射观测和 target 坐标的气象、时间、太阳几何特征；
- 预测 target 坐标的短波辐射。

模型采用 PyTorch 实现，核心模块包括：

- Laplace 空间位置编码；
- 全局/局部多头交叉注意力；
- 目标属性条件化；
- MLP/图网络混合解码器；
- 可选的 k-NN、位置编码和站点条件化模块。

---

## 2. 主要特点

- **条件神经过程式预测**：通过 context 观测预测未观测坐标；
- **多源特征融合**：经纬度、海拔、太阳天顶角、方位角、时间周期、ERA5/Himawari 辐射；
- **全局或局部注意力**：支持 Global Attention、Localized Attention 和 Hybrid Graph Decoder；
- **多种数据版本**：ERA5、v1.0、v1.1、v2.0 配置；
- **超参数优化**：集成 Optuna TPE；
- **训练监控**：支持 TensorBoard；
- **推理脚本**：支持网格数据评估和单点/站点预测。

---

## 3. 方法概述

### 3.1 输入与输出

数据张量形状为：

```text
X: (time, coords, features)
```

当前特征约定为：

| 索引 | 特征 | 含义 |
|---:|---|---|
| 0 | `latitude_norm` | 归一化纬度 |
| 1 | `longitude_norm` | 归一化经度 |
| 2 | `elevation_norm` | 归一化海拔 |
| 3 | `zenith_cos` | 太阳天顶角余弦 |
| 4 | `azimuth_sin` | 太阳方位角正弦 |
| 5 | `azimuth_cos` | 太阳方位角余弦 |
| 6 | `minute_sin` | 日内时间正弦编码 |
| 7 | `minute_cos` | 日内时间余弦编码 |
| 8 | `day_sin` | 年内时间正弦编码 |
| 9 | `day_cos` | 年内时间余弦编码 |
| 10 | `ghi_norm` | 归一化短波辐射，作为预测目标 |
| 11 | `ghi` | 原始短波辐射，单位 W/m² |

模型输入为前 10 个特征，预测目标为 `ghi_norm`：

```text
x_context = X[:, context_idx, :10]
y_context = X[:, context_idx, 10]

x_target  = X[:, target_idx, :10]
y_target  = X[:, target_idx, 10]
```

短波辐射通常使用 `log1p + z-score` 归一化：

```python
ghi_log = np.log1p(ghi)
ghi_norm = (ghi_log - solar_mu) / solar_sigma
```

反归一化：

```python
ghi = np.expm1(ghi_norm * solar_sigma + solar_mu)
```

### 3.2 Context/Target 划分

`SMACNPDataset` 对每个时间步随机划分坐标：

```text
context_fraction = 0.5
```

表示：

- 50% 的坐标进入 context；
- 剩余 50% 的坐标进入 target。

训练时通过 context 观测预测 target 辐射。验证和测试时也可以使用固定划分，避免每次验证的 target 集合不同。

### 3.3 训练目标

当前主训练脚本在归一化辐射目标上使用均方误差：

```python
loss_fn = nn.MSELoss()
```

模型输出为单个确定性预测值。若需要概率预报，可以将输出头改为 21 个分位数并使用 Pinball Loss。

### 3.4 模型结构

```text
Context (s, x, y)                 Target (s, x)
        │                                │
        ├── Laplace Location Encoder ────┤
        │                                │
        ├── Attribute Cross-Attention ───┤
        │                                │
        └────────── Decoder ─────────────┘
                         │
                    Target GHI
```

主要源码：

```text
src/climate_gridder/
├── smacnp_d.py      # 主模型：SMACNP_ALL_D
├── smacnp_p.py      # 变体/实验版本
├── dataset.py       # SMACNPDataset
└── utils.py         # 空间属性拆分、随机种子等工具
```

---

## 4. 目录结构

```text
SolarRadiation_Retrieval/
├── configs/                    # 训练配置
│   ├── config_ERA5.yaml
│   ├── config_ERA5_v1.0.yaml
│   ├── config_ERA5_v1.1.yaml
│   └── config_ERA5_v2.0.yaml
├── data/
│   ├── china_shp/             # 中国边界矢量数据
│   ├── coords/                # 示例坐标 CSV
│   └── system_id_information.csv
├── logs/
│   └── config/                # TensorBoard 日志和训练日志
├── outputs/
│   ├── checkpoints/           # 模型权重
│   └── predictions/           # 推理结果
├── scripts/
│   ├── data_process/          # 数据下载、特征构建、NPZ 生成
│   ├── infer/                 # 推理脚本
│   └── train/                 # 常规训练与 TPE 调参
├── src/climate_gridder/       # 模型、数据集与工具
├── run.sh                     # 常规训练启动脚本
├── tpe_train.sh               # TPE 调参启动脚本
└── README.md
```

---

## 5. 数据说明

### 5.1 仓库中包含的数据

- 示例坐标文件：`data/coords/coords_01.csv` 至 `coords_07.csv`；
- 中国边界 shapefile：`data/china_shp/`；
- 环测仪站点信息：`data/system_id_information.csv`；
- 部分训练日志和 TensorBoard events；
- 四个模型 checkpoint：
  - `smacnp_best_ERA5.pt`
  - `smacnp_best_v1.0.pt`
  - `smacnp_best_v1.1.pt`
  - `smacnp_best_v2.0.pt`

### 5.2 仓库中未包含的数据

以下文件体积较大，未提交到 GitHub：

- ERA5 原始 NetCDF/GRIB 文件；
- Himawari-8/9 原始影像或辐射 CSV；
- MirrorEarth 下载数据；
- 数字高程模型 `data/DEM/etopo1.nc`；
- 完整环测仪观测数据；
- 训练/验证/测试 `.npz` 数据集；
- 数据库导出文件。

### 5.3 期望的 NPZ 格式

训练脚本期望读取：

```text
X: (time, coords, 12) float32
feature_names: (12,)
elevation_mu: float32
elevation_sigma: float32
solar_mu: float32
solar_sigma: float32
```

示例路径布局：

```text
/data/solar/era5_solar/MirrorEarth/
└── coords_04/
    ├── coords_04_train_v1.1.npz
    └── coords_04_val_v1.1.npz
```

运行前需要修改 `configs/*.yaml` 中的路径，使其指向本地数据。

---

## 6. 环境安装

建议使用 Python 3.10+，并根据 CUDA 版本安装 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
```

安装基础依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy pyyaml tensorboard optuna
pip install xarray netCDF4 pvlib geopandas matplotlib openpyxl
pip install requests cdsapi psycopg2-binary
```

模型依赖 PyTorch Geometric：

```bash
pip install torch-geometric
```

`torch-cluster` 需要与 PyTorch/CUDA 版本匹配，请根据 PyG 官方安装页面选择对应 wheel：

```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

`torch-cluster` 通常还需要 `torch-scatter`。如果安装失败，请先确认 PyTorch 和 CUDA 版本，再使用对应的 PyG wheel。

---

## 7. 快速开始

### 7.1 设置项目路径

```bash
export PROJECT_PATH=/path/to/SolarRadiation_Retrieval
export PYTHONPATH=$PWD
```

`run.sh` 和 `tpe_train.sh` 中存在服务器绝对路径，运行前请改成你的本地路径。

### 7.2 数据准备流程

数据准备脚本位于：

```text
scripts/data_process/
```

典型流程：

```text
1. 生成坐标样本
   generate_coordinate_samples.py

2. 下载或准备 ERA5/Himawari 数据
   download_era5_ghi_ecmwf.py
   download_era5_ghi_mirrorearth.py
   download_himawari_ghi_jaxa.py
   download_himawari_ghi_openmeteo.py

3. 计算太阳几何和时间特征
   build_solar_features.py

4. 构建训练/验证 NPZ
   build_train_val_sets.py
```

其中 `build_train_val_sets.py` 中包含时间范围、`project`、`label` 等硬编码参数，使用前需要按任务修改。

### 7.3 开始训练

```bash
python scripts/train/Pre-Trained.py \
  configs/config_ERA5_v1.0.yaml
```

训练配置包含：

```yaml
model:
  lr: 1.0e-05
  batch_size: 256
  epochs: 1000
  patience: 50
  train_sets: /path/to/train.npz
  val_sets: /path/to/val.npz
  checkpoints: outputs/checkpoints/smacnp_best_v1.0.pt
```

### 7.4 TensorBoard

```bash
tensorboard --logdir logs/config --bind_all --port 6060
```

然后访问：

```text
http://<服务器地址>:6060
```

### 7.5 TPE 超参数搜索

```bash
python scripts/train/TPE_Trainer.py \
  --device cuda:0 \
  --train_sets /path/to/train.npz \
  --val_sets /path/to/val.npz \
  --context_fraction 0.5 \
  --send 42 \
  --saved_parame_path outputs/checkpoints \
  --n_trials 30
```

> 当前脚本保留了参数名 `send`，实际含义是随机种子。建议后续改名为 `seed`。

### 7.6 推理

网格/测试集推理：

```bash
PYTHONPATH=$PWD python scripts/infer/infer_test.py
```

单点/站点预测：

```bash
PYTHONPATH=$PWD python scripts/infer/infer_point.py
```

推理脚本包含原始服务器绝对路径和固定配置路径，使用前需要修改：

```text
CONFIG_FILE
test_file
checkpoints
data/DEM/etopo1.nc
```

---

## 8. 配置与版本

| 配置文件 | 说明 |
|---|---|
| `config_ERA5.yaml` | ERA5 基线配置 |
| `config_ERA5_v1.0.yaml` | v1.0 数据版本 |
| `config_ERA5_v1.1.yaml` | v1.1 数据版本 |
| `config_ERA5_v2.0.yaml` | v2.0 数据版本 |

不同版本主要对应不同的数据预处理、夜间样本筛选和归一化策略。训练前应确认配置文件中的：

```text
train_sets
val_sets
checkpoints
log_dir
device
```

均指向当前环境。

---

## 9. 输出

训练和推理结果保存在：

```text
outputs/
├── checkpoints/
│   ├── smacnp_best_ERA5.pt
│   ├── smacnp_best_v1.0.pt
│   ├── smacnp_best_v1.1.pt
│   └── smacnp_best_v2.0.pt
└── predictions/
    ├── test.csv
    └── <biz_id>.csv
```

预测 CSV 通常包含：

```text
time
pred / pre
y_target
residual
```

使用时注意：`pred`/`pre` 应该已经在原始 W/m² 空间完成反归一化。

---

## 10. 联系方式

如有问题，请在 GitHub Issues 中提交，或联系项目维护者(jinhm@mail2.sysu.edu.cn)。
