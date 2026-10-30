from pathlib import Path
import torch
from src.climate_gridder.smacnp import SMACNP_ALL
from src.climate_gridder.dataset import SMACNPDataset
import yaml 
from torch.utils.data import DataLoader
import os
import pandas as pd
import numpy as np

def regression_metrics(pred, target, eps=1e-12):
    """计算 RMSE、MAE 和 Pearson correlation。"""
    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    error = pred - target

    rmse = torch.sqrt(torch.mean(error ** 2)).item()
    mae = torch.mean(torch.abs(error)).item()

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()

    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2)
        * torch.sum(target_centered ** 2)
    )

    if denominator.item() < eps:
        r = float("nan")
    else:
        r = (
            torch.sum(pred_centered * target_centered)
            / denominator
        ).item()

    return rmse, mae, r

CONFIG_FILE = Path('/home/jinhm/PhD_Project/SolarRadiation_Retrieval/configs/config.yaml') 
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
device = torch.device(model_cfg['device'])
device = torch.device('cuda:0')

test_file = '/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04_2026/coords_04_test.npz'
train_dataset = SMACNPDataset(test_file, 0.2, 42)
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = SMACNP_ALL(
        input_dim=model_cfg['input_dim'],
        spatial_dim=model_cfg['spatial_dim'],
        attr_dim=model_cfg['attr_dim'],
        y_dim=model_cfg['y_dim'],
        r_dim=model_cfg['r_dim'], # 属性
        w_dim=model_cfg['w_dim'], # 坐标
        v_dim=model_cfg['v_dim'], # 方差的，我取消了
        hidden_size=model_cfg['hidden_size'], # MLP 的维度
        num_heads=model_cfg['num_heads'],
        dropout_rate=model_cfg['dropout_rate'],
        spatial_indices=model_cfg['spatial_indices'],
        use_hybrid_decoder=model_cfg['use_hybrid_decoder'],
        use_adaptive_knn=model_cfg['use_adaptive_knn'],
        use_positional_encoding=model_cfg['use_positional_encoding']
)
checkpoint = torch.load(
    model_cfg['checkpoints'],
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model = model.to(device)
model.eval()

data = []

loaded = np.load(f"/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04_2026/coords_04_test.npz",allow_pickle=False)
mu = loaded["solar_mu"]
sigma = loaded["solar_sigma"]

for batch in train_loader:
    x_context = batch['x_context'].to(device,dtype=torch.float32)
    y_context = batch['y_context'].to(device,dtype=torch.float32)
    x_target = batch['x_target'].to(device,dtype=torch.float32)
    # y_target = batch['y_target'].to(device,dtype=torch.float32)

    with torch.no_grad():
        pre = model(x_context, y_context, x_target)
    data.append(pre.detach().float().cpu().numpy())

# 按 batch 维度合并
pred_all = np.concatenate(data, axis=0)

# 保存 CSV
output_dir = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)

# 只取第一个坐标
coord_index = 100
# 反归一化

pre = np.expm1(pred_all[:, coord_index,0] * sigma + mu)
y_target = loaded['X'][:,coord_index,-1]

print(pre.shape,y_target.shape)

result_df = pd.DataFrame({
    "pred": pre,
    "y_target": y_target,
    "residual": pre-y_target,
})

result_df.to_csv(
    output_dir / 'test.csv',
    index=False,
    encoding="utf-8-sig",
    float_format="%.8f",
)

