import time
from pathlib import Path
import torch
import numpy as np
from src.climate_gridder.smacnp import SMACNP_ALL
from src.climate_gridder.dataset import SMACNPDataset
import yaml 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import torch.nn as nn
from datetime import datetime

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    单轮训练函数
    Args:
        model: 需要训练的 PyTorch 模型
        dataloader: 训练数据 DataLoader
        loss_fn: 损失函数（如 nn.MSELoss)
        optimizer: 优化器（如 Adam、SGD 等）
        device: 计算设备（'cuda' 或 'cpu')
    Returns:
        平均训练损失(float)
    """
    model.train()
    total_loss = 0
    for batch in dataloader:
        # 读取数据
        x_context = batch['x_context'].to(device,dtype=torch.float32)
        y_context = batch['y_context'].to(device,dtype=torch.float32)
        x_target = batch['x_target'].to(device,dtype=torch.float32)
        y_target = batch['y_target'].to(device,dtype=torch.float32)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(x_context,y_context,x_target)
        # 计算损失
        loss = loss_fn(outputs, y_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_target.size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, loss_fn, device):
        """
        验证函数
        用于模型在验证集上的性能评估，并将结果汇报给 Optuna，支持 early stopping。
        Args:
            model: 当前训练的 PyTorch 模型
            trial: Optuna 的 trial 对象，用于超参数优化和中间结果打点
            epoch: 当前 epoch 轮次
            dataloader: 验证数据的 DataLoader
            loss_fn: 损失函数（如 nn.MSELoss)
            device: 运行设备（'cuda' 或 'cpu')
        Returns:
            val_rmse: 验证集的 RMSE 评估数值
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                # 读取数据
                x_context = batch['x_context'].to(device,dtype=torch.float32)
                y_context = batch['y_context'].to(device,dtype=torch.float32)
                x_target = batch['x_target'].to(device,dtype=torch.float32)
                y_target = batch['y_target'].to(device,dtype=torch.float32)
                # 前向传播
                outputs = model(x_context,y_context,x_target)
                # 计算损失
                loss = loss_fn(outputs, y_target)

                batch_size = y_target.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        val_mse = total_loss / total_samples
        val_rmse = np.sqrt(val_mse)
    
        return val_rmse


if __name__ == '__main__':

    # ====================================================
    # 1. load config.yaml 
    # ====================================================
    CONFIG_FILE = Path(os.getenv("PROJECT_PATH")) / 'configs' / 'config.yaml'
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    device = torch.device(model_cfg['device'])

    # 每次训练创建独立日志目录
    writer = SummaryWriter(log_dir=model_cfg['log_dir'])

    # ====================================================
    # 2. loading model 
    # ====================================================
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
    ).to(device)

    # ====================================================
    # 3. Define search space
    # ====================================================
    lr = model_cfg['lr']
    batch_size = model_cfg['batch_size']
    
    # ====================================================
    # 4. Initialize the model
    # ====================================================
    # 配置训练数据管道
    train_dataset = SMACNPDataset(model_cfg['train_sets'], model_cfg['context_fraction'], model_cfg['send'])
    val_dataset = SMACNPDataset(model_cfg['val_sets'], model_cfg['context_fraction'], model_cfg['send'])

    train_dataloader = DataLoader(train_dataset, batch_size= batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size= batch_size)
    # 加载设备
    # # 初始化模型，替换成你自己的模型

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr)
    # 定义损失函数
    loss_fn = nn.MSELoss()

    # ====================================================
    # 5. Train and Validate model
    # ====================================================
    best_val_rmse = np.inf
    patience = model_cfg["patience"]
    wait = 0

    for epoch in range(1, model_cfg['epochs'] + 1):

        epoch_start = time.perf_counter()
        train_rmse = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_rmse = validate(model,val_dataloader, loss_fn, device)
        epoch_time = time.perf_counter() - epoch_start

        # 打印训练信息💻
        print(
            f"Epoch [{epoch:03d}/{model_cfg['epochs'] + 1:03d}] | "
            f"Train RMSE: {train_rmse:.6f} | "
            f"Val RMSE: {val_rmse:.6f} | "
            f"Time: {epoch_time:.2f}s",
            flush=True,
        )

        # 记录每一轮指标
        writer.add_scalar("RMSE/Train", train_rmse, epoch)
        writer.add_scalar("RMSE/Validation", val_rmse, epoch)
        # 立即写入磁盘
        writer.flush()

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            wait = 0

            # 保存最优模型
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg
                },
                model_cfg["checkpoints"],
            )
        else:
            wait += 1

        if wait >= patience:
            break

