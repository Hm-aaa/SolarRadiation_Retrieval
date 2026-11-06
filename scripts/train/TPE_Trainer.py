
import os
import numpy as np
import optuna
from src.climate_gridder.smacnp import SMACNP_ALL
from src.climate_gridder.dataset import SMACNPDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import json
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TPETrainer:
    """
    TPETrainer类用于基于Optuna TPE(Tree-structured Parzen Estimator)算法自动优化深度学习模型的超参数。
    该类集成了模型的训练、验证、参数搜索、早停和模型最佳权重保存等功能，主要用于太阳能功率预测任务中的深度学习模型的高效调优与自动化实验管理。
    """
    def __init__(self,
        device,
        train_sets,
        val_sets,
        context_fraction,
        send,
        saved_parame_path
    ) -> None:
        """
        Args:
            cuda: GPU 设备
        """
        self.device = torch.device(device)
        self.train_sets = train_sets
        self.val_sets = val_sets
        self.context_fraction = context_fraction
        self.send = send
        self.saved_parame_path = saved_parame_path

        os.makedirs(self.saved_parame_path, exist_ok=True)


    def train_one_epoch(self,model, loss_fn,optimizer,dataloader):
        """
        单轮训练函数
        Returns:
            平均训练损失(float)
        """
        model.train()
        total_loss = 0
        for batch in dataloader:
            # 读取数据
            x_context = batch['x_context'].to(self.device,dtype=torch.float32)
            y_context = batch['y_context'].to(self.device,dtype=torch.float32)
            x_target = batch['x_target'].to(self.device,dtype=torch.float32)
            y_target = batch['y_target'].to(self.device,dtype=torch.float32)
            optimizer.zero_grad()
            # 前向传播
            outputs = model(x_context,y_context,x_target)
            # 计算损失
            loss = loss_fn(outputs, y_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_target.size(0)
        return total_loss / len(dataloader.dataset)
   
    def validate(self,model, trial, epoch,loss_fn,dataloader):
        """
        验证函数
        用于模型在验证集上的性能评估，并将结果汇报给 Optuna，支持 early stopping。
        Returns:
            val_rmse: 验证集的 RMSE 评估数值
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                # 读取数据
                x_context = batch['x_context'].to(self.device,dtype=torch.float32)
                y_context = batch['y_context'].to(self.device,dtype=torch.float32)
                x_target = batch['x_target'].to(self.device,dtype=torch.float32)
                y_target = batch['y_target'].to(self.device,dtype=torch.float32)
                # 前向传播
                outputs = model(x_context,y_context,x_target)
                # 计算损失
                loss = loss_fn(outputs, y_target)

                batch_size = y_target.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        val_mse = total_loss / total_samples
        val_rmse = np.sqrt(val_mse)
        
        trial.report(val_rmse, epoch)

        # Optional pruning: stop bad trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_rmse

    def objective(self, trial):
        """
        Optuna的目标函数(objective)

        该函数定义了每次Optuna超参数优化迭代的完整流程,包括:
        - 利用当前trial动态加载模型和超参数(如学习率、batch size)
        - 构建训练集和验证集的数据管道
        - 执行模型的训练与验证，并记录每轮的验证集性能(RMSE)
        - 进行early stopping和prune机制,避免无效trial浪费计算资源
        - 返回当前trial的最优验证集RMSE,供Optuna用于超参数搜索

        Args:
            trial (optuna.trial.Trial): Optuna框架传入的trial对象,负责管理每次超参数采样和评估打点

        Returns:
            float: 当前trial对应的最优验证集RMSE作为优化目标
        """
    
        # ====================================================
        # 1. loading model 
        # ====================================================
        model = SMACNP_ALL(
                input_dim= 10,
                spatial_dim= 2,
                attr_dim= 8,
                y_dim= 1,
                r_dim=trial.suggest_categorical("r_dim", [32, 64, 128, 256, 512]), # 属性
                w_dim=trial.suggest_categorical("w_dim", [32, 64, 128, 256, 512]), # 坐标
                v_dim=trial.suggest_categorical("v_dim", [32, 64, 128, 256, 512]), # 方差的，我取消了
                hidden_size=trial.suggest_categorical("hidden_size", [256, 512, 1024]), # MLP 的维度
                num_heads=trial.suggest_categorical("num_heads", [4, 8, 16]),
                dropout_rate=trial.suggest_float( "dropout_rate",0.1, 0.3,step=0.05),
                spatial_indices= [0,1],
                use_hybrid_decoder= False,
                use_adaptive_knn= False,
                use_positional_encoding= False
        ).to(self.device)
        # ====================================================
        # 1. Define search space
        # ====================================================
        # 这里的 log=True 表示在对数（logarithmic）尺度下搜索学习率超参数；
        # 适用于学习率这类通常分布在不同数量级上的超参数，使得搜索更加高效和均匀。
        lr = trial.suggest_float( "lr",1e-5, 5e-4,log=True)
        batch_size = 1024
        
        # ====================================================
        # 3. Initialize the model
        # ====================================================
        # 配置训练数据管道
        train_dataset = SMACNPDataset(self.train_sets,self.context_fraction, self.send)
        val_dataset = SMACNPDataset(self.val_sets, self.context_fraction, self.send)

        train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size= batch_size)
        # 加载设备
        # # 初始化模型，替换成你自己的模型

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr)
        # 定义损失函数
        loss_fn = nn.MSELoss()

        # ====================================================
        # 4. Train and Validate model
        # ====================================================
        best_val_rmse = np.inf
        patience = 20
        wait = 0

        for epoch in range(1, 1000 + 1):
            self.train_one_epoch(model, loss_fn, optimizer,train_dataloader)
            val_rmse = self.validate(model, trial, epoch,loss_fn,val_dataloader)

            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                wait = 0
                
            else:
                wait += 1

            if wait >= patience:
                break

        return best_val_rmse

    def run_tuneing(self, n_trials=50):
        """
        使用Optuna基于TPE算法对模型超参数进行优化(超参数调优）。
        该函数自动化地搜索并记录最优的超参数组合及其对应的最小RMSE结果。
        参数:
            n_trials: 要尝试的不同超参数组合次数，次数越多，越容易获得更优解，但计算量增加。
        保存:
            在输出目录下生成 best_tpe_params.json 文件,包含最佳RMSE和对应的超参数。
        """
        # 下述 Optuna 参数会对实验结果产生较大影响：
        # 1. sampler（采样器）：决定超参数搜索空间的探索方式（如TPE、Random等），影响超参数找到最优解的效率和全局性。
        # 2. pruner（剪枝器）：用于提前终止效果不佳的 trial，加快实验；但不当配置有时可能错过后期改善的参数组合。
        # 3. n_trials（实验次数）：尝试不同超参数组合的次数，次数越多更容易找到更优解，但计算量增加。
        # 4. objective/目标函数中设定的调优超参数（如 lr、batch_size、kernel_size 等）：这些直接影响模型训练过程和最终性能，是影响最大的部分，建议关注调优空间设置及范围。
        self.description('tuneing')
        sampler = optuna.samplers.TPESampler(
            seed=34,
            n_startup_trials=10
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage="sqlite:///tpe_study.db",
            study_name="smacnp_tuning",
            load_if_exists=True
        )
        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        os.makedirs(self.saved_parame_path, exist_ok=True)
        with open(f"{self.saved_parame_path}/best_tpe_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_rmse": best_value,
                    "best_params": best_params
                },
                f,
                indent=4,
                ensure_ascii=False
            )
        print('全部Tuneing完成✅')

    def description(self, trial):
        if isinstance(trial, optuna.trial.Trial):
            print(f'【超参数调优模式】调优中模型(TPE算法)，当前设备：{self.device}')
        else:
             print(f'【训练模式】正在使用模型，当前设备：{self.device}')


if __name__ == '__main__':

    import argparse

    set_seed(42)

    parser = argparse.ArgumentParser(description='TPETrainer 超参数优化脚本')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU设备')
    parser.add_argument('--train_sets', type=str, default='/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04/coords_04_train_v1.1.npz', help='训练集')
    parser.add_argument('--val_sets', type=str, default='/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04/coords_04_val_v1.1.npz', help='验证集')
    parser.add_argument('--context_fraction', type=float, default=0.5, help='上下文比列')
    parser.add_argument('--send', type=int, default=42, help='随机数')
    parser.add_argument('--saved_parame_path', type=str, default='/home/jinhm/PhD_Project/SolarRadiation_Retrieval/outputs/checkpoints', help='最优参数保存路径')
    parser.add_argument('--n_trials', type=int, default=100, help='搜算空间')
    args = parser.parse_args()

    trainer = TPETrainer(
        device=args.device,
        train_sets=args.train_sets,
        val_sets=args.val_sets,
        context_fraction=args.context_fraction,
        send=args.send,
        saved_parame_path=args.saved_parame_path
    )

    trainer.run_tuneing(n_trials=args.n_trials)