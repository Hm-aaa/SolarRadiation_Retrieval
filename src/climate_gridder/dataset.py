import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SMACNPDataset(Dataset):
    """
        information 
    """
    def __init__(self, data_dir, context_fraction=0.5, send = 42):
        self.data_dir = data_dir
        self.context_fraction = context_fraction
        self.rng = np.random.default_rng(send)
        # 加载数据📊
        self.data = np.load(data_dir,allow_pickle=False)['X']

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        # self.data.shape = (time, coords, features)
        # 当前时刻的数据，形状为 (coords, features)
        current = self.data[idx, :, :]

        n_coords = current.shape[0]
        ratio = self.context_fraction

        # 随机选择 ratio% 的坐标索引
        n_content = int(n_coords * ratio)

        content_idx = self.rng.choice(
            n_coords,
            size=n_content,
            replace=False,
        )

        # 剩下的坐标
        target_idx = np.setdiff1d(
            np.arange(n_coords),
            content_idx,
            assume_unique=False,
        )

        # 输入内容和目标
        x_context = self.data[idx, content_idx, :-1]  # (n_content, features)
        y_context = self.data[idx, content_idx, -1]

        x_target = self.data[idx, target_idx, :-1]    # (剩余坐标数, features)
        y_target = self.data[idx, target_idx, -1]

        return {
            "x_context": torch.as_tensor(x_context, dtype=torch.float32),
            "y_context": torch.as_tensor(y_context, dtype=torch.float32).unsqueeze(-1),
            "x_target": torch.as_tensor(x_target, dtype=torch.float32),
            "y_target": torch.as_tensor(y_target, dtype=torch.float32).unsqueeze(-1)
        }

if __name__ == "__main__":
    data = SMACNPDataset(
        data_dir = '/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_04/coords_04_train.npz', 
    )
    
    # 2. 创建 DataLoader
    dataloader = DataLoader(
        data,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=True
    )

    for batch in dataloader:
        print(batch['x_context'].shape,batch['y_context'].shape)

