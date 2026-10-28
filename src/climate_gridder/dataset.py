import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
import pandas as pd
class SMACNPDataset(Dataset):
    """
        information 
    """
    def __init__(self, data_dir, split='train', context_fraction=0.5):
        self.data_dir = data_dir
        self.split = split
        self.context_fraction = context_fraction

        # 训练 2015-2024
        # 验证 2025
        # 根据训练，验证，测试读取数据。
        if split == 'train':
            start = "2015-01-01 00:00:00"
            end = "2024-12-31 23:59:59"
        else:
            start = "2025-01-01 00:00:00"
            end = "2025-12-31 23:59:59"
        
        data_List = []
        for coords_file_item in glob.glob(f'{data_dir}/*'):
            data = self._load_data_by_time_range(
                csv_file=coords_file_item,
                start_time=start,
                end_time=end
            )
            # 先删除 TIME_UTC，避免变成 object 数组
            data = data.drop(columns=["TIME_UTC"], errors="ignore")
            data_List.append(data.to_numpy(dtype=np.float32))

        # 合并
        data_List = np.stack(data_List, axis=0)
        print("data_List shape:", data_List.shape)
        print("data_List dtype:", data_List.dtype)

    def __len__(self):
        return len(self.day_dirs) 

    def __getitem__(self, idx):
        pass

if __name__ == "__main__":
    data = SMACNPDataset(
        data_dir = '/home/jinhm/data/solar/era5_solar/MirrorEarth/coords_03', 
        split='train',
        context_fraction=0.5
    )
    print(data)