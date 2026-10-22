import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn



class SMACNPDataset_ALL(Dataset):
    """
    A PyTorch Dataset for loading daily SMACNP data.
    This version is designed to handle multi-target (e.g., temp and precip) data
    prepared by the `prepare_data.py` script.
    """
    def __init__(self, data_dir, split='train', context_fraction=0.5, inference=False, use_fixed_seed=False, on_grid_training=True, model_type="CONVCNP"):
        self.data_dir = data_dir
        self.split = split
        self.context_fraction = context_fraction if not inference else 1.0
        self.inference = inference
        self.use_fixed_seed = use_fixed_seed
        self.on_grid_training = on_grid_training
        self.model_type = model_type # New parameter

        self.day_dirs = sorted(glob.glob(os.path.join(data_dir, "day_*")))
        if not self.day_dirs:
            raise FileNotFoundError(f"No 'day_*' directories found in {data_dir}")

        total_days = len(self.day_dirs)
        if total_days > 1:
            train_days = int(0.8 * total_days)
            train_days = min(train_days, total_days - 1)
        else:
            train_days = total_days
            print("Warning: Only 1 day found. Using it for both training and validation.")
        
        print(f"Total days: {total_days}, Training days: {train_days}, Validation days: {total_days - train_days}")

        if split == 'train':
            self.day_dirs = self.day_dirs[:train_days]
        elif split == 'val':
            self.day_dirs = self.day_dirs[train_days:]
        elif split != 'all':
            raise ValueError("split must be 'train', 'val', or 'all'")

        # --- Load static grid if it exists (for ConvCNP and GriddedTNP) ---
        self.static_grid = None
        if self.model_type in ["CONVCNP", "GRIDDEDTNP"]:
            static_grid_path = os.path.join(data_dir, "static_grid_predictors.npy")
            if os.path.exists(static_grid_path):
                try:
                    self.static_grid = torch.from_numpy(np.load(static_grid_path)).float()
                    print(f"Successfully loaded static grid with shape: {self.static_grid.shape}")
                except Exception as e:
                    print(f"Warning: Could not load or process static_grid_predictors.npy. Error: {e}")
        
        # --- Load edge_index if model_type is GCNP ---
        self.edge_index = None
        if self.model_type == "GCNP":
            edge_index_path = os.path.join(data_dir, "graph_edge_index.npy")
            if os.path.exists(edge_index_path):
                try:
                    self.edge_index = torch.from_numpy(np.load(edge_index_path)).long()
                    print(f"Successfully loaded graph edge_index with shape: {self.edge_index.shape}")
                except Exception as e:
                    print(f"Warning: Could not load or process graph_edge_index.npy. Error: {e}")
            else:
                raise FileNotFoundError(f"graph_edge_index.npy not found in {data_dir} for GCNP model.")


    def __len__(self):
        return len(self.day_dirs)

    def __getitem__(self, idx):
        day_dir = self.day_dirs[idx]
        try:
            all_station_attributes = np.load(os.path.join(day_dir, "context_attributes_X.npy"))
            all_station_targets = np.load(os.path.join(day_dir, "context_values_y_normalized.npy"))
        except FileNotFoundError:
            return_dict = {'x_context_all': torch.empty(0), 'y_context': torch.empty(0),
                    'x_target_all': torch.empty(0), 'y_target': torch.empty(0)}
            # Ensure static_grid or edge_index is still passed if available, even if data is empty
            if self.static_grid is not None:
                return_dict['static_grid'] = self.static_grid
            if self.edge_index is not None:
                return_dict['edge_index'] = self.edge_index
            return return_dict

        num_stations = all_station_attributes.shape[0]
        if num_stations <= 1:
            return_dict = {'x_context_all': torch.empty(0), 'y_context': torch.empty(0),
                    'x_target_all': torch.empty(0), 'y_target': torch.empty(0)}
            # Ensure static_grid or edge_index is still passed if available, even if data is empty
            if self.static_grid is not None:
                return_dict['static_grid'] = self.static_grid
            if self.edge_index is not None:
                return_dict['edge_index'] = self.edge_index
            return return_dict

        if not self.inference:
            if self.use_fixed_seed:
                rng = np.random.default_rng(seed=idx)
            else:
                rng = np.random

            max_possible_context = num_stations - 1
            num_context = max(1, min(max_possible_context, int(self.context_fraction * num_stations)))
            
            all_indices = np.arange(num_stations)
            context_indices = rng.choice(all_indices, num_context, replace=False)

            if self.on_grid_training:
                # "On-the-grid" training: target set includes all stations,
                # forcing the model to learn to reconstruct context points.
                target_indices = all_indices
            else:
                # "Off-the-grid" training: target set is only the unseen stations,
                # a stricter test of generalization.
                context_mask = np.zeros(num_stations, dtype=bool)
                context_mask[context_indices] = True
                target_indices = all_indices[~context_mask]


            x_context_all = torch.tensor(all_station_attributes[context_indices], dtype=torch.float32)
            y_context = torch.tensor(all_station_targets[context_indices], dtype=torch.float32)
            
            x_target_all = torch.tensor(all_station_attributes[target_indices], dtype=torch.float32)
            y_target = torch.tensor(all_station_targets[target_indices], dtype=torch.float32)
        
        else: # Inference mode remains the same
            x_context_all = torch.tensor(all_station_attributes, dtype=torch.float32)
            y_context = torch.tensor(all_station_targets, dtype=torch.float32)
            try:
                target_grid_attributes = np.load(os.path.join(day_dir, "target_attributes_X_star.npy"))
                x_target_all = torch.tensor(target_grid_attributes, dtype=torch.float32)
                y_target = None 
            except FileNotFoundError:
                x_target_all = x_context_all
                y_target = y_context

        # x_context_all 【B, M, D】
        # y_context 【B, M, 1】
        # x_target_all 【B, N, D】
        # y_target 【B, N, 1】
        return_dict = {'x_context_all': x_context_all, 'y_context': y_context,
                'x_target_all': x_target_all, 'y_target': y_target}
        
        if self.static_grid is not None:
            return_dict['static_grid'] = self.static_grid
        
        if self.edge_index is not None:
            return_dict['edge_index'] = self.edge_index
            
        return return_dict

