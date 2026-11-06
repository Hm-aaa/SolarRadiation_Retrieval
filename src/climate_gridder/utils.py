import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def set_seed(seed: int):
    """
    Sets the seed for reproducibility for random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The following two lines are required for full reproducibility on CUDA
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            print("Note: CUDNN settings not applicable/available.")
    print(f"Reproducibility seed set to {seed}")



# --------------------------- 
# Helper Function to Separate Spatial Coords
# --------------------------- 
def separate_spatial_attributes(x_all, spatial_indices):
    """
    Separates spatial coordinates (s) and other attributes (x) from a combined tensor.
    This function is robust to `spatial_indices` being either a list or a tensor.
    The input dimension is inferred directly from the input tensor `x_all`.
    """
    device = x_all.device
    input_dim = x_all.shape[-1]

    # Handle spatial_indices tensor creation robustly to avoid UserWarning
    if not torch.is_tensor(spatial_indices):
        s_indices_tensor = torch.tensor(spatial_indices, device=device, dtype=torch.long)
    else:
        s_indices_tensor = spatial_indices.to(device)

    # Create attribute indices tensor efficiently using a boolean mask
    is_spatial_mask = torch.zeros(input_dim, dtype=torch.bool, device=device)
    is_spatial_mask[s_indices_tensor] = True
    attr_indices_tensor = torch.arange(input_dim, device=device)[~is_spatial_mask]

    s = torch.index_select(x_all, dim=-1, index=s_indices_tensor)
    x_attr = torch.index_select(x_all, dim=-1, index=attr_indices_tensor)
    return s.contiguous(), x_attr.contiguous()

# --------------------------- 
# Loss and Training Functions (With Error Handling)
# --------------------------- 

def Solor_loss_function(predictions, y_target):
    """
    Calculates Gaussian NLL loss for temperature only.

    Args:
        predictions (dict):
            Model outputs containing:
                predictions["temp_mu"]
                predictions["temp_variance"]

        y_target (torch.Tensor):
            Ground truth tensor.

            If shape is (B, M, 2):
                Column 0 = Temperature
                Column 1 = Precipitation

    Returns:
        loss_temp: Temperature Gaussian NLL loss
    """

    # ---------------------------
    # 1. Extract temperature target
    # ---------------------------
    y_temp = y_target[..., 0].unsqueeze(-1)

    # y_temp shape:
    # [B, M, 1]

    # ---------------------------
    # 2. Ensure variance > 0
    # ---------------------------
    eps_var = 1e-4

    temp_variance = predictions["temp_variance"].clamp(
        min=eps_var
    )

    # ---------------------------
    # 3. Gaussian NLL Loss
    # ---------------------------
    gaussian_nll = nn.GaussianNLLLoss(
        reduction="mean",
        eps=eps_var
    )

    loss_temp = gaussian_nll(
        predictions["temp_mu"],
        y_temp,
        temp_variance
    )

    return loss_temp