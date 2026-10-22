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
# ---------------------------
# Temperature Loss Function
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

# --------------------------- 
# Early Stopping Class
# --------------------------- 
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.val_loss_history = []

    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        self.val_loss_history.append(val_loss)
        score = -val_loss # We want to maximize the negative loss (minimize the loss)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        checkpoint = {
            'epoch': epoch + 1, # Save next epoch number
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': val_loss,
        }
        torch.save(checkpoint, self.path)
        self.val_loss_min = val_loss


def plot_hexbin_scatter(
    ax,
    actual,
    predicted,
    mae,
    rmse,
    corr,
    variable_name,
    unit,
    gridsize=50,
    vmin=None,
    vmax=None,
    use_log_scale=False,
    show_ylabel=True,
):
    """Creates a single 2D hexbin density plot on a given Matplotlib axis."""
    if use_log_scale:
        vmin = 1 if vmin is None else vmin
        norm = LogNorm(vmin=max(vmin, 1), vmax=vmax) if vmax is not None else LogNorm(vmin=max(vmin, 1))
        h = ax.hexbin(
            actual, predicted, gridsize=gridsize, cmap="viridis", mincnt=1, norm=norm
        )
    else:
        h = ax.hexbin(
            actual,
            predicted,
            gridsize=gridsize,
            cmap="viridis",
            mincnt=1,
            vmin=vmin,
            vmax=vmax,
        )

    min_val = min(np.nanmin(actual), np.nanmin(predicted))
    max_val = max(np.nanmax(actual), np.nanmax(predicted))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="1:1 Line",
    )

    ax.set_xlabel(f"Actual {variable_name} [{unit}]", fontsize=16)
    if show_ylabel:
        ax.set_ylabel(f"Predicted {variable_name} [{unit}]", fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    stats_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nCorr: {corr:.3f}"
    props = dict(
        boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="none"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=props,
    )
    return h

def make_levels_for_uncertainty(arrays, num: int = 8):
    """
    Build evenly spaced contour levels from a collection of arrays/DataArrays.

    Ensures a positive, finite max so plots remain readable.
    """
    max_vals = []
    for arr in arrays:
        if arr is None:
            continue
        try:
            max_vals.append(float(np.nanmax(arr.values)))
        except Exception:
            continue

    if not max_vals:
        return None

    max_val = np.nanmax(max_vals)
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0

    return np.linspace(0, max_val, num)


def compute_temperature_uncertainty(ds, method: str = "rms"):
    """
    Compute time-aggregated temperature uncertainty from `temperature_std`.

    Args:
        ds: xarray Dataset containing `temperature_std`.
        method: `rms` (root-mean-square) or `mean` (arithmetic mean).

    Returns:
        DataArray of aggregated uncertainty in °C, or None if unavailable.
    """
    if "temperature_std" not in getattr(ds, "data_vars", {}):
        return None

    std_data = ds["temperature_std"]
    if method == "rms":
        return np.sqrt((std_data**2).mean(dim="time"))
    if method == "mean":
        return std_data.mean(dim="time")
    raise ValueError(f"Unknown method: {method}. Use 'rms' or 'mean'.")


def compute_precipitation_uncertainty(ds, method: str = "rms", days_per_year: float = 365.25):
    """
    Compute time-aggregated precipitation uncertainty from `precipitation_std`,
    scaled to annual units using √days_per_year.

    Args:
        ds: xarray Dataset containing `precipitation_std`.
        method: `rms` (root-mean-square) or `mean` (arithmetic mean).
        days_per_year: Scaling factor for annualization (default: 365.25).

    Returns:
        DataArray of annualized uncertainty (mm/year), or None if unavailable.
    """
    if "precipitation_std" not in getattr(ds, "data_vars", {}):
        return None

    std_data = ds["precipitation_std"]
    if method == "rms":
        daily_aggregated_std = np.sqrt((std_data**2).mean(dim="time"))
    elif method == "mean":
        daily_aggregated_std = std_data.mean(dim="time")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rms' or 'mean'.")

    return daily_aggregated_std * np.sqrt(days_per_year)

def _get_pe_suffix_parts(cfg):
    """Helper to get PE suffix parts from a model's config."""
    if cfg.get('use_positional_encoding', False):
        pe_cfg = cfg.get('positional_encoding', {})
        pe_type = pe_cfg.get('pe_type', 'sinusoidal')
        pe_dim = pe_cfg.get('pe_dim', 96)
        # For Gaussian PE, honor `num_centers` as the effective dimension.
        if pe_type == 'gaussian':
            pe_dim = pe_cfg.get('num_centers', pe_dim)
        return [f"pe-{pe_type[:3]}{pe_dim}"] # e.g., pe-sin96
    return []

def get_convcnp_filename_suffix_parts(convcnp_cfg, training_cfg=None):
    """
    Generates a list of descriptive filename suffix parts for the ConvCNP model.
    """
    parts = []

    # Grid size, e.g., "gs224x320"
    grid_size = convcnp_cfg.get('grid_size', [])
    if grid_size and len(grid_size) == 2:
        parts.append(f"gs{grid_size[0]}x{grid_size[1]}")

    # U-Net channel structure, e.g., "uc64-64-128"
    unet_channels = convcnp_cfg.get('unet_channels', [])
    if unet_channels:
        parts.append(f"uc{'-'.join(map(str, unet_channels))}")

    # RBF kernel, e.g., "rbf-gauss"
    rbf_kernel = convcnp_cfg.get('rbf_kernel', 'gaussian')
    parts.append(f"rbf-{rbf_kernel[:5]}") # Abbreviate

    # Encoder sigma, e.g., "senc0.01-L"
    sigma_enc_init = convcnp_cfg.get('sigma_enc_init', 0.01)
    learn_sigma_enc = convcnp_cfg.get('learnable_sigma_enc', True)
    parts.append(f"senc{sigma_enc_init}{'-L' if learn_sigma_enc else ''}")

    # Decoder type and its specific parameters
    decoder_type = convcnp_cfg.get('decoder', 'setconv')
    parts.append(f"dec-{decoder_type}")

    decoder_kwargs = convcnp_cfg.get('decoder_kwargs', {})
    if decoder_type == 'setconv':
        # Decoder sigma, e.g., "sdec0.05-L"
        sigma_dec_init = decoder_kwargs.get('sigma_init', 0.05)
        learn_sigma_dec = decoder_kwargs.get('learnable_sigma', True)
        parts.append(f"sdec{sigma_dec_init}{'-L' if learn_sigma_dec else ''}")
    elif decoder_type == 'attention':
        # Attention parameters, e.g., "att-h8-hd256-do0.01-ln"
        heads = decoder_kwargs.get('num_heads', 8)
        hidden_dim = decoder_kwargs.get('hidden_dim', 256)
        dropout = decoder_kwargs.get('dropout', 0.0)
        use_ln = decoder_kwargs.get('use_layer_norm', False)
        
        part = f"att-h{heads}-hd{hidden_dim}"
        if dropout > 0:
            part += f"-do{dropout}"
        if use_ln:
            part += "-ln"
        parts.append(part)
    elif decoder_type == 'multiscale_setconv':
        # Multi-scale SetConv parameters, e.g., "mss-ns3-s0.01-0.1-L"
        num_scales = decoder_kwargs.get('num_scales', 3)
        sigma_init_range = decoder_kwargs.get('sigma_init_range', (0.01, 0.1))
        learn_sigma = decoder_kwargs.get('learnable_sigma', True)
        parts.append(f"mss-ns{num_scales}-s{sigma_init_range[0]}-{sigma_init_range[1]}{'-L' if learn_sigma else ''}")

    # U-Net dropout, e.g., "do0.1"
    dropout = convcnp_cfg.get('unet_dropout_rate', 0.1)
    if dropout > 0:
        parts.append(f"do{dropout}")

    # U-Net bilinear flag, e.g., "bilin"
    if convcnp_cfg.get('unet_bilinear', True):
        parts.append("bilin")

    # Add stability params to suffix
    eps = convcnp_cfg.get('eps', 1e-4)
    parts.append(f"eps{eps}")
    vf = convcnp_cfg.get('variance_floor', 1e-3)
    parts.append(f"vf{vf}")

    if training_cfg and 'batch_size' in training_cfg:
        parts.append(f"bsz{training_cfg['batch_size']}")

    return parts

def get_smacnp_filename_suffix_parts(cfg, training_cfg=None):
    """Generate filename suffix parts for SMACNP model based on config."""
    parts = []
    
    encoder_cfg = cfg.get('encoder', {})
    configuration = cfg.get('configuration')
    encoder_type = (configuration or encoder_cfg.get('type', 'global')).lower()

    def _get_graph_cfg():
        return cfg.get('graph_cnp') or encoder_cfg.get('gcn_params', {})

    def _get_localized_cfg():
        return cfg.get('localized_cnp') or encoder_cfg.get('localized_params', {})

    def _get_gat_cfg():
        return cfg.get('gat_cnp') or encoder_cfg.get('gat_params', {})
    
    # Add encoder-specific details for clarity
    if encoder_type == 'gat':
        gat_params = _get_gat_cfg()
        k = gat_params.get('k', 20)
        layers = gat_params.get('num_layers', gat_params.get('num_gat_layers', 3))
        parts.append(f"gat{layers}l-k{k}")
        if gat_params.get('use_edge_features', gat_params.get('use_gat_edge_features', False)):
            parts.append('ef')
        if gat_params.get('use_degree_encoding', False):
            deg_dim = gat_params.get('degree_encoding_dim', 16)
            parts.append(f'de{deg_dim}')
    elif encoder_type in ('gcn', 'graph'):
        graph_params = _get_graph_cfg()
        k = graph_params.get('k', graph_params.get('k_gcn', 15))
        layers = graph_params.get('num_layers', graph_params.get('num_gcn_layers', 2))
        label = 'graph' if encoder_type == 'graph' else 'gcn'
        parts.append(f"{label}{layers}l-k{k}")
    elif encoder_type == 'localized':
        localized_params = _get_localized_cfg()
        k = localized_params.get('k', 32)
        parts.append(f"localized-k{k}")
    else:
        parts.append(encoder_type)
    
    # New features
    if cfg.get('use_layer_norm', False):
        parts.append('ln')
    
    # Resolve the positional encoding config from the active encoder to keep filenames aligned
    # with the configuration actually used in training.
    if encoder_type in ('gcn', 'graph'):
        pe_source_cfg = _get_graph_cfg()
    elif encoder_type == 'localized':
        pe_source_cfg = _get_localized_cfg()
    elif encoder_type == 'gat':
        pe_source_cfg = _get_gat_cfg()
    else:
        # Fallback to global/top-level PE settings
        pe_source_cfg = cfg.get('positional_encoding', {}) or cfg

    use_pe = pe_source_cfg.get('use_positional_encoding', cfg.get('use_positional_encoding', False))
    if use_pe:
        pe_type = pe_source_cfg.get('pe_type', pe_source_cfg.get('type', 'sinusoidal'))
        pe_dim = pe_source_cfg.get('pe_dim', 96)
        if pe_type == 'gaussian':
            pe_dim = pe_source_cfg.get('num_centers', pe_dim)
        elif pe_type == 'multi_scale':
            scale_dims = pe_source_cfg.get('pe_scale_dims')
            if scale_dims:
                pe_dim = sum(scale_dims)
        pe_type_short = {'sinusoidal': 'sin', 'gaussian': 'gaus', 'multi_scale': 'ms'}.get(pe_type, pe_type[:3])
        parts.append(f'pe{pe_type_short}{pe_dim}')
    
    if cfg.get('use_adaptive_knn', False):
        parts.append('aknn')
    
    # Decoder
    if cfg.get('decoder', {}).get('use_hybrid', False):
        parts.append('hybrid')
    
    # Laplace attention distance (only tag when non-default to avoid filename churn)
    laplace_p = cfg.get('laplace_distance_p', 1.0)
    if laplace_p != 1.0:
        # Strip trailing zeros for cleaner filenames (e.g., 2.0 -> 2)
        laplace_str = str(laplace_p).rstrip('0').rstrip('.') if isinstance(laplace_p, float) else str(laplace_p)
        parts.append(f"lp{laplace_str}")

    # Architecture details
    bottleneck_size = cfg.get('bottleneck_size')
    r_dim = cfg.get('r_dim', bottleneck_size)
    w_dim = cfg.get('w_dim', bottleneck_size)
    v_dim = cfg.get('v_dim', bottleneck_size)
    
    if r_dim is not None:
        parts.append(f"r{r_dim}")
    if w_dim is not None:
        parts.append(f"w{w_dim}")
    if v_dim is not None:
        parts.append(f"v{v_dim}")

    parts.append(f"h{cfg.get('hidden_size', 192)}")

    # Include attention head count so filenames capture encoder width fully
    num_heads = cfg.get('num_heads')
    if num_heads is not None:
        parts.append(f"nh{num_heads}")

    if training_cfg and 'batch_size' in training_cfg:
        parts.append(f"bsz{training_cfg['batch_size']}")
    
    return parts

def get_gcnp_filename_suffix_parts(gcnp_cfg, training_cfg=None):
    """
    Generates a list of descriptive filename suffix parts for GCNP models based on config.
    """
    parts = []

    parts.append(f"rd{gcnp_cfg.get('r_dim', 128)}")
    parts.append(f"wd{gcnp_cfg.get('w_dim', 128)}")
    parts.append(f"vd{gcnp_cfg.get('v_dim', 128)}")
    parts.append(f"hs{gcnp_cfg.get('hidden_size', 256)}")
    parts.append(f"dr{gcnp_cfg.get('dropout_rate', 0.025)}")
    parts.append(f"k{gcnp_cfg.get('k', 32)}")
    parts.append(f"nlmlp{gcnp_cfg.get('num_layers_mlp', 1)}")

    # Add smooth attention configuration to suffix
    if gcnp_cfg.get('use_gaussian', False):
        parts.append("gauss")
        parts.append(f"gbw{gcnp_cfg.get('gaussian_init_bandwidth', 5.0)}")
    else:
        parts.append("laplace")
        parts.append(f"ltemp{gcnp_cfg.get('laplace_init_temp', 5.0)}")
        parts.append(f"dist{gcnp_cfg.get('distance_metric', 'l2')}")
        
    parts.extend(_get_pe_suffix_parts(gcnp_cfg))

    if training_cfg and 'batch_size' in training_cfg:
        parts.append(f"bsz{training_cfg['batch_size']}")

    return parts
