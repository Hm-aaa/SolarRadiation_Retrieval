import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import separate_spatial_attributes
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial.distance import cdist

### UPDATED IMPORTS ###
try:
    from torch_geometric.nn import GATConv, knn_graph
    from torch_cluster import knn
except ImportError:
    raise ImportError("This model requires torch_geometric and torch_cluster. Please install them: pip install torch-geometric torch-cluster")


class AdaptiveStationConditioning:
    """
    Adaptive station conditioning that balances model predictions with observations.
    
    Key principles:
    1. Only condition where model predictions significantly deviate from observations
    2. Use observation uncertainty to weight conditioning strength
    3. Preserve spatial coherence by using smooth blending kernels
    4. Respect the model's learned uncertainty estimates
    """
    
    def __init__(
        self,
        exact_match_threshold: float = 1e-5,
        influence_radius: float = 0.02,
        base_blend_weight: float = 0.5,
        hard_match_weight: float = 0.9,
        use_adaptive_strength: bool = True,
        deviation_threshold: float = 2.0,
        min_conditioning_strength: float = 0.1,
        max_conditioning_strength: float = 0.9,
        kernel_type: str = "gaussian",
        kernel_sharpness: float = 3.0,
        use_observation_uncertainty: bool = True,
        default_obs_uncertainty_temp: float = 0.5,
        default_obs_uncertainty_precip: float = 0.2,
        enable_outlier_detection: bool = True,
        outlier_threshold: float = 4.0,
    ):
        self.exact_match_threshold = exact_match_threshold
        self.influence_radius = influence_radius
        self.base_blend_weight = base_blend_weight
        self.hard_match_weight = hard_match_weight
        self.use_adaptive_strength = use_adaptive_strength
        self.deviation_threshold = deviation_threshold
        self.min_conditioning_strength = min_conditioning_strength
        self.max_conditioning_strength = max_conditioning_strength
        self.kernel_type = kernel_type
        self.kernel_sharpness = kernel_sharpness
        self.use_observation_uncertainty = use_observation_uncertainty
        self.default_obs_uncertainty_temp = default_obs_uncertainty_temp
        self.default_obs_uncertainty_precip = default_obs_uncertainty_precip
        self.enable_outlier_detection = enable_outlier_detection
        self.outlier_threshold = outlier_threshold
    
    def compute_kernel_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute smooth kernel weights based on distance."""
        normalized_dist = distances / (self.influence_radius + 1e-8)
        
        if self.kernel_type == "gaussian":
            weights = torch.exp(-self.kernel_sharpness * normalized_dist ** 2)
        elif self.kernel_type == "wendland":
            clamped = torch.clamp(1 - normalized_dist, min=0)
            weights = clamped ** 4 * (4 * normalized_dist + 1)
        elif self.kernel_type == "linear":
            weights = torch.clamp(1 - normalized_dist, min=0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        weights = weights * (distances < self.influence_radius).float()
        return weights
    
    def compute_adaptive_strength(
        self,
        predicted_mean: torch.Tensor,
        predicted_std: torch.Tensor,
        observed_value: torch.Tensor,
    ) -> torch.Tensor:
        deviation = torch.abs(predicted_mean - observed_value)
        z_score = deviation / (predicted_std + 1e-8)
        excess_deviation = torch.clamp(z_score - self.deviation_threshold, min=0)
        strength = self.min_conditioning_strength + (
            self.max_conditioning_strength - self.min_conditioning_strength
        ) * torch.sigmoid(excess_deviation - 1)
        return strength
    
    def detect_outliers(
        self,
        predicted_mean: torch.Tensor,
        predicted_std: torch.Tensor,
        observed_value: torch.Tensor,
    ) -> torch.Tensor:
        deviation = torch.abs(predicted_mean - observed_value)
        z_score = deviation / (predicted_std + 1e-8)
        is_outlier = z_score > self.outlier_threshold
        return is_outlier
    
    def __call__(
        self,
        predictions_dict: Dict[str, torch.Tensor],
        x_context_all: torch.Tensor,
        y_context: torch.Tensor,
        x_target_all: torch.Tensor,
        spatial_indices: list,
        obs_uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = x_context_all.device
        s_context = x_context_all[..., spatial_indices]
        s_target = x_target_all[..., spatial_indices]
        
        B, M, _ = s_target.shape
        distances = torch.cdist(s_target, s_context, p=2)
        
        min_distances, nearest_idx = distances.min(dim=-1)
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, M)
        nearest_obs = y_context[batch_idx, nearest_idx]
        
        kernel_weights = self.compute_kernel_weights(min_distances)
        exact_match_mask = min_distances < self.exact_match_threshold
        
        conditioned = {}
        
        for key, pred in predictions_dict.items():
            if pred is None:
                conditioned[key] = None
                continue
            
            is_temp = 'temp' in key.lower()
            is_precip = 'precip' in key.lower() and 'prob' not in key.lower()
            is_variance = 'variance' in key.lower()
            
            if not (is_temp or is_precip):
                conditioned[key] = pred
                continue
            
            obs_channel = 0 if is_temp else 1
            obs_values = nearest_obs[..., obs_channel:obs_channel+1]
            
            if is_variance:
                variance_reduction = 1.0 - kernel_weights.unsqueeze(-1) * 0.8
                min_var = 1e-6
                variance_reduction = torch.where(
                    exact_match_mask.unsqueeze(-1),
                    torch.full_like(variance_reduction, 0.05),
                    variance_reduction
                )
                conditioned_pred = torch.clamp(pred * variance_reduction, min=min_var)
            else:
                var_key = key.replace('_mu', '_variance')
                if var_key in predictions_dict and predictions_dict[var_key] is not None:
                    pred_std = torch.sqrt(predictions_dict[var_key] + 1e-8)
                else:
                    pred_std = torch.ones_like(pred) * 0.5
                
                if self.use_adaptive_strength:
                    adaptive_strength = self.compute_adaptive_strength(
                        pred, pred_std, obs_values
                    )
                    blend_weights = kernel_weights.unsqueeze(-1) * adaptive_strength
                else:
                    blend_weights = kernel_weights.unsqueeze(-1) * self.base_blend_weight
                
                if self.enable_outlier_detection:
                    is_outlier = self.detect_outliers(pred, pred_std, obs_values)
                    blend_weights = torch.where(
                        is_outlier,
                        blend_weights * 0.1,
                        blend_weights
                    )
                
                blend_weights = torch.where(
                    exact_match_mask.unsqueeze(-1),
                    torch.full_like(blend_weights, self.hard_match_weight),
                    blend_weights
                )
                
                conditioned_pred = (1.0 - blend_weights) * pred + blend_weights * obs_values
            
            conditioned[key] = conditioned_pred
        
        return conditioned


def apply_station_conditioning_conservative(
    predictions_dict: Dict[str, torch.Tensor],
    x_context_all: torch.Tensor,
    y_context: torch.Tensor,
    x_target_all: torch.Tensor,
    spatial_indices: list,
    conditioning_mode: str = "adaptive",
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Wrapper function for station conditioning with preset configurations.
    """
    if conditioning_mode == "none":
        return predictions_dict
    
    presets = {
        "minimal": {
            "exact_match_threshold": 1e-5,
            "influence_radius": 1e-5,
            "hard_match_weight": 0.95,
            "use_adaptive_strength": False,
            "base_blend_weight": 0.0,
        },
        "conservative": {
            "exact_match_threshold": 1e-5,
            "influence_radius": 0.01,
            "hard_match_weight": 0.9,
            "use_adaptive_strength": True,
            "deviation_threshold": 2.5,
            "min_conditioning_strength": 0.05,
            "max_conditioning_strength": 0.7,
            "kernel_type": "wendland",
            "enable_outlier_detection": True,
            "outlier_threshold": 4.0,
        },
        "moderate": {
            "exact_match_threshold": 1e-5,
            "influence_radius": 0.02,
            "hard_match_weight": 0.95,
            "use_adaptive_strength": True,
            "deviation_threshold": 2.0,
            "min_conditioning_strength": 0.1,
            "max_conditioning_strength": 0.8,
            "kernel_type": "gaussian",
            "kernel_sharpness": 3.0,
            "enable_outlier_detection": True,
        },
        "aggressive": {
            "exact_match_threshold": 1e-5,
            "influence_radius": 0.03,
            "hard_match_weight": 0.98,
            "use_adaptive_strength": False,
            "base_blend_weight": 0.8,
            "kernel_type": "gaussian",
            "kernel_sharpness": 2.0,
            "enable_outlier_detection": False,
        },
    }
    
    if conditioning_mode not in presets:
        raise ValueError(f"Unknown mode: {conditioning_mode}. Choose from {list(presets.keys())}")
    
    config = {**presets[conditioning_mode], **kwargs}
    conditioner = AdaptiveStationConditioning(**config)
    
    return conditioner(
        predictions_dict=predictions_dict,
        x_context_all=x_context_all,
        y_context=y_context,
        x_target_all=x_target_all,
        spatial_indices=spatial_indices,
    )


# --- Helper function for building MLPs ---
def _create_mlp(input_dim, output_dim, hidden_size, num_hidden_layers, dropout_rate, final_activation=None):
    """Helper function to create a multi-layer perceptron."""
    layers = []
    if num_hidden_layers == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, output_dim))

    if final_activation:
        layers.append(final_activation)
    return nn.Sequential(*layers)


# ============================ 
# SPATIAL POSITIONAL ENCODING (FIXED)
# ============================ 
class SpatialPositionalEncoding(nn.Module):
    """
    Generates positional encodings for spatial coordinates.
    Automatically handles normalized [0, 1] coordinates.
    """
    def __init__(self, spatial_dim=2, encoding_dim=64, 
                 coordinate_range='normalized',  
                 original_bounds=None,
                 max_wavelength=None,
                 learnable=False, use_gaussian=False):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.encoding_dim = encoding_dim
        self.use_gaussian = use_gaussian
        self.coordinate_range = coordinate_range
        
        # Validate configuration
        if coordinate_range == 'original' and original_bounds is None:
            raise ValueError("original_bounds must be provided when using coordinate_range='original'")
        
        # Store original bounds for denormalization if needed
        if original_bounds is not None:
            self.register_buffer('original_bounds', torch.tensor(original_bounds, dtype=torch.float32))
        else:
            self.original_bounds = None
        
        # Auto-compute max_wavelength based on coordinate range
        if max_wavelength is None:
            max_wavelength = 10.0 if coordinate_range == 'normalized' else 10000.0
        
        if use_gaussian:
            # =================================================================
            # IMPLICIT CENTER STRATEGY FOR ROMANIAN TERRITORY
            # =================================================================
            # We use a fixed 10x10 grid (100 centers) to ensure adequate coverage
            # of the normalized domain, regardless of the requested encoding_dim.
            # If encoding_dim < 100, we learn a projection.
            # =================================================================
            grid_resolution = 10
            num_implicit_centers = grid_resolution ** 2  # 100 centers
            
            # Create grid on [0, 1]^2
            x = torch.linspace(0, 1, grid_resolution)
            y = torch.linspace(0, 1, grid_resolution)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            grid_centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1) # (100, 2)
            
            # Handle coordinate range scaling
            if coordinate_range == 'normalized':
                centers = grid_centers
                # Width approx matching grid spacing (0.1)
                log_widths = torch.log(torch.ones(num_implicit_centers) * (1.0 / grid_resolution))
            else:
                # Naive scaling for original coords (risky but consistent with previous logic)
                centers = grid_centers * 100
                log_widths = torch.zeros(num_implicit_centers)
            
            # Register parameters
            if learnable:
                self.centers = nn.Parameter(centers)
                self.log_widths = nn.Parameter(log_widths)
            else:
                self.register_buffer('centers', centers)
                self.register_buffer('log_widths', log_widths)
                
            # Projection layer if output dim differs from implicit centers
            self.num_centers = num_implicit_centers
            if self.num_centers != encoding_dim:
                self.gaussian_proj = nn.Linear(self.num_centers, encoding_dim)
                # Initialize to keep signal magnitude
                nn.init.xavier_uniform_(self.gaussian_proj.weight)
                nn.init.zeros_(self.gaussian_proj.bias)
            else:
                self.gaussian_proj = None
                
        else:
            # Sinusoidal encoding - ensure exact dimension match
            # Each spatial dimension gets pairs of sin/cos
            num_pairs_per_dim = encoding_dim // (2 * spatial_dim)
            remaining = encoding_dim - (num_pairs_per_dim * 2 * spatial_dim)
            self.num_pairs_per_dim = num_pairs_per_dim
            self.remaining = remaining
            
            # Create linear projection layer in __init__ if needed
            if remaining > 0:
                self.linear_proj = nn.Linear(spatial_dim, remaining)
                nn.init.xavier_uniform_(self.linear_proj.weight)
            else:
                self.linear_proj = None
            
            # Treat freq_bands as true frequencies (not divisors) for standard 2π scaling
            min_freq = 0.5 if coordinate_range == 'normalized' else 1.0
            freq_bands = torch.exp(torch.linspace(
                np.log(min_freq), np.log(max_wavelength), num_pairs_per_dim
            ))
            if learnable:
                self.freq_bands = nn.Parameter(freq_bands)
            else:
                self.register_buffer('freq_bands', freq_bands)
    
    def denormalize_coords(self, coords):
        """Convert [0, 1] normalized coordinates back to original scale."""
        if self.original_bounds is not None:
            mins = self.original_bounds[:, 0].view(1, 1, -1)
            maxs = self.original_bounds[:, 1].view(1, 1, -1)
            return coords * (maxs - mins) + mins
        return coords
    
    def forward(self, spatial_coords):
        """
        Args:
            spatial_coords: (B, N, spatial_dim) tensor of coordinates
        Returns:
            (B, N, encoding_dim) positional encodings
        """
        B, N, D = spatial_coords.shape
        
        # Optionally denormalize coordinates
        if self.coordinate_range == 'original' and self.original_bounds is not None:
            coords = self.denormalize_coords(spatial_coords)
        else:
            coords = spatial_coords
        
        if self.use_gaussian:
            # Gaussian RBF encoding
            # (B, N, D) vs (1, implicit_centers, D) -> (B, N, implicit_centers)
            dists = torch.cdist(coords, self.centers.unsqueeze(0).expand(B, -1, -1), p=2)
            widths = torch.exp(self.log_widths).unsqueeze(0).unsqueeze(0) + 1e-8
            
            # Raw RBF features (B, N, 100)
            encoding = torch.exp(-dists ** 2 / (2 * widths ** 2))
            
            # Project to requested dimension if needed (e.g., 100 -> 32)
            if self.gaussian_proj is not None:
                encoding = self.gaussian_proj(encoding)
            
            return encoding
        else:
            # Sinusoidal encoding - guaranteed to produce exact encoding_dim
            encodings = []
            for dim in range(D):
                coord = coords[..., dim:dim+1]
                # Standard sinusoidal PE: use frequencies with 2π scaling
                scaled = 2 * math.pi * coord * (self.freq_bands.unsqueeze(0).unsqueeze(0) + 1e-8)
                sin_enc = torch.sin(scaled)
                cos_enc = torch.cos(scaled)
                encodings.append(sin_enc)
                encodings.append(cos_enc)
            
            pe = torch.cat(encodings, dim=-1)
            
            # Use pre-created linear projection
            if self.remaining > 0 and self.linear_proj is not None:
                extra_features = self.linear_proj(coords)
                pe = torch.cat([pe, extra_features], dim=-1)
            
            # Assert instead of truncate
            assert pe.shape[-1] == self.encoding_dim, \
                f"Dimension mismatch: got {pe.shape[-1]}, expected {self.encoding_dim}"
            
            return pe


class MultiScalePositionalEncoding(nn.Module):
    """
    Multi-scale positional encoding with proper dimension handling.
    """
    def __init__(self, spatial_dim=2, encoding_dims=None, 
                 scales=None, coordinate_range='normalized',
                 original_bounds=None, learnable=False):
        super().__init__()
        
        # Ensure encoding_dims are provided
        if encoding_dims is None:
            encoding_dims = [32, 32, 32]
        
        # Auto-compute scales based on coordinate range
        if scales is None:
            if coordinate_range == 'normalized':
                scales = [0.1, 0.5, 1.0]
            else:
                scales = [1.0, 10.0, 100.0]
        
        # Ensure matching dimensions
        if len(scales) != len(encoding_dims):
            raise ValueError(f"scales ({len(scales)}) and encoding_dims ({len(encoding_dims)}) must have same length")
        
        self.scales = scales
        self.encoding_dims = encoding_dims
        self.total_dim = sum(encoding_dims)
        
        self.encoders = nn.ModuleList([
            SpatialPositionalEncoding(
                spatial_dim, dim, 
                coordinate_range=coordinate_range,
                original_bounds=original_bounds,
                max_wavelength=scale * (10.0 if coordinate_range == 'normalized' else 10000.0),
                learnable=learnable
            )
            for dim, scale in zip(encoding_dims, scales)
        ])
        
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
    def forward(self, spatial_coords):
        encodings = []
        weights = torch.softmax(self.scale_weights.to(spatial_coords.device), dim=0)
        
        for i, encoder in enumerate(self.encoders):
            encoding = encoder(spatial_coords) * weights[i]
            encodings.append(encoding)
        
        return torch.cat(encodings, dim=-1)


# ============================ 
# ADAPTIVE KNN MODULE (IMPROVED)
# ============================ 
class AdaptiveKNN:
    """Dynamically adjusts k based on local point density."""
    def __init__(self, min_k=5, max_k=50, density_percentile=0.25, use_median=True):
        self.min_k = min_k
        self.max_k = max_k
        self.density_percentile = density_percentile
        self.use_median = use_median
    
    def compute_adaptive_k(self, s_points, base_k):
        """Compute adaptive k for each point based on local density."""
        B, N, D = s_points.shape
        
        # Handle small context sets
        if N <= max(base_k, self.min_k):
            return max(1, min(base_k, N - 1, self.max_k))
        
        # Compute pairwise distances
        distances = torch.cdist(s_points, s_points, p=2)
        
        # Get distance to k-th nearest neighbor as density measure
        distances_copy = distances.clone()
        distances_copy.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        
        # Sort and get k-th nearest distance
        k_for_density = min(base_k, N - 1)
        kth_distances, _ = torch.topk(distances_copy, k_for_density, dim=-1, largest=False)
        local_density_measure = kth_distances.mean(dim=-1)
        
        # Compute adaptive k: inverse relationship with density
        density_quantiles = torch.quantile(
            local_density_measure, 
            torch.tensor([self.density_percentile, 1-self.density_percentile], 
                        device=s_points.device), 
            dim=1).T
        
        # Normalize density to [0, 1] with numerical stability
        density_range = (density_quantiles[:, 1] - density_quantiles[:, 0]).unsqueeze(1) + 1e-8
        density_normalized = (local_density_measure - density_quantiles[:, 0].unsqueeze(1)) / density_range
        density_normalized = density_normalized.clamp(0, 1)
        
        # Map to k range (inverse relationship)
        adaptive_k = self.min_k + (1 - density_normalized) * (self.max_k - self.min_k)
        
        # Use median instead of mean for robustness
        if self.use_median:
            adaptive_k = adaptive_k.median().int().item()
        else:
            adaptive_k = adaptive_k.mean().int().item()
        
        return min(max(adaptive_k, self.min_k), min(self.max_k, N - 1))


# --- Helper function for finding nearest neighbors ---
def find_k_nearest_context_points(s_target, s_context, k, use_adaptive=False, adaptive_knn=None):
    """For each target point, finds the k nearest context points."""
    B, M, D = s_target.shape
    N = s_context.shape[1]
    
    # Use adaptive k if enabled
    if use_adaptive and adaptive_knn is not None:
        k = adaptive_knn.compute_adaptive_k(s_context, k)
    
    k = min(k, N)

    if M == 0:
        return torch.empty((B, 0, k), dtype=torch.long, device=s_target.device)

    # Flatten the batches
    s_target_flat = s_target.view(B * M, D)
    s_context_flat = s_context.view(B * N, D)

    # Create batch vectors
    batch_target = torch.arange(B, device=s_target.device).repeat_interleave(M)
    batch_context = torch.arange(B, device=s_context.device).repeat_interleave(N)

    assign_index = knn(s_context_flat, s_target_flat, k, batch_x=batch_context, batch_y=batch_target)
    context_indices_flat, _ = assign_index

    # Convert global flattened indices back to per-batch local indices
    nearest_indices_local = context_indices_flat % N

    # Reshape to (B, M, k)
    return nearest_indices_local.view(B, M, k), k


# ============================ 
# CROSS ATTENTION MODULES
# ============================ 
class MultiHeadCrossAttentionWrapper(nn.Module):
    """Wrapper for multi-head cross attention supporting separate key/value inputs."""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
    
    def forward(self, query, key, value):
        """
        Args:
            query: (B, M, embed_dim) - queries from target
            key: (B, N, embed_dim) - keys from context
            value: (B, N, embed_dim) - values from context (can differ from key!)
        """
        attn_output, _ = self.attention(query, key, value, need_weights=False)
        return attn_output


class LaplaceAttention(nn.Module):
    """Laplace kernel attention for spatial relationships."""
    def __init__(self, init_temp=1.0, min_temp=1e-3, max_temp=10.0, p_norm=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.p_norm = float(p_norm)
        if self.p_norm <= 0:
            raise ValueError("p_norm must be greater than 0 for LaplaceAttention.")

    def forward(self, query, key, value):
        """
        Paper-faithful Laplace attention.
        Args:
            query: s* (target spatial coords)
            key: s (context spatial coords)  
            value: w = MLP(s, y) (encoded context)
        """
        dists = torch.cdist(query, key, p=self.p_norm)
        temp = torch.exp(self.log_temp).clamp(min=self.min_temp, max=self.max_temp)
        weights = F.softmax(-dists / (temp + 1e-8), dim=-1)
        return torch.bmm(weights, value)


# ============================ 
# ORIGINAL SMACNP MODULES
# ============================ 
class MeanLocationEncoder(nn.Module):
    """
    PAPER-FAITHFUL: Mean-location encoder using Laplace attention.
    
    From paper Section 3.1:
    - w = MLP_φ(s, y) maps (s, y) into representation
    - w* = Laplace attention(query=s*, key=s, value=w)
    
    Note: This is ALREADY paper-faithful because:
    - Key = s (spatial coords only)
    - Value = w = MLP(s, y) (includes target)
    """
    def __init__(self, spatial_dim, y_dim, hidden_size, w_dim, dropout_rate, num_layers_mlp=1,
                 laplace_p=1.0):
        super().__init__()
        self.mlp_pre_attn = _create_mlp(spatial_dim + y_dim, w_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.laplace_attention = LaplaceAttention(p_norm=laplace_p)
    
    def forward(self, s_context, y_context, s_target):
        # w = MLP(s, y)
        sy_context = torch.cat([s_context, y_context], dim=-1)
        w = self.mlp_pre_attn(sy_context)
        
        # Laplace attention: query=s*, key=s, value=w
        return self.laplace_attention(query=s_target, key=s_context, value=w)


class MeanDecoder(nn.Module):
    """Decoder for mean predictions, combining r* and w*."""
    def __init__(self, r_dim, w_dim, spatial_dim, attr_dim, hidden_size, y_dim, dropout_rate, 
                 num_layers_mlp=1, final_activation=None):
        super().__init__()
        input_decoder_dim = r_dim + w_dim + spatial_dim + attr_dim
        self.mlp_decoder = _create_mlp(input_decoder_dim, y_dim, hidden_size, num_layers_mlp, 
                                       dropout_rate, final_activation)
    
    def forward(self, r_star, w_star, s_target, x_attr_target):
        decoder_input = torch.cat([r_star, w_star, s_target, x_attr_target], dim=-1)
        return self.mlp_decoder(decoder_input)


class VarianceDecoder(nn.Module):
    """Decoder for variance predictions."""
    def __init__(self, v_dim, spatial_dim, attr_dim, hidden_size, y_dim, dropout_rate, num_layers_mlp=1):
        super().__init__()
        input_decoder_dim = v_dim + spatial_dim + attr_dim
        self.mlp_decoder = _create_mlp(input_decoder_dim, y_dim, hidden_size, num_layers_mlp, 
                                       dropout_rate, nn.Softplus())
    
    def forward(self, v_star, s_target, x_attr_target):
        decoder_input = torch.cat([v_star, s_target, x_attr_target], dim=-1)
        return self.mlp_decoder(decoder_input) + 1e-6


# ============================ 
# PAPER-FAITHFUL GLOBAL ENCODERS
# ============================ 
class GlobalMeanAttributeEncoder(nn.Module):
    """
    PAPER-FAITHFUL: Mean-attribute encoder with SEPARATED key and value encodings.
    
    From paper Section 3.1:
    - r = MLP_θ(x, y) maps (x, y) into representation (VALUE)
    - Multi-head attention where query attends to x (KEY) to retrieve r (VALUE)
    - r* = Multi-head attention(r, x, x*)
    
    Key insight: Keys are derived from x ONLY (no y), but values include y.
    This prevents attention weights from being influenced by target values.
    """
    def __init__(self, input_dim, y_dim, hidden_size, r_dim, num_heads, dropout_rate, 
                 num_layers_mlp=1, use_positional_encoding=False, pe_dim=64, pe_type='sinusoidal', 
                 pe_scales=None, pe_scale_dims=None, pe_coordinate_range='normalized', 
                 pe_original_bounds=None, pe_learnable=False, use_layer_norm=True, **kwargs):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        # Initialize positional encoding
        if use_positional_encoding:
            if pe_type == 'multi_scale':
                if pe_scale_dims is None:
                    num_scales = len(pe_scales) if pe_scales else 3
                    base_dim = pe_dim // num_scales
                    remainder = pe_dim % num_scales
                    pe_scale_dims = [base_dim] * num_scales
                    if remainder > 0:
                        pe_scale_dims[-1] += remainder
                
                self.pos_encoder = MultiScalePositionalEncoding(
                    spatial_dim=2,
                    encoding_dims=pe_scale_dims,
                    scales=pe_scales,
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds,
                    learnable=pe_learnable
                )
                actual_pe_dim = sum(pe_scale_dims)
            else:
                self.pos_encoder = SpatialPositionalEncoding(
                    spatial_dim=2,
                    encoding_dim=pe_dim,
                    learnable=pe_learnable,
                    use_gaussian=(pe_type == 'gaussian'),
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds
                )
                actual_pe_dim = pe_dim
            
            actual_input_dim = input_dim + actual_pe_dim
        else:
            actual_input_dim = input_dim
        
        # ============================================
        # PAPER-FAITHFUL: Separate Key and Value MLPs
        # ============================================
        
        # Value encoding: r = MLP(x, y) - includes target values
        self.mlp_value = _create_mlp(actual_input_dim + y_dim, r_dim, hidden_size, 
                                     num_layers_mlp, dropout_rate)
        
        # Key encoding: MLP(x) - attributes ONLY, NO target values!
        self.mlp_key = _create_mlp(actual_input_dim, r_dim, hidden_size, 
                                   num_layers_mlp, dropout_rate)
        
        # Query encoding: MLP(x*) - target attributes
        self.mlp_query = _create_mlp(actual_input_dim, r_dim, hidden_size, 
                                     num_layers_mlp, dropout_rate)
        
        self.cross_attention = MultiHeadCrossAttentionWrapper(r_dim, num_heads, dropout_rate)
        
        if use_layer_norm:
            self.key_norm = nn.LayerNorm(r_dim)
            self.value_norm = nn.LayerNorm(r_dim)
            self.query_norm = nn.LayerNorm(r_dim)
            self.output_norm = nn.LayerNorm(r_dim)
    
    def forward(self, x_context_all, y_context, x_target_all, s_context=None, s_target=None, **kwargs):
        # Add positional encoding if enabled
        if self.use_positional_encoding and s_context is not None:
            pe_context = self.pos_encoder(s_context)
            pe_target = self.pos_encoder(s_target)
            x_context_enhanced = torch.cat([x_context_all, pe_context], dim=-1)
            x_target_enhanced = torch.cat([x_target_all, pe_target], dim=-1)
        else:
            x_context_enhanced = x_context_all
            x_target_enhanced = x_target_all
        
        # ============================================
        # PAPER-FAITHFUL: Separate Key and Value
        # ============================================
        
        # Value: r = MLP(x, y) - includes target values
        xy_context = torch.cat([x_context_enhanced, y_context], dim=-1)
        r_value = self.mlp_value(xy_context)
        
        # Key: MLP(x) - NO target values! Pure attribute-based similarity
        r_key = self.mlp_key(x_context_enhanced)
        
        # Query: MLP(x*) - target attributes
        r_query = self.mlp_query(x_target_enhanced)
        
        if self.use_layer_norm:
            r_value = self.value_norm(r_value)
            r_key = self.key_norm(r_key)
            r_query = self.query_norm(r_query)
        
        # Cross attention with SEPARATE key and value
        output = self.cross_attention(query=r_query, key=r_key, value=r_value)
        
        if self.use_layer_norm:
            output = self.output_norm(output)
        
        return output


class GlobalVarianceEncoder(nn.Module):
    """
    PAPER-FAITHFUL: Variance encoder.
    
    From paper Section 3.1:
    - Variance is INDEPENDENT of target values y
    - v = MLP_φ(s, x) - encodes spatial + attributes
    - v* = Multi-head attention(v, s*, x*)
    
    Note: Since variance encoder doesn't use y, the key/value separation 
    is less critical. We still implement it consistently for clarity.
    """
    def __init__(self, input_dim, hidden_size, v_dim, num_heads, dropout_rate, 
                 num_layers_mlp=1, use_positional_encoding=False, pe_dim=64, pe_type='sinusoidal', 
                 pe_scales=None, pe_scale_dims=None, pe_coordinate_range='normalized', 
                 pe_original_bounds=None, pe_learnable=False, use_layer_norm=True, spatial_dim=2, **kwargs):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        # Initialize positional encoding
        if use_positional_encoding:
            if pe_type == 'multi_scale':
                if pe_scale_dims is None:
                    num_scales = len(pe_scales) if pe_scales else 3
                    base_dim = pe_dim // num_scales
                    remainder = pe_dim % num_scales
                    pe_scale_dims = [base_dim] * num_scales
                    if remainder > 0:
                        pe_scale_dims[-1] += remainder
                
                self.pos_encoder = MultiScalePositionalEncoding(
                    spatial_dim=2,
                    encoding_dims=pe_scale_dims,
                    scales=pe_scales,
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds,
                    learnable=pe_learnable
                )
                actual_pe_dim = sum(pe_scale_dims)
            else:
                self.pos_encoder = SpatialPositionalEncoding(
                    spatial_dim=2,
                    encoding_dim=pe_dim,
                    learnable=pe_learnable,
                    use_gaussian=(pe_type == 'gaussian'),
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds
                )
                actual_pe_dim = pe_dim
            
            actual_input_dim = input_dim + actual_pe_dim
        else:
            actual_input_dim = input_dim
        
        # Include spatial dimension in input (variance encoder uses s + x)
        actual_input_dim += spatial_dim
        
        # v = MLP(s, x) for both key and value (no y involved anyway)
        self.mlp_kv = _create_mlp(actual_input_dim, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        
        # Query: MLP(s*, x*)
        self.mlp_query = _create_mlp(actual_input_dim, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        
        self.cross_attention = MultiHeadCrossAttentionWrapper(v_dim, num_heads, dropout_rate)
        
        if use_layer_norm:
            self.kv_norm = nn.LayerNorm(v_dim)
            self.query_norm = nn.LayerNorm(v_dim)
            self.output_norm = nn.LayerNorm(v_dim)
    
    def forward(self, x_context_all, x_target_all, s_context=None, s_target=None, **kwargs):
        # Add positional encoding if enabled
        if self.use_positional_encoding and s_context is not None:
            pe_context = self.pos_encoder(s_context)
            pe_target = self.pos_encoder(s_target)
            x_context_enhanced = torch.cat([x_context_all, pe_context], dim=-1)
            x_target_enhanced = torch.cat([x_target_all, pe_target], dim=-1)
        else:
            x_context_enhanced = x_context_all
            x_target_enhanced = x_target_all
        
        # Concatenate spatial coords (variance uses s + x)
        x_context_enhanced = torch.cat([s_context, x_context_enhanced], dim=-1)
        x_target_enhanced = torch.cat([s_target, x_target_enhanced], dim=-1)
        
        # v = MLP(s, x) - no y involved
        v_kv = self.mlp_kv(x_context_enhanced)
        v_query = self.mlp_query(x_target_enhanced)
        
        if self.use_layer_norm:
            v_kv = self.kv_norm(v_kv)
            v_query = self.query_norm(v_query)
        
        output = self.cross_attention(query=v_query, key=v_kv, value=v_kv)
        
        if self.use_layer_norm:
            output = self.output_norm(output)
        
        return output


# ============================ 
# PAPER-FAITHFUL LOCALIZED ENCODERS
# ============================ 
class LocalizedMeanAttributeEncoder(nn.Module):
    """
    PAPER-FAITHFUL: Localized mean-attribute encoder with k-NN attention
    and SEPARATED key/value encodings.
    """
    def __init__(self, input_dim, y_dim, hidden_size, r_dim, num_heads, dropout_rate, k, 
                 num_layers_mlp=1, use_adaptive_knn=False, adaptive_knn_params=None,
                 use_positional_encoding=False, pe_dim=64, pe_type='sinusoidal', 
                 pe_scales=None, pe_scale_dims=None, pe_coordinate_range='normalized', 
                 pe_original_bounds=None, pe_learnable=False,
                 use_layer_norm=True, **kwargs):
        super().__init__()
        self.k = k
        self.r_dim = r_dim
        self.use_adaptive_knn = use_adaptive_knn
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        if use_adaptive_knn and adaptive_knn_params:
            self.adaptive_knn = AdaptiveKNN(**adaptive_knn_params)
        else:
            self.adaptive_knn = None
        
        # Initialize positional encoding
        if use_positional_encoding:
            if pe_type == 'multi_scale':
                if pe_scale_dims is None:
                    num_scales = len(pe_scales) if pe_scales else 3
                    base_dim = pe_dim // num_scales
                    remainder = pe_dim % num_scales
                    pe_scale_dims = [base_dim] * num_scales
                    if remainder > 0:
                        pe_scale_dims[-1] += remainder
                
                self.pos_encoder = MultiScalePositionalEncoding(
                    spatial_dim=2,
                    encoding_dims=pe_scale_dims,
                    scales=pe_scales,
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds,
                    learnable=pe_learnable
                )
                actual_pe_dim = sum(pe_scale_dims)
            else:
                self.pos_encoder = SpatialPositionalEncoding(
                    spatial_dim=2,
                    encoding_dim=pe_dim,
                    learnable=pe_learnable,
                    use_gaussian=(pe_type == 'gaussian'),
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds
                )
                actual_pe_dim = pe_dim
            
            actual_input_dim = input_dim + actual_pe_dim
        else:
            actual_input_dim = input_dim
        
        # ============================================
        # PAPER-FAITHFUL: Separate Key and Value MLPs
        # ============================================
        
        # Value: MLP(x, y) - includes target
        self.mlp_value = _create_mlp(actual_input_dim + y_dim, r_dim, hidden_size, 
                                     num_layers_mlp, dropout_rate)
        
        # Key: MLP(x) - NO target values
        self.mlp_key = _create_mlp(actual_input_dim, r_dim, hidden_size, 
                                   num_layers_mlp, dropout_rate)
        
        # Query: MLP(x*)
        self.mlp_query = _create_mlp(actual_input_dim, r_dim, hidden_size, 
                                     num_layers_mlp, dropout_rate)
        
        self.cross_attention = MultiHeadCrossAttentionWrapper(r_dim, num_heads, dropout_rate)
        
        if use_layer_norm:
            self.key_norm = nn.LayerNorm(r_dim)
            self.value_norm = nn.LayerNorm(r_dim)
            self.query_norm = nn.LayerNorm(r_dim)
            self.output_norm = nn.LayerNorm(r_dim)
    
    def forward(self, x_context_all, y_context, x_target_all, s_context, s_target, **kwargs):
        B, M, _ = x_target_all.shape
        N = x_context_all.shape[1]
        D = self.r_dim

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            pe_context = self.pos_encoder(s_context)
            pe_target = self.pos_encoder(s_target)
            x_context_enhanced = torch.cat([x_context_all, pe_context], dim=-1)
            x_target_enhanced = torch.cat([x_target_all, pe_target], dim=-1)
        else:
            x_context_enhanced = x_context_all
            x_target_enhanced = x_target_all

        # ============================================
        # PAPER-FAITHFUL: Separate Key and Value
        # ============================================
        
        # Value: MLP(x, y)
        xy_context = torch.cat([x_context_enhanced, y_context], dim=-1)
        r_value = self.mlp_value(xy_context)
        
        # Key: MLP(x) - NO y!
        r_key = self.mlp_key(x_context_enhanced)
        
        # Query: MLP(x*)
        r_query = self.mlp_query(x_target_enhanced)
        
        if self.use_layer_norm:
            r_value = self.value_norm(r_value)
            r_key = self.key_norm(r_key)
            r_query = self.query_norm(r_query)

        # Find k nearest neighbors
        initial_k = min(self.k, N)
        nearest_indices, final_k = find_k_nearest_context_points(
            s_target, s_context, initial_k, 
            use_adaptive=self.use_adaptive_knn,
            adaptive_knn=self.adaptive_knn
        )

        # Gather keys and values for local attention
        batch_idx = torch.arange(B, device=r_key.device).view(B, 1, 1).expand(B, M, final_k)
        gathered_keys = r_key[batch_idx, nearest_indices]    # (B, M, k, D)
        gathered_values = r_value[batch_idx, nearest_indices] # (B, M, k, D)

        # Reshape for batched attention
        r_query_reshaped = r_query.view(B * M, 1, D)
        gathered_keys_reshaped = gathered_keys.view(B * M, final_k, D)
        gathered_values_reshaped = gathered_values.view(B * M, final_k, D)

        # Local cross attention with SEPARATE key and value
        attn_output_reshaped = self.cross_attention(
            r_query_reshaped, gathered_keys_reshaped, gathered_values_reshaped
        )

        attn_output = attn_output_reshaped.view(B, M, D)
        if self.use_layer_norm:
            attn_output = self.output_norm(attn_output)

        return attn_output


class LocalizedVarianceEncoder(nn.Module):
    """Localized variance encoder with k-NN attention (no y involved)."""
    def __init__(self, input_dim, hidden_size, v_dim, num_heads, dropout_rate, k, 
                 num_layers_mlp=1, use_adaptive_knn=False, adaptive_knn_params=None,
                 use_positional_encoding=False, pe_dim=64, pe_type='sinusoidal', 
                 pe_scales=None, pe_scale_dims=None, pe_coordinate_range='normalized', 
                 pe_original_bounds=None, pe_learnable=False,
                 use_layer_norm=True, **kwargs):
        super().__init__()
        self.k = k
        self.v_dim = v_dim
        self.use_adaptive_knn = use_adaptive_knn
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        if use_adaptive_knn and adaptive_knn_params:
            self.adaptive_knn = AdaptiveKNN(**adaptive_knn_params)
        else:
            self.adaptive_knn = None
        
        # Initialize positional encoding
        if use_positional_encoding:
            if pe_type == 'multi_scale':
                if pe_scale_dims is None:
                    num_scales = len(pe_scales) if pe_scales else 3
                    base_dim = pe_dim // num_scales
                    remainder = pe_dim % num_scales
                    pe_scale_dims = [base_dim] * num_scales
                    if remainder > 0:
                        pe_scale_dims[-1] += remainder
                
                self.pos_encoder = MultiScalePositionalEncoding(
                    spatial_dim=2,
                    encoding_dims=pe_scale_dims,
                    scales=pe_scales,
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds,
                    learnable=pe_learnable
                )
                actual_pe_dim = sum(pe_scale_dims)
            else:
                self.pos_encoder = SpatialPositionalEncoding(
                    spatial_dim=2,
                    encoding_dim=pe_dim,
                    learnable=pe_learnable,
                    use_gaussian=(pe_type == 'gaussian'),
                    coordinate_range=pe_coordinate_range,
                    original_bounds=pe_original_bounds
                )
                actual_pe_dim = pe_dim
            
            actual_input_dim = input_dim + actual_pe_dim
        else:
            actual_input_dim = input_dim
        
        # v = MLP(x) for key/value (no y in variance encoder)
        self.mlp_kv = _create_mlp(actual_input_dim, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.mlp_query = _create_mlp(actual_input_dim, v_dim, hidden_size, num_layers_mlp, dropout_rate)
        self.cross_attention = MultiHeadCrossAttentionWrapper(v_dim, num_heads, dropout_rate)
        
        if use_layer_norm:
            self.kv_norm = nn.LayerNorm(v_dim)
            self.query_norm = nn.LayerNorm(v_dim)
            self.output_norm = nn.LayerNorm(v_dim)
    
    def forward(self, x_context_all, x_target_all, s_context, s_target, **kwargs):
        B, M, _ = x_target_all.shape
        N = x_context_all.shape[1]
        D = self.v_dim

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            pe_context = self.pos_encoder(s_context)
            pe_target = self.pos_encoder(s_target)
            x_context_enhanced = torch.cat([x_context_all, pe_context], dim=-1)
            x_target_enhanced = torch.cat([x_target_all, pe_target], dim=-1)
        else:
            x_context_enhanced = x_context_all
            x_target_enhanced = x_target_all

        v_kv = self.mlp_kv(x_context_enhanced)
        v_query = self.mlp_query(x_target_enhanced)
        
        if self.use_layer_norm:
            v_kv = self.kv_norm(v_kv)
            v_query = self.query_norm(v_query)

        initial_k = min(self.k, N)
        nearest_indices, final_k = find_k_nearest_context_points(
            s_target, s_context, initial_k,
            use_adaptive=self.use_adaptive_knn,
            adaptive_knn=self.adaptive_knn
        )

        batch_idx = torch.arange(B, device=v_kv.device).view(B, 1, 1).expand(B, M, final_k)
        gathered_kv = v_kv[batch_idx, nearest_indices]

        v_query_reshaped = v_query.view(B * M, 1, D)
        gathered_kv_reshaped = gathered_kv.view(B * M, final_k, D)

        attn_output_reshaped = self.cross_attention(
            v_query_reshaped, gathered_kv_reshaped, gathered_kv_reshaped
        )

        attn_output = attn_output_reshaped.view(B, M, D)
        if self.use_layer_norm:
            attn_output = self.output_norm(attn_output)

        return attn_output


# ============================ 
# HYBRID GRAPH DECODERS
# ============================ 
class HybridGraphMeanDecoder(nn.Module):
    """
    A hybrid decoder for mean predictions that blends local point-wise predictions 
    with graph-based refinements.
    """
    def __init__(self, local_decoder: nn.Module, gat_hidden_channels: int, gat_heads: int,
                 decoder_graph_k: int, min_points_for_graph: int, initial_alpha_bias: float,
                 dropout_rate: float = 0.1, use_adaptive_knn: bool = False, adaptive_knn_params=None):
        super().__init__()
        self.local_decoder = local_decoder
        self.decoder_graph_k = decoder_graph_k
        self.min_points_for_graph = min_points_for_graph
        self.use_adaptive_knn = use_adaptive_knn
        if use_adaptive_knn and adaptive_knn_params:
            self.adaptive_knn = AdaptiveKNN(**adaptive_knn_params)
        else:
            self.adaptive_knn = None
        
        self.gat1 = GATConv(1, gat_hidden_channels, heads=gat_heads, dropout=dropout_rate)
        self.gat2 = GATConv(gat_hidden_channels * gat_heads, 1, heads=1, concat=False, dropout=dropout_rate)
        self.alpha_logit = nn.Parameter(torch.tensor(initial_alpha_bias, dtype=torch.float32))

    def forward(self, r_star, w_star, s_target, x_attr_target):
        # 1. Get Base Local Prediction
        local_pred = self.local_decoder(r_star, w_star, s_target, x_attr_target)
        
        B, M, _ = s_target.shape

        # If too few points, skip graph refinement
        if M < self.min_points_for_graph:
            return local_pred

        batch_vector = torch.arange(B, device=s_target.device).repeat_interleave(M)
        s_target_flat = s_target.view(-1, s_target.shape[-1])
        
        # Use adaptive k if enabled
        k = self.decoder_graph_k
        if self.use_adaptive_knn and self.adaptive_knn:
            k = self.adaptive_knn.compute_adaptive_k(s_target, k)
        k = min(k, M - 1)
        
        # Build graph on target points to smooth predictions
        edge_index = knn_graph(s_target_flat, k=k, batch=batch_vector)
        
        local_pred_flat = local_pred.view(-1, 1)
        x = F.elu(self.gat1(local_pred_flat, edge_index))
        graph_pred_flat = self.gat2(x, edge_index)
        graph_pred = graph_pred_flat.view(B, M, 1)

        # Learnable weighting between local and graph prediction
        alpha = torch.sigmoid(self.alpha_logit)
        return (1 - alpha) * local_pred + alpha * graph_pred


class HybridGraphVarianceDecoder(nn.Module):
    """
    A hybrid decoder for variance predictions that performs refinement in stable log-space.
    """
    def __init__(self, local_decoder: nn.Module, gat_hidden_channels: int, gat_heads: int,
                 decoder_graph_k: int, min_points_for_graph: int, initial_alpha_bias: float,
                 dropout_rate: float = 0.1, use_adaptive_knn: bool = False, adaptive_knn_params=None):
        super().__init__()
        self.local_decoder = local_decoder
        self.decoder_graph_k = decoder_graph_k
        self.min_points_for_graph = min_points_for_graph
        self.use_adaptive_knn = use_adaptive_knn
        if use_adaptive_knn and adaptive_knn_params:
            self.adaptive_knn = AdaptiveKNN(**adaptive_knn_params)
        else:
            self.adaptive_knn = None
        
        self.gat1 = GATConv(1, gat_hidden_channels, heads=gat_heads, dropout=dropout_rate)
        self.gat2 = GATConv(gat_hidden_channels * gat_heads, 1, heads=1, concat=False, dropout=dropout_rate)
        self.alpha_logit = nn.Parameter(torch.tensor(initial_alpha_bias, dtype=torch.float32))

    def forward(self, v_star, s_target, x_attr_target):
        local_pred = self.local_decoder(v_star, s_target, x_attr_target)
        
        B, M, _ = s_target.shape
        
        if M < self.min_points_for_graph:
            return local_pred

        # Work in log-space for numerical stability with variance
        log_local_pred = torch.log(local_pred.clamp(min=1e-6))
        
        batch_vector = torch.arange(B, device=s_target.device).repeat_interleave(M)
        s_target_flat = s_target.view(-1, s_target.shape[-1])
        
        k = self.decoder_graph_k
        if self.use_adaptive_knn and self.adaptive_knn:
            k = self.adaptive_knn.compute_adaptive_k(s_target, k)
        k = min(k, M - 1)
        
        edge_index = knn_graph(s_target_flat, k=k, batch=batch_vector)

        log_local_pred_flat = log_local_pred.view(-1, 1)
        x = F.elu(self.gat1(log_local_pred_flat, edge_index))
        log_graph_pred_flat = self.gat2(x, edge_index)
        
        # Convert back from log-space
        graph_pred = torch.exp(log_graph_pred_flat.view(B, M, 1))

        alpha = torch.sigmoid(self.alpha_logit)
        return (1 - alpha) * local_pred + alpha * graph_pred


# ============================ 
# MAIN SMACNP MODEL (FULL IMPLEMENTATION)
# ============================ 
class SMACNP_ALL(nn.Module):
    def __init__(self, input_dim, spatial_dim, attr_dim, y_dim,
                 r_dim, w_dim, v_dim, hidden_size,
                 num_heads, dropout_rate, spatial_indices,
                 num_layers_mlp=1,
                 laplace_distance_p=1.0,
                 
                 # Encoder Controls
                 use_localized_attention=False, k_localized_encoder=32,

                 # Layer Normalization Control
                 use_layer_norm=True,
                 
                 # Positional Encoding Controls (Complete)
                 use_positional_encoding=False,
                 pe_dim=64,
                 pe_type='sinusoidal',
                 pe_scales=None,
                 pe_scale_dims=None,
                 pe_coordinate_range='normalized',
                 pe_original_bounds=None,
                 pe_learnable=False,
                 
                 # Adaptive KNN Controls (Complete)
                 use_adaptive_knn=False,
                 adaptive_min_k=5,
                 adaptive_max_k=50,
                 adaptive_density_percentile=0.25,
                 adaptive_use_median=True,
                 
                 # Decoder Controls
                 use_hybrid_decoder=False, k_hybrid_decoder=10,
                 gat_hidden_channels=16, gat_heads=4,
                 min_points_for_graph=10, initial_alpha_bias=-1.0,
                 **kwargs):
        
        super().__init__()
        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        self.attr_dim = attr_dim
        self.y_dim = y_dim
        self.spatial_indices = spatial_indices
        self.laplace_distance_p = laplace_distance_p
        
        # Store configuration flags
        self.use_hybrid_decoder = use_hybrid_decoder
        self.use_adaptive_knn = use_adaptive_knn
        self.use_layer_norm = use_layer_norm
        self.use_positional_encoding = use_positional_encoding

        # Track encoder style to handle spatial/attribute separation strictly
        self.uses_global_attention = not use_localized_attention
        
        if input_dim != spatial_dim + attr_dim:
            raise ValueError(
                f"input_dim ({input_dim}) must equal spatial_dim + attr_dim ({spatial_dim + attr_dim})"
            )

        # Decoder always consumes attribute-only tensors
        encoder_attr_dim = attr_dim

        # Determine input dims for specific encoders
        # Global encoder only sees attributes (Paper Faithful)
        # Graph/Local encoders might need spatial info for node features
        if self.uses_global_attention:
            encoder_input_dim = attr_dim
        else:
            encoder_input_dim = spatial_dim + attr_dim
        
        # Prepare adaptive KNN parameters
        adaptive_knn_params = {
            'min_k': adaptive_min_k,
            'max_k': adaptive_max_k,
            'density_percentile': adaptive_density_percentile,
            'use_median': adaptive_use_median
        } if use_adaptive_knn else None
        
        # Initialize adaptive KNN for graph construction if needed
        if use_adaptive_knn:
            self.adaptive_knn_encoder = AdaptiveKNN(**adaptive_knn_params)
        else:
            self.adaptive_knn_encoder = None

        # --- 1. Mean-Location Encoder (Laplace Attention) ---
        # "w = MLP(s, y)" -> Always same structure
        self.mean_loc_encoder = MeanLocationEncoder(
            spatial_dim, y_dim, hidden_size, w_dim, dropout_rate, num_layers_mlp,
            laplace_p=self.laplace_distance_p
        )
        
        # --- 2. Mean-Attribute & Variance Encoders ---
        encoder_kwargs = {
            'dropout_rate': dropout_rate,
            'num_layers_mlp': num_layers_mlp,
            'use_positional_encoding': use_positional_encoding,
            'pe_dim': pe_dim,
            'pe_type': pe_type,
            'pe_scales': pe_scales,
            'pe_scale_dims': pe_scale_dims,
            'pe_coordinate_range': pe_coordinate_range,
            'pe_original_bounds': pe_original_bounds,
            'pe_learnable': pe_learnable,
            'use_adaptive_knn': use_adaptive_knn,
            'adaptive_knn_params': adaptive_knn_params,
        }
        
        if use_localized_attention:
            print("Using Localized Attention encoder (Paper Faithful Keys)")
            self.mean_attr_encoder = LocalizedMeanAttributeEncoder(
                encoder_input_dim, y_dim, hidden_size, r_dim, num_heads,
                k=k_localized_encoder, use_layer_norm=use_layer_norm, **encoder_kwargs
            )
            # self.variance_encoder = LocalizedVarianceEncoder(
            #     encoder_input_dim, hidden_size, v_dim, num_heads,
            #     k=k_localized_encoder, use_layer_norm=use_layer_norm, **encoder_kwargs
            # )
        else: # Global Attention (Default)
            print("Using Global Attention encoder (Paper Faithful Keys: Key=Attr, Value=Attr+Y)")
            self.mean_attr_encoder = GlobalMeanAttributeEncoder(
                encoder_input_dim, y_dim, hidden_size, r_dim, num_heads,
                use_layer_norm=use_layer_norm, **encoder_kwargs
            )
            # self.variance_encoder = GlobalVarianceEncoder(
            #     encoder_input_dim, hidden_size, v_dim, num_heads,
            #     use_layer_norm=use_layer_norm, spatial_dim=spatial_dim, **encoder_kwargs
            # )
        
        # --- 3. Decoders ---
        # Base local decoders
        local_temp_mean_decoder = MeanDecoder(r_dim, w_dim, spatial_dim, encoder_attr_dim, 
                                              hidden_size, y_dim, dropout_rate, num_layers_mlp)
        # local_temp_variance_decoder = VarianceDecoder(v_dim, spatial_dim, encoder_attr_dim, 
        #                                               hidden_size, 1, dropout_rate, num_layers_mlp)

        if use_hybrid_decoder:
            print("Using Hybrid decoder with Graph Refinement")
            hybrid_kwargs = {
                'decoder_graph_k': k_hybrid_decoder,
                'gat_hidden_channels': gat_hidden_channels,
                'gat_heads': gat_heads,
                'min_points_for_graph': min_points_for_graph,
                'initial_alpha_bias': initial_alpha_bias,
                'dropout_rate': dropout_rate,
                'use_adaptive_knn': use_adaptive_knn,
                'adaptive_knn_params': adaptive_knn_params
            }
            self.temp_mean_decoder = HybridGraphMeanDecoder(local_temp_mean_decoder, **hybrid_kwargs)
            # self.temp_variance_decoder = HybridGraphVarianceDecoder(local_temp_variance_decoder, **hybrid_kwargs)
        else:
            self.temp_mean_decoder = local_temp_mean_decoder
            # self.temp_variance_decoder = local_temp_variance_decoder

    def forward(self, x_context_all, y_context, x_target_all):
        """
        Forward pass with robust handling of spatial vs attribute data.
        """
        # 1. Separate Spatial (s) and Attributes (x_attr)
        # This is critical for Paper-Faithful implementation
        s_context, x_attr_context = separate_spatial_attributes(x_context_all, self.spatial_indices)
        s_target, x_attr_target = separate_spatial_attributes(x_target_all, self.spatial_indices)

        # 3. Prepare Inputs for Encoders
        if self.uses_global_attention:
            # Global Encoder (Paper Faithful):
            # Input is purely attributes. The encoder will receive 'y' separately.
            x_context_input = x_attr_context
            x_target_input = x_attr_target
        else:
            # GCN/GAT/Local Encoders:
            # Often benefit from seeing coordinates concatenated with attributes
            # to learn local relations, even if graph structure is passed separately.
            x_context_input = torch.cat([s_context, x_attr_context], dim=-1)
            x_target_input = torch.cat([s_target, x_attr_target], dim=-1)
        
        # 5. Execute Encoders
        
        # A. Mean-Location Encoder (Laplace: s, y)
        w_star = self.mean_loc_encoder(s_context, y_context, s_target)
        
        # B. Mean-Attribute Encoder (Attention: x -> r)
        # Note: We pass all possible args; specific encoders pick what they need
        encoder_kwargs = {
            'x_context_all': x_context_input,
            'y_context': y_context,
            'x_target_all': x_target_input,
            's_context': s_context,
            's_target': s_target,
        }
        
        r_star = self.mean_attr_encoder(**encoder_kwargs)
        
        # C. Variance Encoder (Attention: s+x -> v)
        variance_kwargs = {
            'x_context_all': x_context_input, # Or attr only, depending on class
            'x_target_all': x_target_input,
            's_context': s_context,
            's_target': s_target,
        }
        
        # v_star = self.variance_encoder(**variance_kwargs)

        # 6. Execute Decoders
        # Decoders take: r*, w*, s_target, and target_attributes
        temp_mu = self.temp_mean_decoder(r_star, w_star, s_target, x_attr_target)
        # temp_variance = self.temp_variance_decoder(v_star, s_target, x_attr_target)

        return temp_mu

if __name__ == "__main__":
    # GPU smoke test for SMACNP_ALL
    device = torch.device("cuda:1")

    B, N_ctx, N_tgt = 2, 10000, 1000
    spatial_dim = 2
    attr_dim = 3
    input_dim = spatial_dim + attr_dim
    y_dim = 1
    spatial_indices = [0, 1]

    model = SMACNP_ALL(
        input_dim=input_dim,
        spatial_dim=spatial_dim,
        attr_dim=attr_dim,
        y_dim=y_dim,
        r_dim=128, # 属性
        w_dim=64, # 坐标
        v_dim=256, # 方差的，我取消了
        hidden_size=512, # MLP 的维度
        num_heads=16,
        dropout_rate=0.1,
        spatial_indices=spatial_indices,
        use_hybrid_decoder=False,
        use_adaptive_knn=False,
        use_positional_encoding=False
    ).to(device)
    model.eval()

    x_context_all = torch.randn(B, N_ctx, input_dim, device=device)
    y_context = torch.randn(B, N_ctx, y_dim, device=device)
    x_target_all = torch.randn(B, N_tgt, input_dim, device=device)

    with torch.no_grad():
        out = model(x_context_all, y_context, x_target_all)

    print("SMACNP_ALL GPU smoke test OK")
    print("temp_mu shape:", tuple(out["solar_mu"].shape))
    # print("temp_variance shape:", tuple(out["temp_variance"].shape))
    


    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量:       {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")
    print(f"FP32 模型大小:  {total_params * 4 / 1024**2:.2f} MiB")