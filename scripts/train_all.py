import torch
import time
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import random
import torch.nn as nn
import os
import argparse
import yaml

from smacnp import SMACNP_ALL
from dataset import SMACNPDataset_ALL

from utils import EarlyStopping, set_seed, get_smacnp_filename_suffix_parts, Solor_loss_function

def gamma_nll_loss(mu, variance, target):
    """
    Gamma NLL for mean/variance parameterization (AMP-safe).
    All inputs should be float32.
    """
    EPS = 1e-4  # fp16-safe floor
    target = target.clamp(min=EPS)
    variance = variance.clamp(min=EPS)
    mu = mu.clamp(min=EPS)

    shape = (mu.pow(2) / variance).clamp(min=1e-4, max=1e6)
    rate  = (mu / variance).clamp(min=1e-4, max=1e6)

    log_prob = (shape * torch.log(rate)
                - torch.lgamma(shape)
                + (shape - 1) * torch.log(target)
                - rate * target)
    return -log_prob


def combined_loss_function(predictions, y_target, loss_weights, precip_distribution, scalers, pos_weight_cap=20.0):
    # Ensure scalers live on the same device as targets/predictions (robust to CPU/GPU placement).
    device = y_target.device
    scalers_dev = {k: v.to(device, non_blocking=True) for k, v in scalers.items()}

    # Entire loss in float32 (outside autocast)
    temp_mu = predictions['temp_mu'].float()
    temp_variance = predictions['temp_variance'].float()
    precip_prob_logits = predictions['precip_prob_logits'].float()
    precip_mu_norm = predictions['precip_mu'].float()
    precip_variance_norm = predictions['precip_variance'].float()
    y_target = y_target.float()

    EPS_VAR = 1e-4

    # Temperature Gaussian NLL (targets are already normalized)
    temp_variance = temp_variance.clamp(min=EPS_VAR)
    y_temp_norm = y_target[..., 0].unsqueeze(-1)
    temp_loss = nn.GaussianNLLLoss(reduction='none')(temp_mu, y_temp_norm, temp_variance).mean()

    # Rain/no-rain classification
    y_precip_norm = y_target[..., 1].unsqueeze(-1)
    is_rainy = (y_precip_norm > 1e-6).float()
    num_pos = is_rainy.sum()
    num_neg = is_rainy.numel() - num_pos
    pos_weight = (num_neg / num_pos.clamp(min=1)).clamp(max=pos_weight_cap)
    prob_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)(precip_prob_logits, is_rainy).mean()

    # Amount loss on rainy samples only
    rainy_mask = is_rainy.bool().squeeze(-1)
    amount_loss = torch.tensor(0.0, device=y_target.device)
    if rainy_mask.any():
        if precip_distribution == 'lognormal':
            # Work in normalized transform space; use Gaussian NLL with fp16-safe eps
            pv = precip_variance_norm[rainy_mask].clamp(min=EPS_VAR)
            amount_loss_unreduced = nn.GaussianNLLLoss(reduction='none')(
                precip_mu_norm[rainy_mask], y_precip_norm[rainy_mask], pv
            )
            amount_loss = amount_loss_unreduced.mean()

        elif precip_distribution == 'gamma':
            # MinMax inverse: x = (x_norm - min_) / scale_
            mu_precip_denorm = (precip_mu_norm[rainy_mask] - scalers_dev['precip_min']) / scalers_dev['precip_scale']
            y_precip_denorm  = (y_precip_norm[rainy_mask]  - scalers_dev['precip_min']) / scalers_dev['precip_scale']
            var_precip_denorm = precip_variance_norm[rainy_mask] / (scalers_dev['precip_scale'] ** 2)

            amount_loss_unreduced = gamma_nll_loss(mu_precip_denorm, var_precip_denorm, y_precip_denorm)
            amount_loss = amount_loss_unreduced.mean()
        else:
            raise ValueError(f"Unknown precip_distribution: {precip_distribution}")

    return (loss_weights['temp'] * temp_loss +
            loss_weights['precip_prob'] * prob_loss +
            loss_weights['precip_amount'] * amount_loss)


def train_epoch(model, train_loader, optimizer, scaler, clip_grad_norm,
                device, loss_weights, precip_distribution, scalers, use_amp):
    """
    Train epoch with support for graph models.
    """
    model.train()
    train_loss = 0.0
    processed_batches = 0

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    nb = (device.type == "cuda")  # non_blocking only matters for CUDA

    for i, batch in enumerate(train_loader):
        # --- START MODIFICATION ---
        # Convert dict batch to object-like batch for non-graph models
        if isinstance(batch, dict):
            from types import SimpleNamespace
            batch = SimpleNamespace(
                x=batch['x_context_all'],
                y=batch['y_context'],
                x_target_all=batch['x_target_all'],
                y_target=batch['y_target'],
                static_grid=batch.get('static_grid'),
                batch=None # No batch index for standard dataloader
            )
        # --- END MODIFICATION ---

        # --- MODIFIED: Handle graph data ---
        if batch.x.nelement() == 0: # Assuming x is the primary context feature
            continue

        x_context_all_dev = batch.x.to(device, non_blocking=nb)
        y_context_dev = batch.y.to(device, non_blocking=nb)
        x_target_all_dev = batch.x_target_all.to(device, non_blocking=nb)
        y_target_dev = batch.y_target.to(device, non_blocking=nb)

        # Prepare model-specific keyword arguments
        kwargs = {}
        # --- END MODIFIED SECTION ---

        optimizer.zero_grad(set_to_none=True)

        try:
            # Forward in AMP
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                predictions = model(x_context_all_dev, y_context_dev, x_target_all_dev)


            # Loss in fp32 (outside autocast)
            loss = combined_loss_function(predictions, y_target_dev, loss_weights, precip_distribution, scalers).float()

            if not torch.isfinite(loss):
                print("Warning: NaN or Inf loss detected in training. Skipping update.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            processed_batches += 1

        except Exception as e:
            import traceback
            print(f"Error during training batch {i}: {e}")
            traceback.print_exc()
            continue

    return 0.0 if processed_batches == 0 else (train_loss / processed_batches)


def validate_epoch(model, val_loader, device, scalers, loss_weights, precip_distribution, use_amp):
    """
    Validation epoch with support for graph models.
    """
    model.eval()
    val_loss = 0.0
    temp_mae_sum = temp_rmse_sum = 0.0
    precip_mae_sum = precip_rmse_sum = 0.0
    temp_samples = precip_samples = 0
    processed_batches = 0

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    nb = (device.type == "cuda")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # --- START MODIFICATION ---
            # Convert dict batch to object-like batch for non-graph models
            if isinstance(batch, dict):
                from types import SimpleNamespace
                batch = SimpleNamespace(
                    x=batch['x_context_all'],
                    y=batch['y_context'],
                    x_target_all=batch['x_target_all'],
                    y_target=batch['y_target'],
                    static_grid=batch.get('static_grid'),
                    batch=None # No batch index for standard dataloader
                )
            # --- END MODIFICATION ---

            # --- MODIFIED: Handle graph data ---
            if batch.x.nelement() == 0:
                continue

            x_context_all = batch.x.to(device, non_blocking=nb)
            y_context = batch.y.to(device, non_blocking=nb)
            x_target_all = batch.x_target_all.to(device, non_blocking=nb)
            y_target = batch.y_target.to(device, non_blocking=nb)

            # --- END MODIFIED SECTION ---

            try:
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    predictions = model(x_context_all, y_context, x_target_all)

                # Loss in fp32
                loss = combined_loss_function(predictions, y_target, loss_weights, precip_distribution, scalers).float()
                if not torch.isfinite(loss):
                    print("Warning: NaN or Inf loss detected in validation. Skipping metrics.")
                    continue

                val_loss += loss.item()
                processed_batches += 1

                # Metrics calculation
                device = y_target.device
                scalers_dev = {k: v.to(device, non_blocking=True) for k, v in scalers.items()}

                y_temp_norm = y_target[..., 0].float()
                mu_temp_norm = predictions['temp_mu'].squeeze(-1).float()
                mu_temp_denorm = (mu_temp_norm - scalers_dev['temp_min']) / scalers_dev['temp_scale']
                y_temp_denorm  = (y_temp_norm  - scalers_dev['temp_min']) / scalers_dev['temp_scale']

                temp_mae_sum += torch.abs(mu_temp_denorm - y_temp_denorm).sum().item()
                temp_rmse_sum += ((mu_temp_denorm - y_temp_denorm) ** 2).sum().item()
                temp_samples  += y_temp_norm.numel()

                y_precip_norm = y_target[..., 1].float()
                prob_precip = torch.sigmoid(predictions['precip_prob_logits'].squeeze(-1).float())
                mu_precip_norm = predictions['precip_mu'].squeeze(-1).float()

                if precip_distribution == 'lognormal':
                    mu_precip_denorm_trans = (mu_precip_norm - scalers_dev['precip_min']) / scalers_dev['precip_scale']
                    predicted_precip_amount_if_rain = torch.expm1(mu_precip_denorm_trans)
                    y_precip_denorm_trans = (y_precip_norm - scalers_dev['precip_min']) / scalers_dev['precip_scale']
                    actual_precip_amount = torch.expm1(y_precip_denorm_trans)
                elif precip_distribution == 'gamma':
                    predicted_precip_amount_if_rain = (mu_precip_norm - scalers_dev['precip_min']) / scalers_dev['precip_scale']
                    actual_precip_amount = (y_precip_norm  - scalers_dev['precip_min']) / scalers_dev['precip_scale']
                else:
                    raise ValueError(f"Unknown precip_distribution: {precip_distribution}")

                predicted_precip_amount = prob_precip * predicted_precip_amount_if_rain.clamp(min=0)
                precip_mae_sum  += torch.abs(predicted_precip_amount - actual_precip_amount).sum().item()
                precip_rmse_sum += ((predicted_precip_amount - actual_precip_amount) ** 2).sum().item()
                precip_samples  += y_precip_norm.numel()

            except Exception as e:
                import traceback
                print(f"Error during validation batch {i}: {e}")
                traceback.print_exc()
                continue

    if processed_batches == 0:
        return float('inf'), (float('inf'), float('inf')), (float('inf'), float('inf'))

    avg_val_loss = val_loss / processed_batches
    temp_metrics = (temp_mae_sum / temp_samples, np.sqrt(temp_rmse_sum / temp_samples))
    precip_metrics = (precip_mae_sum / precip_samples, np.sqrt(precip_rmse_sum / precip_samples))
    return avg_val_loss, temp_metrics, precip_metrics


def get_positional_encoding_params(cfg):
    """Extracts positional encoding parameters, supporting both legacy and flat configs."""
    if cfg is None:
        return {'use_positional_encoding': False}

    # Accept either a nested "positional_encoding" block or flat keys (as in base_config).
    if 'positional_encoding' in cfg:
        pe_cfg = cfg.get('positional_encoding', {})
        use_pe = cfg.get('use_positional_encoding', pe_cfg.get('use_positional_encoding', False))
    else:
        pe_cfg = cfg
        use_pe = pe_cfg.get('use_positional_encoding', cfg.get('use_positional_encoding', False))

    if not use_pe:
        return {'use_positional_encoding': False}

    def _get(keys, default=None):
        for key in keys:
            if key in pe_cfg:
                return pe_cfg[key]
        return default

    params = {
        'use_positional_encoding': True,
        'pe_dim': _get(['pe_dim', 'dim'], 96),
        'pe_type': _get(['pe_type', 'type'], 'sinusoidal'),
        'pe_coordinate_range': _get(['pe_coordinate_range', 'coordinate_range'], 'normalized'),
        'pe_original_bounds': _get(['pe_original_bounds', 'original_bounds'], None),
        'pe_learnable': _get(['pe_learnable', 'learnable', 'learnable_frequencies'], False)
    }

    pe_type = params['pe_type']
    if pe_type == 'sinusoidal':
        params.update({
            'max_wavelength': _get(['max_wavelength'], 10.0),
            'learnable_frequencies': _get(['learnable_frequencies', 'pe_learnable'], True),
        })
    elif pe_type == 'gaussian':
        params.update({
            'num_centers': _get(['num_centers'], 48),
        })
    elif pe_type == 'multi_scale':
        params.update({
            'pe_scales': _get(['scales', 'pe_scales'], [0.1, 0.5, 1.0]),
            'pe_scale_dims': _get(['scale_dims', 'pe_scale_dims'], None),
            'learnable_scale_weights': _get(['learnable_scale_weights'], True),
        })
        scale_dims = params['pe_scale_dims']
        if scale_dims:
            params['pe_dim'] = sum(scale_dims)

    return params


def _merge_config_dicts(base_cfg: dict | None, override_cfg: dict | None) -> dict:
    """
    Shallow-merge two config dicts, while also merging a nested `positional_encoding` block.

    This is used to make PE settings robust to being specified either at the top-level
    (e.g. `model.smacnp.use_positional_encoding`) or inside encoder-specific blocks
    (e.g. `model.smacnp.encoder.localized_params.use_positional_encoding`).
    """
    merged: dict = dict(base_cfg or {})
    override_cfg = override_cfg or {}
    merged.update(override_cfg)

    base_pe = (base_cfg or {}).get('positional_encoding')
    override_pe = override_cfg.get('positional_encoding')
    if isinstance(base_pe, dict) or isinstance(override_pe, dict):
        pe_merged: dict = {}
        if isinstance(base_pe, dict):
            pe_merged.update(base_pe)
        if isinstance(override_pe, dict):
            pe_merged.update(override_pe)
        merged['positional_encoding'] = pe_merged

    return merged


def get_smacnp_model_from_config(cfg):
    """Initializes an SMACNP_ALL model from a configuration dictionary."""
    smacnp_cfg = cfg['smacnp']

    # Handle bottleneck_size as a default for r_dim, w_dim, v_dim
    bottleneck_size = smacnp_cfg.get('bottleneck_size')
    r_dim = smacnp_cfg.get('r_dim', bottleneck_size)
    w_dim = smacnp_cfg.get('w_dim', bottleneck_size)
    v_dim = smacnp_cfg.get('v_dim', bottleneck_size)

    if not all([r_dim, w_dim, v_dim]):
        raise ValueError("Config must specify either bottleneck_size or all of r_dim, w_dim, and v_dim.")

    # Defaults shared by all encoder styles
    encoder_cfg = smacnp_cfg.get('encoder', {})
    localized_params = encoder_cfg.get('localized_params', {})

    k_localized_encoder = localized_params.get('k', 32)

    configuration = smacnp_cfg.get('configuration')
    encoder_type = (configuration or encoder_cfg.get('type', 'global')).lower()
    pe_source_cfg = None

    if encoder_type == 'localized':
        localized_cfg = smacnp_cfg.get('localized_cnp') or localized_params
        k_localized_encoder = localized_cfg.get('k', k_localized_encoder)
        pe_source_cfg = localized_cfg
        use_localized_attention = True
    else:
        # Default to global attention (global block is optional)
        global_cfg = smacnp_cfg.get('global_cnp') or {}
        pe_source_cfg = global_cfg if global_cfg else smacnp_cfg
        use_localized_attention = False
        encoder_type = 'global'

    # Decoder settings
    decoder_cfg = smacnp_cfg.get('decoder', {})
    use_hybrid_decoder = decoder_cfg.get('use_hybrid', False)

    hybrid_params = decoder_cfg.get('hybrid_params', {})
    k_hybrid_decoder = hybrid_params.get('k', 15)
    gat_hidden_channels = hybrid_params.get('gat_hidden_channels', 32)
    gat_heads = hybrid_params.get('gat_heads', 4)
    min_points_for_graph = hybrid_params.get('min_points_for_graph', 10)
    initial_alpha_bias = hybrid_params.get('initial_alpha_bias', -0.5)

    # Layer Normalization
    use_layer_norm = smacnp_cfg.get('use_layer_norm', False)


    # Adaptive KNN settings
    use_adaptive_knn = smacnp_cfg.get('use_adaptive_knn', False)
    adaptive_knn_cfg = smacnp_cfg.get('adaptive_knn', {})
    adaptive_min_k = adaptive_knn_cfg.get('min_k', 5)
    adaptive_max_k = adaptive_knn_cfg.get('max_k', 50)
    adaptive_density_percentile = adaptive_knn_cfg.get('density_percentile', 0.25)
    adaptive_use_median = adaptive_knn_cfg.get('use_median', True)

    # --- Fully specified Positional Encoding parameters ---
    pe_kwargs = get_positional_encoding_params(_merge_config_dicts(smacnp_cfg, pe_source_cfg))
    use_positional_encoding = pe_kwargs.get('use_positional_encoding', False)

    # Log configuration for debugging
    print("SMACNP Configuration Summary:")
    print(f"  Encoder: {configuration or encoder_type}")
    if use_localized_attention:
        print(f"    Localized k: {k_localized_encoder}")
    print(f"  Layer Norm: {use_layer_norm}")
    print(f"  Positional Encoding: {use_positional_encoding}", end="")
    if use_positional_encoding:
        print(f" (type: {pe_kwargs['pe_type']}, dim: {pe_kwargs.get('pe_dim', 'N/A')})")
    else:
        print()
    print(f"  Adaptive KNN: {use_adaptive_knn}", end="")
    if use_adaptive_knn:
        print(f" (range: {adaptive_min_k}-{adaptive_max_k}, percentile: {adaptive_density_percentile}, use_median: {adaptive_use_median})")
    else:
        print()
    print(f"  Hybrid Decoder: {use_hybrid_decoder}")

    model = SMACNP_ALL(
        input_dim=smacnp_cfg['input_dim'],
        spatial_dim=smacnp_cfg['spatial_dim'],
        attr_dim=smacnp_cfg['attr_dim'],
        y_dim=smacnp_cfg['y_dim'],
        r_dim=r_dim,
        w_dim=w_dim,
        v_dim=v_dim,
        hidden_size=smacnp_cfg['hidden_size'],
        num_heads=smacnp_cfg['num_heads'],
        dropout_rate=smacnp_cfg['dropout_rate'],
        spatial_indices=smacnp_cfg['spatial_indices'],
        num_layers_mlp=smacnp_cfg.get('num_layers_mlp', 1),

        # Encoder params
        use_localized_attention=use_localized_attention,
        k_localized_encoder=k_localized_encoder,

        # Layer Normalization
        use_layer_norm=use_layer_norm,

        # Adaptive KNN
        use_adaptive_knn=use_adaptive_knn,
        adaptive_min_k=adaptive_min_k,
        adaptive_max_k=adaptive_max_k,
        adaptive_density_percentile=adaptive_density_percentile,
        adaptive_use_median=adaptive_use_median,

        # Decoder params
        use_hybrid_decoder=use_hybrid_decoder,
        k_hybrid_decoder=k_hybrid_decoder,
        gat_hidden_channels=gat_hidden_channels,
        gat_heads=gat_heads,
        min_points_for_graph=min_points_for_graph,
        initial_alpha_bias=initial_alpha_bias,

        # Attention kernel distance norm
        laplace_distance_p=smacnp_cfg.get('laplace_distance_p', 1.0),

        # Positional Encoding (unpack all kwargs)
        **pe_kwargs
    )
    return model


def main(cfg):
    """Main training function with graph model support."""
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # --- SET SEED FOR REPRODUCIBILITY ---
    set_seed(cfg['project']['seed'])
    # -------------------------------------

    try:
        data_cfg = cfg['data']
        model_type = cfg['model']['type']
        if model_type in ['GCNP', 'GRAPHNP', 'HYBRIDGRAPH']:
            processed_data_dir = data_cfg['processed_dir_gnn']
        else:
            processed_data_dir = data_cfg['processed_dir_all']

        scalers = {
            'temp_min': torch.from_numpy(np.load(os.path.join(processed_data_dir, "target_temp_min.npy"))).to(device, dtype=torch.float32),
            'temp_scale': torch.from_numpy(np.load(os.path.join(processed_data_dir, "target_temp_scale.npy"))).to(device, dtype=torch.float32),
        }
        print("Loaded and transferred scalers to device.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find scaler .npy files. Run prepare_data.py first. Error: {e}")

    print("Loading datasets...")
    train_cfg = cfg['training']
    use_amp = train_cfg.get('use_mixed_precision', False) and device.type == 'cuda'
    if use_amp:
        print("Using Automatic Mixed Precision (AMP).")

    train_dataset = SMACNPDataset_ALL(
        data_dir=processed_data_dir, split='train',
        context_fraction=train_cfg['context_fraction_train'], use_fixed_seed=False,
        on_grid_training=train_cfg.get('on_grid_training', True),
        model_type=model_type
    )
    val_dataset = SMACNPDataset_ALL(
        data_dir=processed_data_dir, split='val',
        context_fraction=0.5, use_fixed_seed=True,
        on_grid_training=train_cfg.get('on_grid_training', True),
        model_type=model_type
    )

    def _seed_worker(worker_id: int) -> None:
        # Derive NumPy/Python seeds from PyTorch's per-worker seed.
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    num_workers = int(train_cfg.get('num_workers', 4))
    train_generator = torch.Generator()
    train_generator.manual_seed(int(cfg['project']['seed']))
    val_generator = torch.Generator()
    val_generator.manual_seed(int(cfg['project']['seed']) + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        generator=train_generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=val_generator,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    # --- END MODIFIED SECTION ---

    if len(train_dataset) == 0: raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0: raise ValueError("Validation dataset is empty.")

    MODEL_TO_RUN = cfg['model']['type']
    print(f"Initializing {MODEL_TO_RUN} model...")
    
    checkpoint_dir = cfg['outputs']['checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)

    if MODEL_TO_RUN == "SMACNP":
        model = get_smacnp_model_from_config(cfg['model']).to(device)
        model_cfg = cfg['model']['smacnp'] # for suffix
        training_cfg = cfg.get('training', {})

        suffix_parts = get_smacnp_filename_suffix_parts(model_cfg)
        if training_cfg.get('on_grid_training', True):
            suffix_parts.append("onGrid")
        else:
            suffix_parts.append("offGrid")
        suffix_parts.append(training_cfg.get('precip_distribution', 'lognormal'))
        suffix_parts.append(f"bsz{training_cfg.get('batch_size')}")
        suffix = "_" + "_".join(suffix_parts)
        checkpoint_path = os.path.join(checkpoint_dir, f"smacnp{suffix}_best.pth")
        learning_rate = train_cfg['smacnp_lr']

    else:
        raise ValueError(f"Unknown model type: {MODEL_TO_RUN}")

    # --- Optimizer ---
    optimizer_cfg = train_cfg.get('optimizer', {'name': 'Adam'})
    optimizer_name = optimizer_cfg.get('name', 'Adam')
    weight_decay = float(optimizer_cfg.get('weight_decay', 1e-5))

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer in config: '{optimizer_name}'")
    print(f"Using {optimizer_name} optimizer with LR={learning_rate:.6f} and Weight Decay={weight_decay:.6f}")

    scaler = GradScaler(enabled=use_amp)

    # --- Scheduler ---
    scheduler_cfg = train_cfg['scheduler']
    scheduler_name = scheduler_cfg.get('name', 'ReduceLROnPlateau')
    print(f"Using '{scheduler_name}' learning rate scheduler.")

    if scheduler_name == 'CosineAnnealingLR':
        num_epochs = train_cfg['num_epochs']
        warmup_epochs = int(scheduler_cfg.get('warmup_epochs', 0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=float(scheduler_cfg.get('eta_min', 1e-7))
        )
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg['T_0'],
            T_mult=scheduler_cfg.get('T_mult', 1),
            eta_min=float(scheduler_cfg.get('eta_min', 1e-7))
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=scheduler_cfg['factor'],
            patience=scheduler_cfg['patience'],
            min_lr=float(scheduler_cfg.get('min_lr', 0.0)),
        )
    else:
        raise ValueError(f"Unsupported scheduler in config: '{scheduler_name}'")
    
    early_stopper_cfg = train_cfg['early_stopping']
    min_delta = float(early_stopper_cfg.get('min_delta', 0.0))
    early_stopper = EarlyStopping(
        patience=early_stopper_cfg['patience'], verbose=True, delta=min_delta, path=checkpoint_path
    )

    print("--- Starting Training ---")
    start_time = time.time()

    for epoch in range(train_cfg['num_epochs']):
        epoch_start_time = time.time()

        precip_dist = train_cfg.get('precip_distribution', 'lognormal')
        
        # --- MODIFIED: Pass is_graph_model flag ---
        train_loss = train_epoch(model, train_loader, optimizer, scaler, train_cfg['clip_grad_norm'],
                               device, train_cfg['loss_weights'], precip_dist, scalers, use_amp)
        val_loss, (temp_mae, temp_rmse), (precip_mae, precip_rmse) = validate_epoch(
            model, val_loader, device, scalers, train_cfg['loss_weights'], precip_dist, use_amp)
        # --- END MODIFIED ---

        # Scheduler & LR Update (unchanged)
        warmup_epochs = int(scheduler_cfg.get('warmup_epochs', 0))
        
        if scheduler_name in ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts'] and epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler_name == 'CosineAnnealingWarmRestarts':
                T_0 = scheduler_cfg['T_0']
                patience = int(scheduler_cfg.get('patience_on_restart', 0))
                is_restart_epoch = (epoch - warmup_epochs) % T_0 == (T_0 - 1)
                
                can_step = True
                if is_restart_epoch and patience > 0:
                    if len(early_stopper.val_loss_history) > patience:
                        recent_best = min(list(early_stopper.val_loss_history)[-patience:])
                        if val_loss < recent_best + 1e-4:
                            print(f"[Scheduler] Postponing restart: Val loss improved in the last {patience} epochs.")
                            can_step = False
                if can_step:
                    scheduler.step()
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        print(
            f"""Epoch {epoch+1}/{train_cfg['num_epochs']} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}
	Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}
	Temp Metrics:  MAE={temp_mae:.4f}, RMSE={temp_rmse:.4f}
	Precip Metrics: MAE={precip_mae:.4f}, RMSE={precip_rmse:.4f}"""
        )

        if val_loss != float('inf'):
            early_stopper(val_loss, model, optimizer, scheduler, epoch)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

    total_training_time = time.time() - start_time
    print(f"--- Training Finished. Total time: {total_training_time:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a climate gridding model using a YAML configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    try:
       with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {args.config}")

    main(cfg)
