import re
import matplotlib.pyplot as plt
import argparse
import os
import yaml
import sys

# Add script directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_convcnp_filename_suffix_parts, get_smacnp_filename_suffix_parts

def parse_log_file(log_file_path):
    """
    Parses a training log file to extract train and validation losses.
    
    Args:
        log_file_path (str): The path to the log file.

    Returns:
        tuple: A tuple containing two lists (train_losses, val_losses).
    """
    train_losses = []
    val_losses = []

    # Regex to find the line with both train and validation loss
    loss_pattern = re.compile(r"Train Loss: ([\d.-]+) \| Val Loss: ([\d.-]+)")

    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = loss_pattern.search(line)
                if match:
                    train_loss = float(match.group(1))
                    val_loss = float(match.group(2))
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None, None

    return train_losses, val_losses

def plot_loss_curves(train_losses, val_losses, output_path):
    """
    Plots and saves the training and validation loss curves.
    """
    if not train_losses or not val_losses:
        print("No loss data to plot.")
        return

    epochs = range(1, len(train_losses) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    plt.plot(epochs, train_losses, label='Training Loss', color='dodgerblue', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='darkorange', linestyle='--', linewidth=2)
    plt.ylim(-8, 2)
    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Negative Log-Likelihood)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Saved training loss plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss from a log file.")
    parser.add_argument(
        '--log_file', 
        type=str, 
        default='sh/logs/train_20250806_151017.log',
        help='Path to the training log file.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to the YAML configuration file used for the training run.'
    )
    args = parser.parse_args()

    # Load config to generate suffix
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {args.config}. Cannot generate descriptive filename.")
        cfg = None

    suffix_parts = []
    if cfg:
        # Safely access nested keys
        model_type = cfg.get('model', {}).get('type', 'SMACNP')
        training_cfg = cfg.get('training', {})

        if model_type == "SMACNP":
            smacnp_cfg = cfg.get('model', {}).get('smacnp', {})
            suffix_parts.extend(get_smacnp_filename_suffix_parts(smacnp_cfg))
        elif model_type == "GRIDDEDTNP":
            tnp_cfg = cfg.get('model', {}).get('griddedtnp', {})
            encoder_type = tnp_cfg.get('encoder_type', 'unknown').lower()
            is_equivariant = tnp_cfg.get('equivariant', False)
            suffix = f"{encoder_type}_te" if is_equivariant else encoder_type
            suffix_parts.append(f"tnp_{suffix}")
        elif model_type == "CONVCNP":
            convcnp_cfg = cfg.get('model', {}).get('convcnp', {})
            suffix_parts.append('convcnp')
            suffix_parts.extend(get_convcnp_filename_suffix_parts(convcnp_cfg))

        if training_cfg.get('on_grid_training', True):
            suffix_parts.append("onGrid")
        else:
            suffix_parts.append("offGrid")
        suffix_parts.append(training_cfg.get('precip_distribution', 'lognormal'))
        suffix_parts.append(f"bsz{training_cfg.get('batch_size')}")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    train_losses, val_losses = parse_log_file(args.log_file)

    # Generate output path from config
    output_dir = "outputs/plots"
    scheduler_type = "unknownScheduler"
    if cfg and 'scheduler' in cfg.get('training', {}):
        scheduler_type = cfg['training']['scheduler'].get('name', 'unknownScheduler')

    # Extract timestamp from log file name
    timestamp_match = re.search(r'train_(\d{8}_\d{6})', args.log_file)
    timestamp_str = f"{timestamp_match.group(1)}_" if timestamp_match else ""

    filename = f"training/loss_curve_{timestamp_str}{scheduler_type}{suffix}.png"
    output_plot_path = os.path.join(output_dir, filename)
    plot_loss_curves(train_losses, val_losses, output_plot_path)