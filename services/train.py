"""
Unified Training Script.

This script acts as a factory to train different model architectures (CIFAR-10,
CIFAR-10 Large, GTSRB) using a consistent pipeline.

Usage:
    python -m services.train.train --model-type cifar10 --epochs 10 --device cuda
    python -m services.train.train --model-type gtsrb --storage-path "s3://my-bucket/..."
"""

import argparse
import os
import torch
from torch import nn
from torch import optim
from dotenv import load_dotenv

# Shared Services
from services.storage_manager import manage_checkpoint, save_checkpoint
from services.core import train_models

# Import Loaders and Models
from services.dataloaders.cifar10_loader import get_cifar10_loaders
from services.dataloaders.gtsrb_loader import get_gtsrb_loaders
from services.models.cifar10_model import SimpleCIFAR10CNN
from services.models.cifar10_large_model import Network as WideResNet
from services.models.gtsrb_model import GTSRBModel

load_dotenv()


def get_training_components(model_type, device, learning_rate, batch_size):
    """
    Factory function that returns the specific configuration for the requested model type.

    Args:
        model_type (str): The identifier of the model (cifar10, cifar10_large, gtsrb).
        device (torch.device): The computation device.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: (model_clean, model_robust, train_loader, OptimizerClass, optimizer_kwargs, prefix_name)
    """

    # 1. CIFAR-10 Standard Configuration
    if model_type == "cifar10":
        # We ignore the test loader here (_) as evaluation is handled separately
        train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
        model_clean = SimpleCIFAR10CNN().to(device)
        model_robust = SimpleCIFAR10CNN().to(device)

        # CIFAR-10 SimpleCNN typically uses Adam
        opt_cls = optim.Adam
        opt_kwargs = {"lr": learning_rate}
        prefix = "cifar10"

    # 2. CIFAR-10 Large Configuration (WideResNet)
    elif model_type == "cifar10_large":
        train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
        model_clean = WideResNet().to(device)
        model_robust = WideResNet().to(device)

        # WideResNet typically uses SGD with Momentum
        opt_cls = optim.SGD
        opt_kwargs = {"lr": learning_rate, "momentum": 0.9, "weight_decay": 5e-4}
        prefix = "cifar10_large"

    # 3. GTSRB Configuration
    elif model_type == "gtsrb":
        train_loader, _ = get_gtsrb_loaders(batch_size=batch_size)
        model_clean = GTSRBModel().to(device)
        model_robust = GTSRBModel().to(device)

        # GTSRB typically uses Adam
        opt_cls = optim.Adam
        opt_kwargs = {"lr": learning_rate}
        prefix = "gtsrb"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_clean, model_robust, train_loader, opt_cls, opt_kwargs, prefix


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Training Script for Robustness Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Key argument to select the model architecture
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cifar10", "cifar10_large", "gtsrb"],
        help="The architecture to train.",
    )

    # Standard training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda, mps, cpu).",
    )

    # Adversarial Training parameters (PGD)
    parser.add_argument(
        "--epsilon", type=float, default=8 / 255, help="PGD epsilon (L-inf norm)."
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=1.0,
        help="Probability of adversarial attack during training.",
    )
    parser.add_argument("--alpha", type=float, default=2 / 255, help="PGD step size.")
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of PGD steps."
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        default=True,
        help="Use random start for PGD.",
    )

    # Unified Storage configuration (Local / S3)
    default_storage = os.getenv("CHECKPOINT_ROOT", "checkpoints")
    parser.add_argument(
        "--storage-path",
        type=str,
        default=default_storage,
        help="Root path for checkpoints (local path or s3://uri).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="If set, overwrite existing checkpoints and retrain.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"🚀 Starting Unified Training Pipeline: {args.model_type} on {device}")
    print(f"💾 Storage Path: {args.storage_path}")

    # 1. Retrieve specific components based on model type
    model_clean, model_robust, train_loader, OptCls, opt_kwargs, prefix = (
        get_training_components(
            args.model_type, device, args.learning_rate, args.batch_size
        )
    )

    loss_fn = nn.CrossEntropyLoss()

    # Initialize optimizers with the retrieved class and kwargs
    optimizer_clean = OptCls(model_clean.parameters(), **opt_kwargs)
    optimizer_robust = OptCls(model_robust.parameters(), **opt_kwargs)

    # Define standardized filenames (e.g., cifar10_large_clean.pth)
    # If the model type implies a subfolder in your logic, include it in the prefix or handle it via storage-path
    clean_filename = f"{prefix}_clean.pth"
    robust_filename = f"{prefix}_robust.pth"

    # =========================================================
    # A. Train CLEAN Model (Standard)
    # =========================================================
    # Check if model exists locally or on S3
    loaded_clean = manage_checkpoint(
        model_clean, args.storage_path, clean_filename, device
    )

    if not loaded_clean or args.force_retrain:
        print(f"\n⚡️ Training {prefix} (Standard)...")
        train_models(
            train_dataloader=train_loader,
            model=model_clean,
            loss_fn=loss_fn,
            optimizer=optimizer_clean,
            epsilon=args.epsilon,
            prob=args.prob,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            epochs=args.epochs,
            pgd_robust=False,  # Standard training
            device=device,
        )
        # Save locally and upload to S3 if configured
        save_checkpoint(model_clean, args.storage_path, clean_filename)

    # =========================================================
    # B. Train ROBUST Model (Adversarial)
    # =========================================================
    # Check if model exists locally or on S3
    loaded_robust = manage_checkpoint(
        model_robust, args.storage_path, robust_filename, device
    )

    if not loaded_robust or args.force_retrain:
        print(f"\n⚡️ Training {prefix} (Adversarial/Robust)...")
        train_models(
            train_dataloader=train_loader,
            model=model_robust,
            loss_fn=loss_fn,
            optimizer=optimizer_robust,
            epsilon=args.epsilon,
            prob=args.prob,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            epochs=args.epochs,
            pgd_robust=True,  # Adversarial training
            device=device,
        )
        # Save locally and upload to S3 if configured
        save_checkpoint(model_robust, args.storage_path, robust_filename)

    print("\n✅ Training Pipeline Completed.")


if __name__ == "__main__":
    main()
