"""
Unified Training Script.

Usage:
    python -m services.train.train --model-type cifar10 --epochs 10 --device cuda
    python -m services.train.train --model-type cifar10_large --epochs 100 --learning-rate 0.1 --storage-path "s3://..."
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
    """Factory function for model components."""

    # 1. CIFAR-10 Standard Configuration
    if model_type == "cifar10":
        train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
        model_clean = SimpleCIFAR10CNN().to(device)
        model_robust = SimpleCIFAR10CNN().to(device)
        opt_cls = optim.Adam
        opt_kwargs = {"lr": learning_rate}
        prefix = "cifar10"

    # 2. CIFAR-10 Large Configuration (WideResNet)
    elif model_type == "cifar10_large":
        train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
        model_clean = WideResNet().to(device)
        model_robust = WideResNet().to(device)
        # WideResNet uses SGD + Momentum
        opt_cls = optim.SGD
        opt_kwargs = {"lr": learning_rate, "momentum": 0.9, "weight_decay": 5e-4}
        prefix = "cifar10_large"

    # 3. GTSRB Configuration
    elif model_type == "gtsrb":
        train_loader, _ = get_gtsrb_loaders(batch_size=batch_size)
        model_clean = GTSRBModel().to(device)
        model_robust = GTSRBModel().to(device)
        opt_cls = optim.Adam
        opt_kwargs = {"lr": learning_rate}
        prefix = "gtsrb"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_clean, model_robust, train_loader, opt_cls, opt_kwargs, prefix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Training Script for Robustness Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["cifar10", "cifar10_large", "gtsrb"],
        help="The architecture to train.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )

    # Adversarial Params
    parser.add_argument("--epsilon", type=float, default=8 / 255)
    parser.add_argument("--prob", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--random-start", action="store_true", default=True)

    # Storage
    default_storage = os.getenv("CHECKPOINT_ROOT", "checkpoints")
    parser.add_argument(
        "--storage-path",
        type=str,
        default=default_storage,
        help="Root path where models will be saved directy (Local or S3).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Overwrite existing checkpoints.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"🚀 Starting Training: {args.model_type}")
    print(f"💾 Storage Path: {args.storage_path}")

    # 1. Get Components
    model_clean, model_robust, train_loader, OptCls, opt_kwargs, prefix = (
        get_training_components(
            args.model_type, device, args.learning_rate, args.batch_size
        )
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer_clean = OptCls(model_clean.parameters(), **opt_kwargs)
    optimizer_robust = OptCls(model_robust.parameters(), **opt_kwargs)

    # 2. Scheduler Setup (Only for WideResNet/SGD)
    scheduler_clean = None
    scheduler_robust = None

    if args.model_type == "cifar10_large":
        print(f"🔧 Activation Scheduler (CosineAnnealing) pour {args.model_type}")
        scheduler_clean = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_clean, T_max=args.epochs
        )
        scheduler_robust = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_robust, T_max=args.epochs
        )

    # 3. File Names (DIRECT, sans sous-dossier forcé)
    # Le fichier sera stocké directement dans args.storage_path
    clean_filename = f"{prefix}_clean.pth"
    robust_filename = f"{prefix}_robust.pth"

    # =========================================================
    # A. Train CLEAN Model
    # =========================================================
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
            pgd_robust=False,
            device=device,
            scheduler=scheduler_clean,
        )
        save_checkpoint(model_clean, args.storage_path, clean_filename)

    # =========================================================
    # B. Train ROBUST Model
    # =========================================================
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
            pgd_robust=True,
            device=device,
            scheduler=scheduler_robust,
        )
        save_checkpoint(model_robust, args.storage_path, robust_filename)

    print("\n✅ Training Pipeline Completed.")


if __name__ == "__main__":
    main()
