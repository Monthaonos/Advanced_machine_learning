"""
Unified Training Service.
Can be run as a standalone script to train Clean and Robust models sequentially.

It leverages a TOML configuration file for defaults, which can be overridden
by command-line arguments.

Usage:
    python -m services.training --dataset cifar10 --model resnet18 --epochs 20
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Type, Dict, Any

# --- Modular Imports ---
from services.config_manager import load_config
from services.storage_manager import (
    manage_checkpoint,
    save_checkpoint,
    save_training_metrics,
)
from services.core import train_models
from services.dataloaders.factory import get_dataloaders
from services.models.factory import get_model


def get_optimizer_config(
    model_name: str, training_config: Dict[str, Any]
) -> Tuple[Type[optim.Optimizer], Dict[str, Any]]:
    """
    Determines the appropriate optimizer and its parameters based on the architecture.

    Args:
        model_name (str): The name of the model architecture.
        training_config (dict): The 'training' section from config.toml.

    Returns:
        Tuple[Type[optim.Optimizer], dict]: The optimizer class and its keyword arguments.
    """
    name = model_name.lower()
    lr = training_config.get("learning_rate", 0.01)

    # Large models (ResNet, WideResNet) usually train better with SGD + Momentum
    if "resnet" in name or "wideresnet" in name:
        return optim.SGD, {
            "lr": lr,
            "momentum": training_config.get("momentum", 0.9),
            "weight_decay": training_config.get("weight_decay", 5e-4),
        }

    # Simple models usually converge faster with Adam
    else:
        return optim.Adam, {"lr": lr}


def run_training(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Orchestrates the training pipeline for both Standard (Clean) and Adversarial (Robust) models.

    Configuration Priority:
    1. Command Line Arguments (if provided)
    2. config.toml values

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        config (dict): Loaded TOML configuration.
    """
    # 1. Merge Configuration (CLI overrides TOML)
    # We use 'args.param or config...' pattern.
    # Note: args must be None by default for this to work correctly.

    dataset = args.dataset or config["data"]["dataset"]
    model_name = args.model or config["model"]["architecture"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["data"]["batch_size"]
    storage_path = args.storage_path or config["project"]["storage_path"]
    force_retrain = args.force_retrain or config["training"].get(
        "force_retrain", False
    )

    # Device management
    device_name = args.device or config["project"].get("device", "cpu")
    device = torch.device(
        device_name
        if torch.cuda.is_available() or device_name == "cpu"
        else "cpu"
    )

    # Adversarial Hyperparameters
    adv_conf = config["adversarial"]
    epsilon = args.epsilon if args.epsilon is not None else adv_conf["epsilon"]
    alpha = args.alpha if args.alpha is not None else adv_conf["alpha"]
    num_steps = (
        args.num_steps if args.num_steps is not None else adv_conf["num_steps"]
    )
    train_prob = adv_conf.get("train_prob", 1.0)
    random_start = adv_conf.get("random_start", True)

    print(f"🚀 Starting Training Pipeline")
    print(f"   Dataset: {dataset} | Model: {model_name}")
    print(f"   Device:  {device} | Epochs: {epochs}")

    # 2. Get Data (Using Factory)
    print("📦 Loading Data...")
    # Note: get_dataloaders returns 4 values (train, test, classes, channels)
    train_loader, _, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=batch_size
    )
    print(f"   Classes: {num_classes}, Channels: {in_channels}")

    # 3. Prepare Common Configuration
    # Prefix management: CLI args > TOML > Default construction
    prefix = args.prefix or config["model"].get("prefix")
    if not prefix:
        prefix = f"{dataset}_{model_name}"

    loss_fn = nn.CrossEntropyLoss()
    OptCls, opt_kwargs = get_optimizer_config(model_name, config["training"])

    # =========================================================
    # Phase A: Clean Model (Standard Training)
    # =========================================================
    clean_filename = f"{prefix}_{dataset}_{model_name}_clean.pth"

    # Instantiate Model (Using Factory)
    model_clean = get_model(model_name, dataset, num_classes, in_channels).to(
        device
    )

    # Check if model exists
    loaded_clean = manage_checkpoint(
        model_clean, storage_path, clean_filename, device
    )

    if not loaded_clean or force_retrain:
        print(f"\n⚡️ Training [Clean] Model: {clean_filename} ...")

        # Setup Optimizer & Scheduler specific to this run
        optimizer_clean = OptCls(model_clean.parameters(), **opt_kwargs)
        scheduler_clean = None

        # Use Scheduler for complex models (ResNet/WideResNet)
        if "resnet" in model_name.lower():
            scheduler_clean = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_clean, T_max=epochs
            )

        clean_history = train_models(
            train_dataloader=train_loader,
            model=model_clean,
            loss_fn=loss_fn,
            optimizer=optimizer_clean,
            scheduler=scheduler_clean,
            epsilon=0.0,  # Irrelevant for standard training
            train_prob=0.0,
            alpha=0.0,
            num_steps=0,
            epochs=epochs,
            pgd_robust=False,  # <--- Standard Training Mode
            device=device,
        )

        # Save Artifacts
        save_checkpoint(model_clean, storage_path, clean_filename)
        save_training_metrics(
            clean_history,
            storage_path,
            f"{prefix}_{dataset}_{model_name}_clean_loss.csv",
        )
    else:
        print(f"⏩ [Clean] model already exists. Skipping training.")

    # =========================================================
    # Phase B: Robust Model (Adversarial Training)
    # =========================================================
    robust_filename = f"{prefix}_{dataset}_{model_name}_robust.pth"

    # Instantiate a FRESH Model instance for robust training
    model_robust = get_model(model_name, dataset, num_classes, in_channels).to(
        device
    )

    # Check if model exists
    loaded_robust = manage_checkpoint(
        model_robust, storage_path, robust_filename, device
    )

    if not loaded_robust or force_retrain:
        print(f"\n🛡️ Training [Robust] Model (PGD): {robust_filename} ...")

        optimizer_robust = OptCls(model_robust.parameters(), **opt_kwargs)
        scheduler_robust = None

        if "resnet" in model_name.lower():
            scheduler_robust = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_robust, T_max=epochs
            )

        robust_history = train_models(
            train_dataloader=train_loader,
            model=model_robust,
            loss_fn=loss_fn,
            optimizer=optimizer_robust,
            scheduler=scheduler_robust,
            epsilon=epsilon,
            train_prob=train_prob,
            alpha=alpha,
            num_steps=num_steps,
            random_start=random_start,
            epochs=epochs,
            pgd_robust=True,  # <--- Adversarial Training Mode
            device=device,
        )

        # Save Artifacts
        save_checkpoint(model_robust, storage_path, robust_filename)
        save_training_metrics(
            robust_history,
            storage_path,
            f"{prefix}_{dataset}_{model_name}_robust_loss.csv",
        )
    else:
        print(f"⏩ [Robust] model already exists. Skipping training.")

    print("\n✅ All training tasks completed.")


def parse_args():
    """
    Parses command line arguments to optionally override TOML configuration.
    """
    parser = argparse.ArgumentParser(
        description="Unified Training Service. Trains Standard and Robust models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Project & Hardware
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Root directory where checkpoints are stored (Local or S3). Defaults to [project.storage_path].",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Computation device. Defaults to [project.device].",
    )

    # 2. Model & Data Selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "gtsrb"],
        help="Target dataset. Defaults to [data.dataset].",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple_cnn", "resnet18", "wideresnet"],
        help="Model architecture. Defaults to [model.architecture].",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Specific filename prefix. Defaults to [model.prefix] or '{dataset}_{model}'.",
    )

    # 3. Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs. Defaults to [training.epochs].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size. Defaults to [data.batch_size].",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Optimizer learning rate. Defaults to [training.learning_rate].",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if a checkpoint exists.",
    )

    # 4. Adversarial Training Parameters (PGD)
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Max perturbation (L-inf). Defaults to [adversarial.epsilon].",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Step size for PGD. Defaults to [adversarial.alpha].",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of PGD steps. Defaults to [adversarial.num_steps].",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 1. Load defaults from config.toml
    try:
        config = load_config("config.toml")
    except Exception as e:
        print(f"⚠️ Error loading config.toml: {e}")
        print(
            "   Ensure 'config.toml' is at the root and 'tomli' is installed if using Python < 3.11"
        )
        exit(1)

    # 2. Parse CLI args
    args = parse_args()

    # 3. Run Pipeline
    run_training(args, config)
