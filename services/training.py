"""
Training service for standard (clean) and adversarial (robust) models.

Can be run standalone:
    python -m services.training --dataset cifar10 --model resnet18 --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Type, Dict, Any

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
    """Select optimizer and hyperparameters based on architecture."""
    name = model_name.lower()
    lr = training_config.get("learning_rate", 0.01)

    if "resnet" in name or "wideresnet" in name:
        return optim.SGD, {
            "lr": lr,
            "momentum": training_config.get("momentum", 0.9),
            "weight_decay": training_config.get("weight_decay", 5e-4),
        }
    else:
        return optim.Adam, {"lr": lr}


def run_training(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Orchestrates training for both clean and adversarially-trained models.

    Configuration priority: CLI arguments > config.toml values.
    """
    dataset = args.dataset or config["data"]["dataset"]
    model_name = args.model or config["model"]["architecture"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["data"]["batch_size"]
    storage_path = args.storage_path or config["project"]["storage_path"]
    force_retrain = args.force_retrain or config["training"].get(
        "force_retrain", False
    )

    device_name = args.device or config["project"].get("device", "cpu")
    device = torch.device(
        device_name
        if torch.cuda.is_available() or device_name == "cpu"
        else "cpu"
    )

    adv_conf = config["adversarial"]
    epsilon = args.epsilon if args.epsilon is not None else adv_conf["epsilon"]
    alpha = args.alpha if args.alpha is not None else adv_conf["alpha"]
    num_steps = (
        args.num_steps if args.num_steps is not None else adv_conf["num_steps"]
    )
    train_prob = adv_conf.get("train_prob", 1.0)
    random_start = adv_conf.get("random_start", True)

    print(f"[*] Training Pipeline")
    print(f"    Dataset: {dataset} | Model: {model_name}")
    print(f"    Device:  {device} | Epochs: {epochs}")

    train_loader, _, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=batch_size
    )
    print(f"    Classes: {num_classes}, Channels: {in_channels}")

    prefix = args.prefix or config["model"].get("prefix")
    if not prefix:
        prefix = f"{dataset}_{model_name}"

    loss_fn = nn.CrossEntropyLoss()
    OptCls, opt_kwargs = get_optimizer_config(model_name, config["training"])

    # =========================================================
    # Phase A: Clean Model (Standard Training)
    # =========================================================
    clean_filename = f"{prefix}_{dataset}_{model_name}_clean.pth"
    model_clean = get_model(model_name, dataset, num_classes, in_channels).to(
        device
    )

    loaded_clean = manage_checkpoint(
        model_clean, storage_path, clean_filename, device
    )

    if not loaded_clean or force_retrain:
        print(f"\n[*] Training clean model: {clean_filename}")

        optimizer_clean = OptCls(model_clean.parameters(), **opt_kwargs)
        scheduler_clean = None

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
            epsilon=0.0,
            train_prob=0.0,
            alpha=0.0,
            num_steps=0,
            epochs=epochs,
            pgd_robust=False,
            device=device,
        )

        save_checkpoint(model_clean, storage_path, clean_filename)
        save_training_metrics(
            clean_history,
            storage_path,
            f"{prefix}_{dataset}_{model_name}_clean_loss.csv",
        )
    else:
        print(f"[=] Clean model already exists. Skipping training.")

    # =========================================================
    # Phase B: Robust Model (Adversarial Training)
    # =========================================================
    robust_filename = f"{prefix}_{dataset}_{model_name}_robust.pth"
    model_robust = get_model(model_name, dataset, num_classes, in_channels).to(
        device
    )

    loaded_robust = manage_checkpoint(
        model_robust, storage_path, robust_filename, device
    )

    if not loaded_robust or force_retrain:
        print(f"\n[*] Training robust model (PGD): {robust_filename}")

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
            pgd_robust=True,
            device=device,
        )

        save_checkpoint(model_robust, storage_path, robust_filename)
        save_training_metrics(
            robust_history,
            storage_path,
            f"{prefix}_{dataset}_{model_name}_robust_loss.csv",
        )
    else:
        print(f"[=] Robust model already exists. Skipping training.")

    print("\n[+] All training tasks completed.")


def parse_args():
    """Parse CLI arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Training service for standard and robust models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--storage-path", type=str, help="Root checkpoint directory.")
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], help="Computation device."
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "gtsrb"], help="Target dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple_cnn", "resnet18", "wideresnet"],
        help="Model architecture.",
    )
    parser.add_argument("--prefix", type=str, help="Experiment identifier.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
    parser.add_argument(
        "--force-retrain", action="store_true", help="Force retraining."
    )
    parser.add_argument("--epsilon", type=float, help="Max L-inf perturbation.")
    parser.add_argument("--alpha", type=float, help="PGD step size.")
    parser.add_argument("--num-steps", type=int, help="Number of PGD steps.")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        config = load_config("config.toml")
    except Exception as e:
        print(f"[!] Error loading config.toml: {e}")
        exit(1)

    args = parse_args()
    run_training(args, config)
