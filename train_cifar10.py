"""
Main training and evaluation script for CIFAR‑10 models.

This script orchestrates the creation of two convolutional neural networks
based on the ``SimpleCIFAR10CNN`` architecture, trains one with standard
empirical risk minimisation and the other with PGD adversarial training, and
compares their performance on clean and adversarial test data.  It also
provides an option to reload previously saved models to skip the training
phase entirely.

Usage examples:

    # Train both models for 10 epochs and save checkpoints
    python main.py --epochs 10 --save-dir ./checkpoints

    # Load previously saved models and evaluate only
    python main.py --load --save-dir ./checkpoints

The script accepts a number of hyperparameters as command‑line arguments;
run ``python main.py --help`` for a full description.
"""

import os
import argparse
import torch
from torch import nn
from torch import optim

from services.cifar_10.dataloader import get_cifar10_loaders
from services.cifar_10.model import SimpleCIFAR10CNN
from services.train_test import train_models, test_models, test_models_adversarial
from services.utils import save_model, load_model


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Train CIFAR‑10 models with and without PGD adversarial training, then evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini‑batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for each model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8 / 255,
        help="Maximum L∞ perturbation for the PGD attack (training and testing).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="Step size for the PGD attack (training and testing).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of gradient steps for the PGD attack.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Use a random start within the ε‑ball when generating adversarial examples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/cifar_10",
        help="Directory in which to save and load model checkpoints.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Skip training and load existing models from --save-dir instead.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force l'entraînement même si les fichiers de sauvegarde existent.",
    )
    return parser.parse_args()


def build_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Instantiate two identical CNN models on the specified device."""
    model_clean = SimpleCIFAR10CNN().to(device)
    model_robust = SimpleCIFAR10CNN().to(device)
    return model_clean, model_robust


def get_optimizers(
    model_clean: nn.Module,
    model_robust: nn.Module,
    learning_rate: float,
) -> tuple[optim.Optimizer, optim.Optimizer]:
    """Create separate Adam optimizers for the clean and robust models."""
    opt_clean = optim.Adam(model_clean.parameters(), lr=learning_rate)
    opt_robust = optim.Adam(model_robust.parameters(), lr=learning_rate)
    return opt_clean, opt_robust


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Prepare data loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # Instantiate models and optimizers
    model_clean, model_robust = build_models(device)
    optimizer_clean, optimizer_robust = get_optimizers(
        model_clean, model_robust, args.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    clean_ckpt = os.path.join(args.save_dir, "cifar10_clean.pth")
    robust_ckpt = os.path.join(args.save_dir, "cifar10_robust.pth")

    # ---------------------------------------------------------
    # 1. Gestion du modèle CLEAN (Standard)
    # ---------------------------------------------------------
    if os.path.exists(clean_ckpt) and not args.force_retrain:
        print(f"✅ Modèle clean trouvé ({clean_ckpt}). Chargement...")
        load_model(model_clean, clean_ckpt, device=device)
    else:
        print("⚡️ Modèle clean introuvable. Entraînement en cours...")
        train_models(
            train_dataloader=train_loader,
            model=model_clean,
            loss_fn=loss_fn,
            optimizer=optimizer_clean,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            clamp_min=0.0,
            clamp_max=1.0,
            epochs=args.epochs,
            pgd_robust=False,
            device=device,
        )
        save_model(model_clean, clean_ckpt)
        print(f"💾 Modèle clean sauvegardé sous {clean_ckpt}.")

    # ---------------------------------------------------------
    # 2. Gestion du modèle ROBUSTE (Adversarial)
    # ---------------------------------------------------------
    if os.path.exists(robust_ckpt) and not args.force_retrain:
        print(f"✅ Modèle robuste trouvé ({robust_ckpt}). Chargement...")
        load_model(model_robust, robust_ckpt, device=device)
    else:
        print("⚡️ Modèle robuste introuvable. Entraînement PGD en cours...")
        train_models(
            train_dataloader=train_loader,
            model=model_robust,
            loss_fn=loss_fn,
            optimizer=optimizer_robust,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            clamp_min=0.0,
            clamp_max=1.0,
            epochs=args.epochs,
            pgd_robust=True,
            device=device,
        )
        save_model(model_robust, robust_ckpt)
        print(f"💾 Modèle robuste sauvegardé sous {robust_ckpt}.")

    # Evaluate the clean model on clean test data
    print("\nEvaluating clean model on clean test data...")
    clean_loss, clean_acc = test_models(
        test_dataloader=test_loader,
        model=model_clean,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Clean model – loss: {clean_loss:.4f}, acc: {clean_acc:.4f}")

    # Evaluate the robust model on clean test data
    print("Evaluating robust model on clean test data...")
    robust_clean_loss, robust_clean_acc = test_models(
        test_dataloader=test_loader,
        model=model_robust,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Robust model – loss: {robust_clean_loss:.4f}, acc: {robust_clean_acc:.4f}")

    # Evaluate both models on adversarial test data
    print("\nEvaluating clean model on adversarial test data...")
    adv_loss_clean, adv_acc_clean = test_models_adversarial(
        test_dataloader=test_loader,
        model=model_clean,
        loss_fn=loss_fn,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.num_steps,
        random_start=args.random_start,
        clamp_min=0.0,
        clamp_max=1.0,
        device=device,
    )
    print(
        f"Clean model (adv test) – loss: {adv_loss_clean:.4f}, acc: {adv_acc_clean:.4f}"
    )

    print("Evaluating robust model on adversarial test data...")
    adv_loss_robust, adv_acc_robust = test_models_adversarial(
        test_dataloader=test_loader,
        model=model_robust,
        loss_fn=loss_fn,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.num_steps,
        random_start=args.random_start,
        clamp_min=0.0,
        clamp_max=1.0,
        device=device,
    )
    print(
        f"Robust model (adv test) – loss: {adv_loss_robust:.4f}, acc: {adv_acc_robust:.4f}"
    )


if __name__ == "__main__":
    main()
