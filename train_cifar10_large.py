"""
Main training and evaluation script for CIFAR-10 Large models (WideResNet).

This script mirrors the structure of train_cifar10.py but targets the
WideResNet architecture defined in services/cifar_10/cifar_10_large.
It adjusts hyperparameters (batch size, optimizer) to fit GPU memory constraints
and model convergence requirements.

Usage examples:
    python train_model_large.py --epochs 40 --batch-size 64
"""

import os
import argparse
import torch
from torch import nn
from torch import optim

# --- MODIFICATION: Imports spécifiques au modèle Large ---
from services.cifar_10_large.dataloader import get_dataloaders
from services.cifar_10_large.model import Network  # Le WideResNet
# ---------------------------------------------------------

from services.train_test import train_models, test_models, test_models_adversarial
from services.utils import save_model, load_model


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Train Large CIFAR-10 models (WRN) with and without PGD adversarial training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,  # RÉDUIT pour Colab (WRN consomme plus de VRAM)
        help="Mini‑batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,  # AUGMENTÉ car WRN a besoin de plus de temps pour converger
        help="Number of training epochs for each model.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,  # Adapté pour SGD
        help="Learning rate for the Optimizer.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8 / 255,
        help="Maximum L∞ perturbation for the PGD attack.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2 / 255,
        help="Step size for the PGD attack.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=7,  # PGD-7 est un standard efficace pour l'entraînement
        help="Number of gradient steps for the PGD attack.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Use a random start within the ε‑ball.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use.",
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
        help="Skip training and load existing models instead.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force training even if checkpoints exist.",
    )
    return parser.parse_args()


def build_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Instantiate two identical WideResNet models on the specified device."""
    # Instanciation avec les paramètres par défaut définis dans model.py (WRN-28-10)
    model_clean = Network().to(device)
    model_robust = Network().to(device)
    return model_clean, model_robust


def get_optimizers(
    model_clean: nn.Module,
    model_robust: nn.Module,
    learning_rate: float,
) -> tuple[optim.Optimizer, optim.Optimizer]:
    """Create separate Optimizers for the clean and robust models."""
    # Pour WideResNet, SGD avec Momentum est généralement préférable à Adam pour la robustesse
    opt_clean = optim.SGD(
        model_clean.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    opt_robust = optim.SGD(
        model_robust.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    return opt_clean, opt_robust


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Prepare data loaders (Uses the Large model specific dataloader)
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # Instantiate models and optimizers
    model_clean, model_robust = build_models(device)
    optimizer_clean, optimizer_robust = get_optimizers(
        model_clean, model_robust, args.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # --- MODIFICATION: Noms de fichiers distincts pour le modèle Large ---
    clean_ckpt = os.path.join(args.save_dir, "cifar10_large_clean.pth")
    robust_ckpt = os.path.join(args.save_dir, "cifar10_large_robust.pth")
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------
    # 1. Gestion du modèle LARGE CLEAN
    # ---------------------------------------------------------
    if os.path.exists(clean_ckpt) and not args.force_retrain:
        print(f"✅ Modèle Large Clean trouvé ({clean_ckpt}). Chargement...")
        load_model(model_clean, clean_ckpt, device=device)
    else:
        print("⚡️ Modèle Large Clean introuvable. Entraînement en cours...")
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
        print(f"💾 Modèle Large Clean sauvegardé sous {clean_ckpt}.")

    # ---------------------------------------------------------
    # 2. Gestion du modèle LARGE ROBUSTE (Adversarial)
    # ---------------------------------------------------------
    if os.path.exists(robust_ckpt) and not args.force_retrain:
        print(f"✅ Modèle Large Robuste trouvé ({robust_ckpt}). Chargement...")
        load_model(model_robust, robust_ckpt, device=device)
    else:
        print("⚡️ Modèle Large Robuste introuvable. Entraînement PGD en cours...")
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
        print(f"💾 Modèle Large Robuste sauvegardé sous {robust_ckpt}.")

    # Evaluate the clean model on clean test data
    print("\nEvaluating Large Clean model on clean test data...")
    clean_loss, clean_acc = test_models(
        test_dataloader=test_loader,
        model=model_clean,
        loss_fn=loss_fn,
        device=device,
    )
    print(f"Large Clean model – loss: {clean_loss:.4f}, acc: {clean_acc:.4f}")

    # Evaluate the robust model on clean test data
    print("Evaluating Large Robust model on clean test data...")
    robust_clean_loss, robust_clean_acc = test_models(
        test_dataloader=test_loader,
        model=model_robust,
        loss_fn=loss_fn,
        device=device,
    )
    print(
        f"Large Robust model – loss: {robust_clean_loss:.4f}, acc: {robust_clean_acc:.4f}"
    )

    # Evaluate both models on adversarial test data
    print("\nEvaluating Large Clean model on adversarial test data...")
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
        f"Large Clean model (adv test) – loss: {adv_loss_clean:.4f}, acc: {adv_acc_clean:.4f}"
    )

    print("Evaluating Large Robust model on adversarial test data...")
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
        f"Large Robust model (adv test) – loss: {adv_loss_robust:.4f}, acc: {adv_acc_robust:.4f}"
    )


if __name__ == "__main__":
    main()
