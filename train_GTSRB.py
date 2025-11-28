"""
Main training and evaluation script for GTSRB models.

This script trains two ResNet-based models on the German Traffic Sign Recognition Benchmark:
1. Standard Model: Trained with standard empirical risk minimisation.
2. Robust Model: Trained with PGD adversarial training.

It automatically handles the download of the dataset via torchvision if needed.
"""

import os
import argparse
import torch
from torch import nn
from torch import optim

# --- IMPORTS SPÉCIFIQUES GTSRB ---
from services.GTSRB.dataloader import get_gtsrb_loaders
from services.GTSRB.model import GTSRBModel
# ---------------------------------

from services.train_test import train_models, test_models, test_models_adversarial
from services.utils import save_model, load_model


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Train GTSRB models with and without PGD adversarial training, then evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,  # GTSRB images are larger/ResNet is heavier, so smaller batch size might be needed
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
        default=1e-4,  # Lower learning rate for ResNet fine-tuning often works better
        help="Learning rate for the Adam optimizer.",
    )
    # PGD Parameters
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8 / 255,  # Standard perturbation budget
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
        default=10,  # 10 steps is a good trade-off for training
        help="Number of gradient steps for the PGD attack during training.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        default=True,
        help="Use a random start within the ε‑ball.",
    )
    # System & Save
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/gtsrb",  # Separate folder for GTSRB checkpoints
        help="Directory in which to save and load model checkpoints.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force l'entraînement même si les fichiers de sauvegarde existent.",
    )
    return parser.parse_args()


def build_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Instantiate two identical GTSRB models (ResNet-based) on the specified device."""
    # Note: GTSRBModel includes the internal normalization layer
    model_clean = GTSRBModel().to(device)
    model_robust = GTSRBModel().to(device)
    return model_clean, model_robust


def get_optimizers(
    model_clean: nn.Module,
    model_robust: nn.Module,
    learning_rate: float,
) -> tuple[optim.Optimizer, optim.Optimizer]:
    """Create separate Adam optimizers."""
    opt_clean = optim.Adam(model_clean.parameters(), lr=learning_rate)
    opt_robust = optim.Adam(model_robust.parameters(), lr=learning_rate)
    return opt_clean, opt_robust


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"🚀 Running GTSRB training on {device}")

    # 1. Prepare data loaders (GTSRB)
    print("⏳ Loading GTSRB Dataset...")
    train_loader, test_loader = get_gtsrb_loaders(batch_size=args.batch_size)

    # 2. Instantiate models and optimizers
    model_clean, model_robust = build_models(device)
    optimizer_clean, optimizer_robust = get_optimizers(
        model_clean, model_robust, args.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()

    # 3. Checkpoints setup
    os.makedirs(args.save_dir, exist_ok=True)
    clean_ckpt = os.path.join(args.save_dir, "gtsrb_clean.pth")
    robust_ckpt = os.path.join(args.save_dir, "gtsrb_robust.pth")

    # ---------------------------------------------------------
    # A. Modèle CLEAN (Standard)
    # ---------------------------------------------------------
    if os.path.exists(clean_ckpt) and not args.force_retrain:
        print(f"✅ Modèle clean trouvé ({clean_ckpt}). Chargement...")
        load_model(model_clean, clean_ckpt, device=device)
    else:
        print("⚡️ Modèle clean introuvable. Entraînement Standard...")
        train_models(
            train_dataloader=train_loader,
            model=model_clean,
            loss_fn=loss_fn,
            optimizer=optimizer_clean,
            epsilon=args.epsilon,  # Used only if pgd_robust=True
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            clamp_min=0.0,
            clamp_max=1.0,
            epochs=args.epochs,
            pgd_robust=False,  # <--- Standard Training
            device=device,
        )
        save_model(model_clean, clean_ckpt)
        print(f"💾 Modèle clean sauvegardé sous {clean_ckpt}.")

    # ---------------------------------------------------------
    # B. Modèle ROBUSTE (Adversarial Training)
    # ---------------------------------------------------------
    if os.path.exists(robust_ckpt) and not args.force_retrain:
        print(f"✅ Modèle robuste trouvé ({robust_ckpt}). Chargement...")
        load_model(model_robust, robust_ckpt, device=device)
    else:
        print("⚡️ Modèle robuste introuvable. Entraînement Adversarial (PGD)...")
        print(f"   (PGD settings: eps={args.epsilon:.4f}, steps={args.num_steps})")
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
            pgd_robust=True,  # <--- Adversarial Training
            device=device,
        )
        save_model(model_robust, robust_ckpt)
        print(f"💾 Modèle robuste sauvegardé sous {robust_ckpt}.")

    # ---------------------------------------------------------
    # C. ÉVALUATION COMPARATIVE
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("📊 RÉSULTATS D'ÉVALUATION (GTSRB)")
    print("=" * 40)

    # 1. Clean Accuracy
    print("\n1️⃣  Précision sur images NORMALES :")
    _, clean_acc = test_models(test_loader, model_clean, loss_fn, device)
    _, robust_clean_acc = test_models(test_loader, model_robust, loss_fn, device)
    print(f"   🔹 Modèle Standard : {clean_acc:.2%}")
    print(f"   🔸 Modèle Robuste  : {robust_clean_acc:.2%}")

    # 2. Adversarial Accuracy
    print(f"\n2️⃣  Précision sous ATTAQUE PGD (eps={args.epsilon:.4f}) :")

    # Clean Model under attack
    _, adv_acc_clean = test_models_adversarial(
        test_loader,
        model_clean,
        loss_fn,
        args.epsilon,
        args.alpha,
        args.num_steps,
        args.random_start,
        0.0,
        1.0,
        device,
    )
    print(f"   🔹 Modèle Standard : {adv_acc_clean:.2%} (⚠️ Devrait être bas)")

    # Robust Model under attack
    _, adv_acc_robust = test_models_adversarial(
        test_loader,
        model_robust,
        loss_fn,
        args.epsilon,
        args.alpha,
        args.num_steps,
        args.random_start,
        0.0,
        1.0,
        device,
    )
    print(f"   🔸 Modèle Robuste  : {adv_acc_robust:.2%} (🛡️ Devrait être haut)")


if __name__ == "__main__":
    main()
