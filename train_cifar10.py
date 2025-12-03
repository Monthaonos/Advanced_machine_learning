"""
Main training and evaluation script for CIFAR-10 models (SimpleCNN).

This script orchestrates the creation of two convolutional neural networks
based on the ``SimpleCIFAR10CNN`` architecture.

It handles automatic model loading from local disk (or Google Drive) AND
automatic downloading/uploading from S3 (MinIO) if configured.

Usage examples:
    # Local (Mac/PC):
    python train_cifar10.py --epochs 10

    # Onyxia (S3 auto-fetch):
    python train_cifar10.py --s3-bucket "maeltremouille/robustness_training"

    # Colab (Google Drive):
    python train_cifar10.py --save-dir "/content/drive/MyDrive/Checkpoints"
"""

import os
import argparse
import torch
from torch import nn
from torch import optim
import s3fs  # Nécessaire pour la gestion S3

# --- Imports spécifiques au modèle Standard ---
from services.dataloaders.cifar10_loader import get_cifar10_loaders
from services.models.cifar10_model import SimpleCIFAR10CNN
# ----------------------------------------------

from services.train_test import train_models, test_models, test_models_adversarial
from services.utils import save_model, load_model


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 models with S3/Drive support.",
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
        default=10,
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

    # --- ARGUMENTS DE STOCKAGE (Unified Storage) ---
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/cifar_10",
        help="Directory (Local or Drive) to save/load checkpoints.",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="Optional: S3 Bucket root (e.g., 'user/project') for cloud storage.",
    )
    # -----------------------------------------------

    parser.add_argument(
        "--load",
        action="store_true",
        help="Skip training and load existing models (Legacy flag, logic handles this automatically now).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force training even if checkpoints exist.",
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


def manage_checkpoint(model, local_path, s3_bucket=None, device="cpu"):
    """
    Logique unifiée de chargement :
    1. Vérifie le LOCAL (Disque dur ou Google Drive monté).
    2. Vérifie le S3 (si s3_bucket est fourni).
    3. Sinon, retourne False (indique qu'il faut entraîner).
    """
    # 1. Vérification LOCALE
    if os.path.exists(local_path):
        print(f"✅ Modèle trouvé localement : {local_path}")
        try:
            load_model(model, local_path, device=device)
            return True
        except Exception as e:
            print(f"⚠️ Erreur chargement local (fichier corrompu ?) : {e}")

    # 2. Vérification S3 (Si configuré)
    if s3_bucket:
        filename = os.path.basename(local_path)
        s3_path = f"{s3_bucket}/checkpoints/cifar_10/{filename}"

        print(f"☁️  Recherche sur S3 ({s3_path})...")
        try:
            fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
            )

            if fs.exists(s3_path):
                print(f"⬇️  Modèle trouvé sur S3. Téléchargement vers {local_path}...")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                fs.get(s3_path, local_path)

                load_model(model, local_path, device=device)
                print("✅ Modèle téléchargé et chargé avec succès.")
                return True
            else:
                print("❌ Modèle absent du S3.")
        except Exception as e:
            print(f"⚠️ Erreur de connexion/téléchargement S3 : {e}")

    # 3. Aucun modèle trouvé
    return False


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

    # Noms des fichiers (Standard Model)
    clean_ckpt = os.path.join(args.save_dir, "cifar10_clean.pth")
    robust_ckpt = os.path.join(args.save_dir, "cifar10_robust.pth")

    # ---------------------------------------------------------
    # 1. Gestion du modèle CLEAN (Standard)
    # ---------------------------------------------------------
    loaded_clean = manage_checkpoint(model_clean, clean_ckpt, args.s3_bucket, device)

    if not loaded_clean or args.force_retrain:
        print("\n⚡️ Démarrage entraînement CLEAN...")
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
        print(f"💾 Modèle Clean sauvegardé : {clean_ckpt}")

        # Auto-upload S3
        if args.s3_bucket:
            s3_path = f"{args.s3_bucket}/checkpoints/cifar10_clean.pth"
            print(f"⬆️  Upload vers S3 : {s3_path}")
            fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
            )
            fs.put(clean_ckpt, s3_path)

    # ---------------------------------------------------------
    # 2. Gestion du modèle ROBUSTE (Adversarial)
    # ---------------------------------------------------------
    loaded_robust = manage_checkpoint(model_robust, robust_ckpt, args.s3_bucket, device)

    if not loaded_robust or args.force_retrain:
        print("\n⚡️ Démarrage entraînement ROBUSTE (PGD)...")
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
        print(f"💾 Modèle Robuste sauvegardé : {robust_ckpt}")

        # Auto-upload S3
        if args.s3_bucket:
            s3_path = f"{args.s3_bucket}/checkpoints/cifar10_robust.pth"
            print(f"⬆️  Upload vers S3 : {s3_path}")
            fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
            )
            fs.put(robust_ckpt, s3_path)

    # Evaluate the clean model on clean test data
    print("\nEvaluating clean model on clean test data...")
    clean_loss, clean_acc = test_models(test_loader, model_clean, loss_fn, device)
    print(f"Clean model – loss: {clean_loss:.4f}, acc: {clean_acc:.4f}")

    # Evaluate the robust model on clean test data
    print("Evaluating robust model on clean test data...")
    robust_clean_loss, robust_clean_acc = test_models(
        test_loader, model_robust, loss_fn, device
    )
    print(f"Robust model – loss: {robust_clean_loss:.4f}, acc: {robust_clean_acc:.4f}")

    # Evaluate both models on adversarial test data
    print("\nEvaluating clean model on adversarial test data...")
    adv_loss_clean, adv_acc_clean = test_models_adversarial(
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
    print(
        f"Clean model (adv test) – loss: {adv_loss_clean:.4f}, acc: {adv_acc_clean:.4f}"
    )

    print("Evaluating robust model on adversarial test data...")
    adv_loss_robust, adv_acc_robust = test_models_adversarial(
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
    print(
        f"Robust model (adv test) – loss: {adv_loss_robust:.4f}, acc: {adv_acc_robust:.4f}"
    )


if __name__ == "__main__":
    main()
