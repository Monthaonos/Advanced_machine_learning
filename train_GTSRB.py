"""
Main training and evaluation script for GTSRB models.

This script trains two ResNet-based models on the German Traffic Sign Recognition Benchmark.

It handles automatic model loading from local disk (or Google Drive) AND
automatic downloading/uploading from S3 (MinIO) if configured.

Usage examples:
    # Onyxia (S3 auto-fetch):
    python train_gtsrb.py --s3-bucket "maeltremouille/robustness_training"

    # Colab (Google Drive):
    python train_gtsrb.py --save-dir "/content/drive/MyDrive/Checkpoints/GTSRB"
"""

import os
import argparse
import torch
from torch import nn
from torch import optim
import s3fs  # Nécessaire pour la gestion S3

# --- IMPORTS SPÉCIFIQUES GTSRB ---
from services.dataloaders.gtsrb_loader import get_gtsrb_loaders
from services.models.gtsrb_model import GTSRBModel
# ---------------------------------

from services.train_test import train_models, test_models, test_models_adversarial
from services.utils import save_model, load_model


def parse_args() -> argparse.Namespace:
    """Define and parse command‑line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Train GTSRB models with S3/Drive support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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
        default=1e-4,
        help="Learning rate for the Adam optimizer.",
    )
    # PGD Parameters
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
        help="Number of gradient steps for the PGD attack during training.",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        default=True,
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
        default="checkpoints/gtsrb",
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
        "--force-retrain",
        action="store_true",
        help="Force l'entraînement même si les fichiers de sauvegarde existent.",
    )
    return parser.parse_args()


def build_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Instantiate two identical GTSRB models (ResNet-based)."""
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


def manage_checkpoint(model, local_path, s3_bucket=None, device="cpu"):
    """
    Logique unifiée de chargement : Local -> S3 -> False (Train).
    """
    # 1. Vérification LOCALE
    if os.path.exists(local_path):
        print(f"✅ Modèle trouvé localement : {local_path}")
        try:
            load_model(model, local_path, device=device)
            return True
        except Exception as e:
            print(f"⚠️ Erreur chargement local : {e}")

    # 2. Vérification S3 (Si configuré)
    if s3_bucket:
        filename = os.path.basename(local_path)
        # On suppose que le fichier est dans 'checkpoints/gtsrb' ou juste 'checkpoints' selon ta préférence S3.
        # Ici j'aligne sur la structure : bucket/checkpoints/gtsrb_clean.pth pour éviter les conflits
        s3_path = f"{s3_bucket}/checkpoints/cifar_10_large/{filename}"

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
            print(f"⚠️ Erreur S3 : {e}")

    # 3. Rien trouvé
    return False


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"🚀 Running GTSRB training on {device}")

    # 1. Prepare data loaders (GTSRB handles download automatically)
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
    loaded_clean = manage_checkpoint(model_clean, clean_ckpt, args.s3_bucket, device)

    if not loaded_clean or args.force_retrain:
        print("⚡️ Démarrage entraînement Standard...")
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
        print(f"💾 Modèle clean sauvegardé : {clean_ckpt}")

        # Auto-upload S3
        if args.s3_bucket:
            s3_path = f"{args.s3_bucket}/checkpoints/gtsrb_clean.pth"
            print(f"⬆️  Upload vers S3 : {s3_path}")
            fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
            )
            fs.put(clean_ckpt, s3_path)

    # ---------------------------------------------------------
    # B. Modèle ROBUSTE (Adversarial Training)
    # ---------------------------------------------------------
    loaded_robust = manage_checkpoint(model_robust, robust_ckpt, args.s3_bucket, device)

    if not loaded_robust or args.force_retrain:
        print("⚡️ Démarrage entraînement Adversarial (PGD)...")
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
        print(f"💾 Modèle robuste sauvegardé : {robust_ckpt}")

        # Auto-upload S3
        if args.s3_bucket:
            s3_path = f"{args.s3_bucket}/checkpoints/gtsrb_robust.pth"
            print(f"⬆️  Upload vers S3 : {s3_path}")
            fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
            )
            fs.put(robust_ckpt, s3_path)

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
