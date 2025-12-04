import argparse
import os
import torch
import pandas as pd
from dotenv import load_dotenv

# Services
from services.storage_manager import manage_checkpoint, save_results
from services.evaluator import run_evaluation_suite
from services.attacks import (
    fgsm_attack,
    pgd_attack,
    mim_attack,
)

# Loaders & Models
from services.dataloaders.cifar10_loader import get_cifar10_loaders
from services.dataloaders.gtsrb_loader import get_gtsrb_loaders
from services.models.cifar10_model import SimpleCIFAR10CNN
from services.models.cifar10_large_model import Network as WideResNet
from services.models.gtsrb_model import GTSRBModel

load_dotenv()


def get_architecture(arch_name, device):
    if arch_name == "cifar10":
        return SimpleCIFAR10CNN().to(device)
    if arch_name == "cifar10_large":
        return WideResNet().to(device)
    if arch_name == "gtsrb":
        return GTSRBModel().to(device)
    return None


def main():
    parser = argparse.ArgumentParser(description="Advanced Robustness Benchmark")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    default_storage = os.getenv("CHECKPOINT_ROOT", "checkpoints")
    parser.add_argument("--storage-path", type=str, default=default_storage)
    parser.add_argument("--output-filename", type=str, default="benchmark_results.csv")

    # --- NOUVEAU : FILTRE DE CIBLE ---
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "cifar10", "cifar10_large", "gtsrb"],
        help="Quel modèle évaluer ? (cifar10, cifar10_large, gtsrb ou all)",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # ==========================================
    # 1. CONFIGURATION DES ATTAQUES
    # ==========================================
    EPS = 8 / 255
    ALPHA = 2 / 255

    attack_suite = {
        "Clean": {"fn": None},
        "FGSM": {"fn": fgsm_attack, "kwargs": {"epsilon": EPS}},
        "PGD_10": {
            "fn": pgd_attack,
            "kwargs": {"epsilon": EPS, "alpha": ALPHA, "num_steps": 10},
        },
        # Tu peux décommenter si tu veux MIM
        # "MIM_20": {
        #     "fn": mim_attack,
        #     "kwargs": {"epsilon": EPS, "alpha": ALPHA, "num_steps": 20, "decay": 1.0},
        # },
    }

    # ==========================================
    # 2. CATALOGUE DES MODÈLES
    # ==========================================
    full_catalog = [
        # --- GTSRB ---
        {
            "arch": "gtsrb",
            "data": "gtsrb",
            "file": "gtsrb/gtsrb_clean.pth",
            "name": "GTSRB (Standard)",
        },
        {
            "arch": "gtsrb",
            "data": "gtsrb",
            "file": "gtsrb/gtsrb_robust.pth",
            "name": "GTSRB (Robust)",
        },
        # --- CIFAR-10 (Small) ---
        {
            "arch": "cifar10",
            "data": "cifar10",
            "file": "cifar_10/cifar10_clean.pth",
            "name": "Cifar10 Small (Standard)",
        },
        {
            "arch": "cifar10",
            "data": "cifar10",
            "file": "cifar_10/cifar10_robust.pth",
            "name": "Cifar10 Small (Robust)",
        },
        # --- CIFAR-10 Large (WideResNet) ---
        {
            "arch": "cifar10_large",
            "data": "cifar10",
            "file": "cifar_10_large/cifar10_large_clean.pth",
            "name": "WideResNet (Standard)",
        },
        {
            "arch": "cifar10_large",
            "data": "cifar10",
            "file": "cifar_10_large/cifar10_large_robust.pth",
            "name": "WideResNet (Robust)",
        },
    ]

    # ==========================================
    # 3. FILTRAGE
    # ==========================================
    if args.target == "all":
        scenarios = full_catalog
        print("🌍 Mode: Évaluation de TOUS les modèles.")
    else:
        # On ne garde que ceux dont l'architecture correspond à la cible
        scenarios = [s for s in full_catalog if s["arch"] == args.target]
        print(f"🎯 Mode: Cible unique -> {args.target}")

    # ==========================================
    # 4. CHARGEMENT DES DATASETS (Optimisé)
    # ==========================================
    loaders = {}
    needed_datasets = set(s["data"] for s in scenarios)

    if "cifar10" in needed_datasets:
        print("⏳ Chargement CIFAR-10...")
        _, loaders["cifar10"] = get_cifar10_loaders(batch_size=args.batch_size)

    if "gtsrb" in needed_datasets:
        print("⏳ Chargement GTSRB...")
        _, loaders["gtsrb"] = get_gtsrb_loaders(batch_size=args.batch_size)

    all_results = []

    # ==========================================
    # 5. EXÉCUTION
    # ==========================================
    for sc in scenarios:
        print(f"\n🧠 Modèle : {sc['name']}")
        model = get_architecture(sc["arch"], device)

        # Chargement
        loaded = manage_checkpoint(model, args.storage_path, sc["file"], device=device)

        if loaded:
            stats = run_evaluation_suite(
                model=model,
                model_name=sc["name"],
                dataset_name=sc["data"],
                dataloader=loaders[sc["data"]],
                attack_configs=attack_suite,
                device=device,
            )
            all_results.extend(stats)
        else:
            print(
                f"⚠️ Fichier introuvable : {sc['file']} (Vérifie le dossier checkpoints)"
            )

    # ==========================================
    # 6. SAUVEGARDE
    # ==========================================
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n📊 RÉSULTATS :")
        try:
            print(df.pivot(index="Model", columns="Attack", values="Accuracy"))
        except:
            print(df)

        save_results(df, args.storage_path, args.output_filename)
    else:
        print("❌ Aucun résultat.")


if __name__ == "__main__":
    main()
