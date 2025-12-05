import argparse
import os
import torch
import pandas as pd
from dotenv import load_dotenv

# Services (C'est eux qui gèrent la complexité S3 vs Local)
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
    """Factory pour instancier le squelette du modèle vide."""
    if arch_name == "cifar10":
        return SimpleCIFAR10CNN().to(device)
    if arch_name == "cifar10_large" or arch_name == "cifar10_random":
        return WideResNet().to(device)
    if arch_name == "gtsrb":
        return GTSRBModel().to(device)
    return None


def main():
    parser = argparse.ArgumentParser(description="Advanced Robustness Benchmark")

    # 1. Configuration Matérielle
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. Configuration des Chemins (DISSOCIÉS)
    default_ckpt_root = os.getenv("CHECKPOINT_ROOT", "checkpoints")

    # INPUT : Où chercher les modèles ?
    parser.add_argument(
        "--storage-path",
        type=str,
        default=default_ckpt_root,
        help="Racine où sont stockés les modèles (Input). Ex: s3://bucket/checkpoints",
    )

    # OUTPUT : Où mettre le CSV ?
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,  # J'ai mis requis pour t'obliger à choisir, tu peux mettre default="results"
        help="Dossier où sauvegarder le CSV (Output). Ex: s3://bucket/results",
    )

    parser.add_argument("--output-filename", type=str, default="benchmark_results.csv")

    # 3. Cible (Simplifié sans 'all')
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["cifar10", "cifar10_large", "gtsrb"],
        help="Quelle architecture évaluer ?",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # ==========================================
    # A. CONFIGURATION DES ATTAQUES
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
    }

    # ==========================================
    # B. CATALOGUE (Chemins RELATIFS à storage-path)
    # ==========================================
    # Ici, "file" doit être la fin du chemin après storage-path
    full_catalog = [
        # --- GTSRB ---
        {
            "arch": "gtsrb",
            "data": "gtsrb",
            "file": "gtsrb/gtsrb_clean.pth",  # ex: storage-path/gtsrb/gtsrb_clean.pth
            "name": "GTSRB (Standard)",
        },
        {
            "arch": "gtsrb",
            "data": "gtsrb",
            "file": "gtsrb/gtsrb_robust.pth",
            "name": "GTSRB (Robust)",
        },
        # --- CIFAR-10 ---
        {
            "arch": "cifar10",
            "data": "cifar10",
            "file": "cifar_10/cifar10_clean.pth",  # Attention à bien vérifier ce nom de dossier sur S3
            "name": "Cifar10 Small (Standard)",
        },
        {
            "arch": "cifar10",
            "data": "cifar10",
            "file": "cifar_10/cifar10_robust.pth",
            "name": "Cifar10 Small (Robust)",
        },
        # --- CIFAR-10 Large ---
        {
            "arch": "cifar10_large",
            "data": "cifar10",
            "file": "cifar_10_large/cifar10_large_clean.pth",
            "name": "WideResNet (Standard)",
        },
        {
            "arch": "cifar10_random",
            "data": "cifar10",
            "file": "cifar_10_random/cifar10_large_robust.pth",
            "name": "WideResNet (Robust)",
        },
        {
            "arch": "cifar10_random",
            "data": "cifar10",
            "file": "cifar_10_random/cifar10_random_clean.pth",
            "name": "WideResNet (Standard)",
        },
        {
            "arch": "cifar10_random",
            "data": "cifar10",
            "file": "cifar_10_large/cifar10_random_robust.pth",
            "name": "WideResNet (Robust)",
        },
    ]

    # Filtrage simple (plus de "all")
    scenarios = [s for s in full_catalog if s["arch"] == args.target]
    print(f"🎯 Cible : {args.target} ({len(scenarios)} modèles trouvés)")

    # ==========================================
    # C. CHARGEMENT DONNÉES
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
    # D. EXÉCUTION
    # ==========================================
    for sc in scenarios:
        print(f"\n🧠 Modèle : {sc['name']}")
        model = get_architecture(sc["arch"], device)

        # UTILISATION 1 : On lit depuis storage-path (INPUT)
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
            print(f"⚠️ Ignoré (Fichier introuvable) : {sc['file']}")

    # ==========================================
    # E. SAUVEGARDE (DISSOCIÉE)
    # ==========================================
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n📊 Aperçu :")
        try:
            print(df.pivot(index="Model", columns="Attack", values="Accuracy"))
        except:
            print(df.head())

        # UTILISATION 2 : On écrit dans results-path (OUTPUT)
        print(f"\n💾 Sauvegarde vers : {args.results_path}")
        save_results(df, args.results_path, args.output_filename)

    else:
        print("❌ Aucun résultat à sauvegarder.")


if __name__ == "__main__":
    main()
