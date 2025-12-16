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
    if arch_name == "cifar10_large":
        return WideResNet().to(device)
    if arch_name == "gtsrb":
        return GTSRBModel().to(device)
    return None


def parse_args():
    """Gère uniquement la lecture des arguments depuis le terminal."""
    parser = argparse.ArgumentParser(
        description="Advanced Robustness Benchmark"
    )

    # 1. Configuration Matérielle
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 2. Configuration des Chemins
    default_ckpt_root = os.getenv("CHECKPOINT_ROOT", "checkpoints")

    parser.add_argument(
        "--storage-path",
        type=str,
        default=default_ckpt_root,
        help="Racine où sont stockés les modèles (Input).",
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Dossier où sauvegarder le CSV (Output).",
    )

    parser.add_argument(
        "--output-filename", type=str, default="benchmark_results.csv"
    )

    # 3. Cible et Noms
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["cifar10", "cifar10_large", "gtsrb"],
        help="Quelle architecture évaluer ?",
    )

    # --- Argument Prefix ---
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Le préfixe utilisé lors de l'entraînement (ex: 'cifar10_small'). Si vide, utilise le nom de la target.",
    )

    return parser.parse_args()


# --- CHANGEMENT ICI : On isole la logique dans run_evaluation ---
def run_evaluation(args):
    """
    Exécute l'évaluation en utilisant un objet 'args' (Namespace).
    Peut être appelé depuis le terminal OU depuis un autre script Python.
    """
    device = torch.device(args.device)

    # Gestion du préfixe par défaut
    file_prefix = args.prefix if args.prefix else args.target

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
    # B. CATALOGUE DYNAMIQUE
    # ==========================================
    scenarios = [
        {
            "arch": args.target,
            "data": "cifar10" if "cifar10" in args.target else "gtsrb",
            "file": f"{file_prefix}_clean.pth",
            "name": f"{file_prefix} (Standard)",
        },
        {
            "arch": args.target,
            "data": "cifar10" if "cifar10" in args.target else "gtsrb",
            "file": f"{file_prefix}_robust.pth",
            "name": f"{file_prefix} (Robust)",
        },
    ]

    print(f"🎯 Cible : {args.target} avec préfixe '{file_prefix}'")

    # ==========================================
    # C. CHARGEMENT DONNÉES
    # ==========================================
    loaders = {}
    dataset_name = scenarios[0]["data"]

    if dataset_name == "cifar10":
        print("⏳ Chargement CIFAR-10...")
        _, loaders["cifar10"] = get_cifar10_loaders(batch_size=args.batch_size)
    elif dataset_name == "gtsrb":
        print("⏳ Chargement GTSRB...")
        _, loaders["gtsrb"] = get_gtsrb_loaders(batch_size=args.batch_size)

    all_results = []

    # ==========================================
    # D. EXÉCUTION
    # ==========================================
    for sc in scenarios:
        print(f"\n🧠 Modèle : {sc['name']}")
        print(f"   Fichier cherché : {sc['file']}")

        model = get_architecture(sc["arch"], device)

        # On lit depuis storage-path (INPUT)
        loaded = manage_checkpoint(
            model, args.storage_path, sc["file"], device=device
        )

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
    # E. SAUVEGARDE
    # ==========================================
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n📊 Aperçu :")
        try:
            print(df.pivot(index="Model", columns="Attack", values="Accuracy"))
        except:
            print(df.head())

        print(f"\n💾 Sauvegarde vers : {args.results_path}")
        save_results(df, args.results_path, args.output_filename)

    else:
        print("❌ Aucun résultat à sauvegarder.")


def main():
    """Point d'entrée CLI : Lit les args du terminal et lance le cuisinier."""
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
