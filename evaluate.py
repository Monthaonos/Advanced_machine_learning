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
    """Factory to instantiate the model skeleton."""
    if arch_name == "cifar10":
        return SimpleCIFAR10CNN().to(device)
    if arch_name == "cifar10_large":
        return WideResNet().to(device)
    if arch_name == "gtsrb":
        return GTSRBModel().to(device)
    return None


def parse_args():
    """Handles CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced Robustness Benchmark"
    )

    # 1. Hardware Configuration
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 2. Path Configuration
    default_ckpt_root = os.getenv("CHECKPOINT_ROOT", "checkpoints")

    parser.add_argument(
        "--storage-path",
        type=str,
        default=default_ckpt_root,
        help="Root directory for input models.",
    )

    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="Directory to save the results CSV.",
    )

    parser.add_argument(
        "--output-filename", type=str, default="benchmark_results.csv"
    )

    # 3. Target and Naming
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["cifar10", "cifar10_large", "gtsrb"],
        help="Model architecture to evaluate.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Training prefix (e.g., 'cifar10_small'). Defaults to target name if empty.",
    )

    return parser.parse_args()


def run_evaluation(args):
    """
    Executes the evaluation pipeline using provided arguments.
    Can be called via CLI or external Python script.
    """
    device = torch.device(args.device)

    # Handle default prefix
    file_prefix = args.prefix if args.prefix else args.target

    # ==========================================
    # A. ATTACK CONFIGURATION
    # ==========================================
    # Standard perturbation budget and step size for CIFAR-10/GTSRB
    EPS = 8 / 255
    ALPHA = 2 / 255

    attack_suite = {
        "Clean": {"fn": None},
        "FGSM": {"fn": fgsm_attack, "kwargs": {"epsilon": EPS}},
        "PGD_10": {
            "fn": pgd_attack,
            "kwargs": {"epsilon": EPS, "alpha": ALPHA, "num_steps": 10},
        },
        # Momentum Iterative Method (MIM)
        # Uses momentum to stabilize update directions and escape local maxima
        "MIM_10": {
            "fn": mim_attack,
            "kwargs": {
                "epsilon": EPS,
                "alpha": ALPHA,
                "num_steps": 10,
                "decay": 1.0,  # Momentum decay factor (1.0 accumulates full history)
            },
        },
    }

    # ==========================================
    # B. DYNAMIC CATALOG
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

    print(f"🎯 Target: {args.target} with prefix '{file_prefix}'")

    # ==========================================
    # C. DATA LOADING
    # ==========================================
    loaders = {}
    dataset_name = scenarios[0]["data"]

    if dataset_name == "cifar10":
        print("⏳ Loading CIFAR-10...")
        _, loaders["cifar10"] = get_cifar10_loaders(batch_size=args.batch_size)
    elif dataset_name == "gtsrb":
        print("⏳ Loading GTSRB...")
        _, loaders["gtsrb"] = get_gtsrb_loaders(batch_size=args.batch_size)

    all_results = []

    # ==========================================
    # D. EXECUTION LOOP
    # ==========================================
    for sc in scenarios:
        print(f"\n🧠 Model: {sc['name']}")
        print(f"   File: {sc['file']}")

        model = get_architecture(sc["arch"], device)

        # Load model from storage
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
            print(f"⚠️ Skipped (File not found): {sc['file']}")

    # ==========================================
    # E. SAVING RESULTS
    # ==========================================
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n📊 Preview:")
        try:
            print(df.pivot(index="Model", columns="Attack", values="Accuracy"))
        except:
            print(df.head())

        print(f"\n💾 Saving to: {args.results_path}")
        save_results(df, args.results_path, args.output_filename)

    else:
        print("❌ No results to save.")


def main():
    """CLI Entry Point."""
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
