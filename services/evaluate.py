"""
Unified Evaluation Service.
Runs evaluation metrics on Clean and Robust models defined in config.toml.

It leverages the TOML configuration to dynamically build the attack suite
(FGSM, PGD, MIM, etc.) and handles data loading via factories.

Usage:
    python -m services.evaluation --model resnet18 --dataset cifar10
"""

import argparse
import torch
import pandas as pd
from typing import Dict, Any, List

# --- Modular Imports ---
from services.config_manager import load_config
from services.storage_manager import manage_checkpoint, save_results
from services.dataloaders.factory import get_dataloaders
from services.models.factory import get_model
from services.evaluator import run_evaluation_suite
from services.attacks import fgsm_attack, pgd_attack, mim_attack


def build_attack_suite(
    attack_names: List[str], adv_config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically constructs the dictionary of attack configurations based on TOML settings.

    Args:
        attack_names (List[str]): List of attacks to run (e.g., ["Clean", "PGD"]).
        adv_config (dict): The 'adversarial' section from config.toml containing epsilon, alpha, etc.

    Returns:
        Dict: A dictionary compatible with run_evaluation_suite.
    """
    suite = {}
    eps = adv_config["epsilon"]
    alpha = adv_config["alpha"]
    steps = adv_config["num_steps"]

    if "Clean" in attack_names:
        suite["Clean"] = {"fn": None, "kwargs": {}}

    if "FGSM" in attack_names:
        suite["FGSM"] = {
            "fn": fgsm_attack,
            "kwargs": {"epsilon": eps},
        }

    if "PGD" in attack_names:
        suite["PGD"] = {
            "fn": pgd_attack,
            "kwargs": {"epsilon": eps, "alpha": alpha, "num_steps": steps},
        }

    if "MIM" in attack_names:
        suite["MIM"] = {
            "fn": mim_attack,
            "kwargs": {
                "epsilon": eps,
                "alpha": alpha,
                "num_steps": steps,
                "decay": 1.0,
            },
        }

    return suite


def run_evaluation(args: argparse.Namespace, config: Dict[str, Any]):
    """
    Main execution logic for the evaluation pipeline.

    1. Loads Configuration (CLI overrides TOML).
    2. Loads Test Data.
    3. Builds Attack Suite.
    4. Evaluates Clean and Robust models found in storage.
    5. Saves aggregated results to CSV.
    """
    # 1. Merge Configuration
    dataset = args.dataset or config["data"]["dataset"]
    model_name = args.model or config["model"]["architecture"]
    batch_size = args.batch_size or config["evaluation"]["batch_size"]
    storage_path = args.storage_path or config["project"]["storage_path"]

    device_name = args.device or config["project"].get("device", "cpu")
    device = torch.device(
        device_name
        if torch.cuda.is_available() or device_name == "cpu"
        else "cpu"
    )

    # Prefix management
    prefix = args.prefix or config["model"].get("prefix")
    if not prefix:
        prefix = f"{dataset}_{model_name}"

    print(f"📊 Starting Evaluation Pipeline")
    print(f"   Dataset: {dataset} | Model: {model_name}")
    print(f"   Prefix:  {prefix}")
    print(f"   Device:  {device}")

    # 2. Data Loading (Test set only)
    print("📦 Loading Test Data...")
    _, test_loader, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=batch_size
    )

    # 3. Build Attack Suite from Config
    requested_attacks = config["evaluation"].get("attacks_to_run", ["Clean"])
    # Allow CLI override for epsilon/steps if provided, else use TOML
    adv_conf = config["adversarial"].copy()
    if args.epsilon:
        adv_conf["epsilon"] = args.epsilon
    if args.num_steps:
        adv_conf["num_steps"] = args.num_steps

    attack_suite = build_attack_suite(requested_attacks, adv_conf)
    print(f"⚔️  Attacks scheduled: {list(attack_suite.keys())}")

    # ==========================================
    # A. Evaluate Standard (Clean) Model
    # ==========================================
    print("\n🔹 Evaluating Standard (Clean) Model...")
    model_clean = get_model(model_name, dataset, num_classes, in_channels).to(
        device
    )

    clean_filename = f"{prefix}_{dataset}_{model_name}_clean.pth"

    if manage_checkpoint(model_clean, storage_path, clean_filename, device):
        results_clean = run_evaluation_suite(
            model=model_clean,
            model_name=f"{model_name}_Clean",
            dataset_name=dataset,
            dataloader=test_loader,
            attack_configs=attack_suite,
            device=device,
        )
    else:
        print(f"⚠️ Warning: {clean_filename} not found. Skipping.")
        results_clean = []

    # ==========================================
    # B. Evaluate Robust (Adversarial) Model
    # ==========================================
    print("\n🔹 Evaluating Robust (Adversarial) Model...")
    model_robust = get_model(model_name, dataset, num_classes, in_channels).to(
        device
    )

    robust_filename = f"{prefix}_{dataset}_{model_name}_robust.pth"

    if manage_checkpoint(model_robust, storage_path, robust_filename, device):
        results_robust = run_evaluation_suite(
            model=model_robust,
            model_name=f"{model_name}_Robust",
            dataset_name=dataset,
            dataloader=test_loader,
            attack_configs=attack_suite,
            device=device,
        )
    else:
        print(f"⚠️ Warning: {robust_filename} not found. Skipping.")
        results_robust = []

    # ==========================================
    # C. Save Aggregated Results
    # ==========================================
    all_results = results_clean + results_robust

    if all_results:
        df = pd.DataFrame(all_results)

        # Display a nice preview
        print("\n📝 Results Summary:")
        # Pivot table for better readability in console (Model vs Attack)
        try:
            preview = df.pivot(
                index="Model", columns="Attack", values="Accuracy"
            )
            print(preview)
        except Exception:
            print(df[["Model", "Attack", "Accuracy"]])

        # Save to CSV
        output_filename = f"{prefix}_{dataset}_evaluation_report.csv"
        save_results(df, storage_path, output_filename)
    else:
        print("\n❌ No results collected (Models not found).")


def parse_args():
    """
    Parses command line arguments to optionally override TOML configuration.
    """
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Service. Runs attacks on trained models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Project & Hardware
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Root directory where checkpoints are stored (Local or S3). Defaults to [project.storage_path].",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Computation device. Defaults to [project.device].",
    )

    # 2. Model & Data Selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "gtsrb"],
        help="Target dataset. Defaults to [data.dataset].",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model architecture (e.g., 'resnet18'). Defaults to [model.architecture].",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Specific filename prefix (e.g. 'exp_v1'). If None, uses {dataset}_{model}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Evaluation batch size. Defaults to [evaluation.batch_size].",
    )

    # 3. Adversarial Parameters (Specific to PGD/MIM/FGSM)
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Maximum perturbation budget (L-inf norm). Defaults to [adversarial.epsilon].",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of iterations for iterative attacks (PGD/MIM). Defaults to [adversarial.num_steps].",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 1. Load TOML
    try:
        config = load_config("config.toml")
    except Exception as e:
        print(f"⚠️ Error loading config.toml: {e}")
        exit(1)

    # 2. Parse CLI
    args = parse_args()

    # 3. Run
    run_evaluation(args, config)
