import argparse
import torch
import os
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
    Dynamically constructs the attack configurations based on merged settings.
    """
    suite = {}
    eps = adv_config["epsilon"]
    alpha = adv_config["alpha"]
    steps = adv_config["num_steps"]

    if "Clean" in attack_names:
        suite["Clean"] = {"fn": None, "kwargs": {}}

    if "FGSM" in attack_names:
        suite["FGSM"] = {"fn": fgsm_attack, "kwargs": {"epsilon": eps}}

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
    Orchestrates the evaluation pipeline with CLI > TOML priority.
    """
    # 1. Merge Configuration
    dataset = args.dataset or config["data"]["dataset"]
    model_name = args.model or config["model"]["architecture"]
    batch_size = args.batch_size or config["evaluation"].get("batch_size", 128)
    storage_path = args.storage_path or config["project"]["storage_path"]

    # Device management
    device_name = args.device or config["project"].get("device", "cpu")
    device = torch.device(
        device_name
        if torch.cuda.is_available() or device_name == "cpu"
        else "cpu"
    )

    # Prefix management
    prefix = (
        args.prefix
        or config["model"].get("prefix")
        or f"{dataset}_{model_name}"
    )

    # Results directory management (AML/results)
    results_path = storage_path.replace("checkpoints", "results")
    os.makedirs(results_path, exist_ok=True)

    print(f"📊 Starting Evaluation Pipeline")
    print(f"   Dataset: {dataset} | Model: {model_name}")
    print(f"   Prefix:  {prefix} | Device: {device}")

    # 2. Data Loading (Test set only)
    _, test_loader, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=batch_size
    )

    # 3. Build Attack Suite with Merged Adversarial Hyperparameters
    adv_conf = config["adversarial"].copy()
    if args.epsilon is not None:
        adv_conf["epsilon"] = args.epsilon
    if args.num_steps is not None:
        adv_conf["num_steps"] = args.num_steps

    requested_attacks = config["evaluation"].get(
        "attacks_to_run", ["Clean", "PGD"]
    )
    attack_suite = build_attack_suite(requested_attacks, adv_conf)

    # 4. Evaluation Loop
    all_results = []
    variants = ["clean", "robust"]

    for variant in variants:
        print(f"\n🔹 Evaluating Variant: {variant.upper()}")
        model = get_model(model_name, dataset, num_classes, in_channels).to(
            device
        )

        # Build filename: test2_cifar10_resnet18_clean.pth
        filename = f"{prefix}_{dataset}_{model_name}_{variant}.pth"

        if manage_checkpoint(model, storage_path, filename, device):
            stats = run_evaluation_suite(
                model=model,
                model_name=f"{model_name}_{variant.capitalize()}",
                dataset_name=dataset,
                dataloader=test_loader,
                attack_configs=attack_suite,
                device=device,
            )
            all_results.extend(stats)
        else:
            print(
                f"⚠️ Warning: {filename} not found in {storage_path}. Skipping."
            )

    # 5. Save Aggregated Results to AML/results
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n📝 Results Summary:")
        try:
            print(df.pivot(index="Model", columns="Attack", values="Accuracy"))
        except:
            print(df.head())

        output_filename = (
            f"{prefix}_{dataset}_{model_name}_evaluation_report.csv"
        )
        save_results(df, results_path, output_filename)
        print(f"✅ Report saved to: {results_path}/{output_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Evaluation Service")

    # Hardware & Paths
    parser.add_argument(
        "--storage-path", type=str, help="Path to checkpoints."
    )
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"])

    # Model & Data
    parser.add_argument("--dataset", type=str, choices=["cifar10", "gtsrb"])
    parser.add_argument("--model", type=str, help="e.g., resnet18, wideresnet")
    parser.add_argument(
        "--prefix", type=str, help="Filename prefix (e.g., test2)"
    )
    parser.add_argument("--batch-size", type=int)

    # Attack Hyperparameters
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--num-steps", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        config_data = load_config("config.toml")
    except Exception as e:
        print(f"⚠️ Error loading config.toml: {e}")
        exit(1)

    args_data = parse_args()
    run_evaluation(args_data, config_data)
