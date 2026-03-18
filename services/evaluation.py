"""
Phase 2: Quantitative robustness evaluation under L-infinity attacks.

Can be run standalone:
    python -m services.evaluation --dataset gtsrb --model resnet18
"""

import argparse
import torch
import os
import pandas as pd
from typing import Dict, Any, List

from services.config_manager import load_config
from services.storage_manager import manage_checkpoint, save_results
from services.dataloaders.factory import get_dataloaders
from services.models.factory import get_model
from services.evaluator import run_evaluation_suite
from services.attacks import fgsm_attack, pgd_attack, mim_attack


def build_attack_suite(
    attack_names: List[str], adv_config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Construct attack configurations from a list of attack names."""
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
    Evaluate clean and robust model variants against an attack suite.

    Loads both model variants, runs each attack, and exports a comparison report.
    """
    dataset = args.dataset or config["data"]["dataset"]
    model_name = args.model or config["model"]["architecture"]
    batch_size = args.batch_size or config["evaluation"].get("batch_size", 128)
    storage_path = args.storage_path or config["project"]["storage_path"]

    device_name = args.device or config["project"].get("device", "cpu")
    device = torch.device(
        device_name
        if torch.cuda.is_available() or device_name in ["cpu", "mps"]
        else "cpu"
    )

    prefix = (
        args.prefix
        or config["model"].get("prefix")
        or f"{dataset}_{model_name}"
    )

    results_path = storage_path.replace("checkpoints", "results")
    os.makedirs(results_path, exist_ok=True)

    print(f"[*] Evaluation Pipeline")
    print(f"    Dataset: {dataset} | Model: {model_name}")
    print(f"    Prefix:  {prefix} | Device: {device}")

    _, test_loader, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=batch_size
    )

    adv_conf = config["adversarial"].copy()
    if args.epsilon is not None:
        adv_conf["epsilon"] = args.epsilon
    if args.num_steps is not None:
        adv_conf["num_steps"] = args.num_steps

    requested_attacks = config["evaluation"].get(
        "attacks_to_run", ["Clean", "PGD"]
    )
    attack_suite = build_attack_suite(requested_attacks, adv_conf)

    all_results = []

    for variant in ["clean", "robust"]:
        print(f"\n[*] Evaluating variant: {variant.upper()}")

        model = get_model(model_name, dataset, num_classes, in_channels).to(
            device
        )

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
            print(f"[!] {filename} not found in {storage_path}. Skipping.")

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n[+] Results Summary (Accuracy %):")
        try:
            print(df.pivot(index="Model", columns="Attack", values="Accuracy"))
        except Exception:
            print(df.head())

        output_filename = (
            f"{prefix}_{dataset}_{model_name}_evaluation_report.csv"
        )
        save_results(df, results_path, output_filename)
        print(f"[+] Report saved to: {results_path}/{output_filename}")


def parse_args():
    """Parse CLI arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Robustness Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--storage-path", type=str, help="Root checkpoint directory.")
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], help="Computation device."
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "gtsrb"], help="Target dataset."
    )
    parser.add_argument("--model", type=str, help="Target architecture.")
    parser.add_argument("--prefix", type=str, help="Experiment identifier.")
    parser.add_argument("--batch-size", type=int, help="Inference batch size.")
    parser.add_argument("--epsilon", type=float, help="Perturbation budget (L-inf).")
    parser.add_argument("--num-steps", type=int, help="Attack iteration count.")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        config_data = load_config("config.toml")
    except Exception as e:
        print(f"[!] Error loading configuration: {e}")
        exit(1)

    args_data = parse_args()
    run_evaluation(args_data, config_data)
