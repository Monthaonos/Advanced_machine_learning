"""
Main Research Orchestrator.
Central entry point for the adversarial robustness framework.
Coordinates Training (Phase 1), Evaluation (Phase 2), and Patch Analysis (Phase 3).
"""

import argparse
import os
from services.config_manager import load_config
from services.training import run_training
from services.evaluation import run_evaluation
from services.patch_service import run_patch_analysis


def setup_directories(config: dict, prefix: str) -> tuple[str, str]:
    """
    Ensures that experiment-specific directories exist for checkpoints and results.

    Args:
        config (dict): The loaded configuration dictionary.
        prefix (str): Experiment identifier used for directory naming.

    Returns:
        tuple[str, str]: Resolved paths for (storage_path, results_path).
    """
    # We strip any trailing slashes to ensure consistent path joining later
    storage_path = config["project"]["storage_path"].rstrip("/")
    results_path = "results"

    # Creation of the base directory tree
    # Downstream services will handle the creation of _prefix subfolders
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    return storage_path, results_path


def main():
    """
    Primary execution flow: parses CLI arguments, merges configuration,
    and executes requested pipeline phases.
    """
    # --- 1. Argument Parsing Strategy ---
    parser = argparse.ArgumentParser(
        description="Adversarial Robustness Framework - Unified Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution Phase Flags (Control flow)
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to TOML configuration file.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Execute Phase 1: Standard/Adversarial Training.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Execute Phase 2: Quantitative L-inf Evaluation.",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Execute Phase 3: Qualitative/Quantitative Patch Analysis.",
    )

    # Hardware & Paths Overrides
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Override root directory for model checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Force specific hardware device.",
    )

    # Data & Model Hyperparameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "gtsrb"],
        help="Override target dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple_cnn", "resnet18", "wideresnet"],
        help="Override architecture.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Experiment identifier (used for filenames and directories).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override training/evaluation batch size.",
    )

    # Training Overrides
    parser.add_argument(
        "--epochs", type=int, help="Override number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Override optimizer learning rate."
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing weights and force training.",
    )

    # Adversarial Parameter Overrides (L-infinity budget)
    parser.add_argument(
        "--epsilon", type=float, help="Override max L-inf perturbation budget."
    )
    parser.add_argument(
        "--alpha", type=float, help="Override step size for iterative attacks."
    )
    parser.add_argument(
        "--num-steps", type=int, help="Override number of attack iterations."
    )

    # --- 2. Configuration & Workspace Resolution ---
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"[-] Error: Configuration file '{args.config}' not found.")
        return

    # Resolve prefix and workspace hierarchy (CLI takes priority over TOML)
    current_prefix = args.prefix or config["model"].get("prefix", "default")
    resolved_storage, resolved_results = setup_directories(
        config, current_prefix
    )

    # Synchronize resolved paths back into args for downstream service consumption
    args.storage_path = resolved_storage
    args.results_path = resolved_results

    # --- 3. Pipeline Execution (Sequential Orchestration) ---
    # Phase 1: Model Optimization
    if args.train:
        print(
            f"\n🚀 [PIPELINE] Starting Phase 1: Training (Prefix: {current_prefix})"
        )
        run_training(args, config)

    # Phase 2: Formal Robustness Benchmarking
    if args.eval:
        print(
            f"\n📊 [PIPELINE] Starting Phase 2: Evaluation (Prefix: {current_prefix})"
        )
        run_evaluation(args, config)

    # Phase 3: Vulnerability Analysis (Localized Attacks)
    if args.patch:
        print(
            f"\n🛡️ [PIPELINE] Starting Phase 3: Patch Analysis (Prefix: {current_prefix})"
        )
        run_patch_analysis(args, config)


if __name__ == "__main__":
    main()
