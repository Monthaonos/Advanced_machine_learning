"""
Main Research Orchestrator.
Coordinates the three phases of the adversarial robustness pipeline.
"""

import argparse
import os
from services.config_manager import load_config
from services.training import run_training
from services.evaluation import run_evaluation
from services.patch_service import run_patch_analysis


def setup_directories(config, prefix):
    """
    Ensures that experiment-specific directories exist.
    Naming convention: <base_path>_<prefix>
    """
    storage_path = f"{config['project']['storage_path']}_{prefix}"
    results_path = f"results_{prefix}"

    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    return storage_path, results_path


def main():
    # 1. Initialize Argument Parser
    parser = argparse.ArgumentParser(
        description="Adversarial Robustness Framework - Unified Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Execution Phase Flags ---
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to TOML config.",
    )
    parser.add_argument(
        "--train", action="store_true", help="Execute Phase 1: Training."
    )
    parser.add_argument(
        "--eval", action="store_true", help="Execute Phase 2: Evaluation."
    )
    parser.add_argument(
        "--patch", action="store_true", help="Execute Phase 3: Patch Analysis."
    )

    # --- Hardware & Paths Overrides ---
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Override root directory for checkpoints.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Force device.",
    )

    # --- Data & Model Overrides ---
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "gtsrb"],
        help="Override dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple_cnn", "resnet18", "wideresnet"],
        help="Override architecture.",
    )
    parser.add_argument(
        "--prefix", type=str, help="Override filename prefix (e.g., test2)."
    )
    parser.add_argument("--batch-size", type=int, help="Override batch size.")

    # --- Hyperparameters Overrides ---
    parser.add_argument(
        "--epochs", type=int, help="Override number of training epochs."
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Override learning rate."
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints.",
    )

    # --- Adversarial Overrides ---
    parser.add_argument(
        "--epsilon", type=float, help="Override max L-inf perturbation."
    )
    parser.add_argument("--alpha", type=float, help="Override step size.")
    parser.add_argument(
        "--num-steps", type=int, help="Override attack iterations."
    )

    # 2. Load Configuration and Sync with CLI
    args = parser.parse_args()
    config = load_config(args.config)

    # Resolve prefix and setup directories
    # Priority: CLI > TOML > Default
    current_prefix = args.prefix or config["model"].get("prefix", "default")
    resolved_storage, resolved_results = setup_directories(
        config, current_prefix
    )

    # Inject resolved paths back into args for services to use
    args.storage_path = resolved_storage
    args.results_path = resolved_results

    # 3. Conditional Execution of Research Phases
    if args.train:
        print(
            f"\n🚀 [PIPELINE] Starting Phase 1: Training (Prefix: {current_prefix})"
        )
        run_training(args, config)

    if args.eval:
        print(
            f"\n📊 [PIPELINE] Starting Phase 2: Evaluation (Prefix: {current_prefix})"
        )
        # run_evaluation can now use args.results_path to save CSVs
        run_evaluation(args, config)

    if args.patch:
        print(
            f"\n🛡️ [PIPELINE] Starting Phase 3: Patch Analysis (Prefix: {current_prefix})"
        )
        run_patch_analysis(args, config)


if __name__ == "__main__":
    main()
