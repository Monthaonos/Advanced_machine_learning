import os
from types import SimpleNamespace
from services.train import run_training
from evaluate import run_evaluation
from plot_results import run_plotting

# --- Configuration Globale ---
STORAGE_PATH = "checkpoints"  # Peut être remplacé par "s3://votre-bucket/..."
RESULTS_PATH = "results"
PLOTS_DIR = "plots"
DEVICE = "cuda"

# Hyperparamètres
EPOCHS_BASELINE = 50
EPOCHS_WIDERESNET = 100
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 128


def get_file_path(base_dir, filename):
    """Gère la construction de chemin compatible S3 et Local."""
    if base_dir.startswith("s3://"):
        return f"{base_dir.rstrip('/')}/{filename}"
    return os.path.join(base_dir, filename)


def run_pipeline():
    print("[INFO] Initializing Benchmark Pipeline...")

    # =========================================================================
    # 1. ENTRAÎNEMENT : Baseline (Small CNN)
    # =========================================================================
    prefix_small = "cifar10_baseline"
    print(f"\n[INFO] Starting training: {prefix_small}")

    args_train_small = SimpleNamespace(
        model_type="cifar10",
        prefix=prefix_small,
        epochs=EPOCHS_BASELINE,
        batch_size=BATCH_SIZE_TRAIN,
        learning_rate=0.001,  # Adam default
        device=DEVICE,
        storage_path=STORAGE_PATH,
        force_retrain=False,
        # Paramètres par défaut requis par le pipeline
        epsilon=8 / 255,
        prob=1.0,
        alpha=2 / 255,
        num_steps=10,
        random_start=True,
    )
    run_training(args_train_small)

    # =========================================================================
    # 2. ENTRAÎNEMENT : WideResNet (Large Model)
    # =========================================================================
    prefix_large = "cifar10_wideresnet"
    print(f"\n[INFO] Starting training: {prefix_large}")

    args_train_large = SimpleNamespace(
        model_type="cifar10_large",
        prefix=prefix_large,
        epochs=EPOCHS_WIDERESNET,
        batch_size=BATCH_SIZE_TRAIN,
        learning_rate=0.1,  # SGD requires higher LR
        device=DEVICE,
        storage_path=STORAGE_PATH,
        force_retrain=False,
        epsilon=8 / 255,
        prob=1.0,
        alpha=2 / 255,
        num_steps=10,
        random_start=True,
    )
    run_training(args_train_large)

    # =========================================================================
    # 3. ÉVALUATION : Baseline
    # =========================================================================
    csv_small = f"{prefix_small}_results.csv"
    print(f"\n[INFO] Evaluating: {prefix_small}")

    args_eval_small = SimpleNamespace(
        target="cifar10",
        prefix=prefix_small,
        batch_size=BATCH_SIZE_EVAL,
        device=DEVICE,
        storage_path=STORAGE_PATH,
        results_path=RESULTS_PATH,
        output_filename=csv_small,
    )
    run_evaluation(args_eval_small)

    # =========================================================================
    # 4. ÉVALUATION : WideResNet
    # =========================================================================
    csv_large = f"{prefix_large}_results.csv"
    print(f"\n[INFO] Evaluating: {prefix_large}")

    args_eval_large = SimpleNamespace(
        target="cifar10_large",
        prefix=prefix_large,
        batch_size=BATCH_SIZE_EVAL,
        device=DEVICE,
        storage_path=STORAGE_PATH,
        results_path=RESULTS_PATH,
        output_filename=csv_large,
    )
    run_evaluation(args_eval_large)

    # =========================================================================
    # 5. GÉNÉRATION DES GRAPHIQUES (Comparaison)
    # =========================================================================
    print("\n[INFO] Generating comparison plots...")

    path_small = get_file_path(RESULTS_PATH, csv_small)
    path_large = get_file_path(RESULTS_PATH, csv_large)

    args_plot = SimpleNamespace(
        input_files=[path_small, path_large],
        output_dir=PLOTS_DIR,
        plot_name="benchmark_comparison",
    )
    run_plotting(args_plot)

    print("\n[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
