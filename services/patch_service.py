import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from services.attacks import UniversalPatchAttack
from services.storage_manager import manage_checkpoint, save_results
from services.dataloaders.factory import get_dataloaders
from services.models.factory import get_model


def run_patch_analysis(args, config):
    """
    Executes the comprehensive L-0 attack pipeline.

    Workflow:
    1. Resource Setup: Load DataLoaders and model architectures.
    2. Checkpoint Management: Load existing weights for clean and robust variants.
    3. Patch Optimization: Train a new universal patch or load an existing one.
    4. Quantitative Evaluation & Persistence: Compute and save ASR metrics.
    5. Qualitative Visualization: Plot and save sample adversarial grids.
    """
    # Device resolution priority: CLI Argument > Config File > Auto-detect
    device = torch.device(
        args.device
        if args.device
        else (
            config["project"]["device"] if torch.cuda.is_available() else "cpu"
        )
    )

    # Path and Metadata resolution
    storage_path = args.storage_path or config["project"]["storage_path"]
    dataset = args.dataset or config["data"]["dataset"]
    arch = args.model or config["model"]["architecture"]
    prefix = config["model"].get("prefix", "test2")

    # --- 1. Resource Setup ---
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=config["data"].get("batch_size", 32)
    )

    model_clean = get_model(arch, dataset, num_classes, in_channels).to(device)
    model_robust = get_model(arch, dataset, num_classes, in_channels).to(
        device
    )

    manage_checkpoint(
        model_clean,
        storage_path,
        f"{prefix}_{dataset}_{arch}_clean.pth",
        device,
    )
    manage_checkpoint(
        model_robust,
        storage_path,
        f"{prefix}_{dataset}_{arch}_robust.pth",
        device,
    )

    # --- 2. Patch Optimization / Loading ---
    patch_handler = UniversalPatchAttack(model_clean, config["patch_attack"])
    patch_file = f"{prefix}_{dataset}_{arch}_patch.pth"
    patch_path = os.path.join(storage_path, patch_file)

    if not os.path.exists(patch_path) or args.force_retrain:
        print(f"[*] Training new Universal Patch for {arch}...")
        patch_handler.train_patch(train_loader)
        torch.save(patch_handler.patch.data, patch_path)
    else:
        print(f"[*] Loading existing patch from {patch_path}")
        patch_handler.patch.data = torch.load(patch_path).to(device)

    # --- 3. Quantitative Evaluation & Persistence ---
    print("\n📊 Starting Quantitative Patch Analysis...")
    asr_clean = _calculate_asr(patch_handler, model_clean, test_loader, device)
    asr_robust = _calculate_asr(
        patch_handler, model_robust, test_loader, device
    )

    # Prepare statistics for standardized reporting
    patch_stats = [
        {
            "Model": f"{arch}_Clean",
            "Dataset": dataset,
            "Attack": "Universal_Patch",
            "ASR": asr_clean,
        },
        {
            "Model": f"{arch}_Robust",
            "Dataset": dataset,
            "Attack": "Universal_Patch",
            "ASR": asr_robust,
        },
    ]

    df = pd.DataFrame(patch_stats)

    # Consistent output directory resolution (maps 'checkpoints' to 'results')
    results_path = storage_path.replace("checkpoints", "results")
    os.makedirs(results_path, exist_ok=True)

    output_filename = f"{prefix}_{dataset}_{arch}_patch_report.csv"
    save_results(df, results_path, output_filename)

    print(f"✅ Patch report saved to: {results_path}/{output_filename}")

    # --- 4. Qualitative Visualization ---
    print("\n🎨 Generating qualitative grid visualization...")
    _plot_results(
        patch_handler,
        model_clean,
        model_robust,
        test_loader,
        config,
        results_path,
    )


def _calculate_asr(handler, model, loader, device):
    """
    Calculates the Attack Success Rate (ASR).
    Formula: ASR = (Successful Attacks) / (Samples initially correctly classified).
    """
    model.eval()
    total_initially_correct = 0
    successful_attacks = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating ASR", leave=False):
            images, labels = images.to(device), labels.to(device)

            clean_preds = model(images).argmax(dim=1)
            correct_mask = clean_preds == labels

            num_correct = correct_mask.sum().item()
            if num_correct == 0:
                continue

            total_initially_correct += num_correct

            patched_images = handler.apply_patch(images[correct_mask])
            patched_preds = model(patched_images).argmax(dim=1)

            successful_attacks += (
                (patched_preds != labels[correct_mask]).sum().item()
            )

    return (
        (successful_attacks / total_initially_correct) * 100
        if total_initially_correct > 0
        else 0.0
    )


def _plot_results(
    handler, m_clean, m_robust, loader, config, results_path=None
):
    """
    Generates a 1x4 qualitative grid and saves it as a PNG file.
    """
    m_clean.eval()
    m_robust.eval()

    images, labels = next(iter(loader))
    images, labels = (
        images[:4].to(handler.device),
        labels[:4].to(handler.device),
    )

    patched = handler.apply_patch(images)
    with torch.no_grad():
        preds_c = m_clean(patched).argmax(dim=1)
        preds_r = m_robust(patched).argmax(dim=1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for i in range(4):
        img = np.transpose(patched[i].detach().cpu().numpy(), (1, 2, 0))
        axes[i].imshow(np.clip(img, 0, 1))
        color = "green" if preds_r[i] == labels[i] else "red"
        axes[i].set_title(
            f"GT: {labels[i].item()}\nStd: {preds_c[i].item()}\nRob: {preds_r[i].item()}",
            color=color,
            fontsize=12,
            fontweight="bold",
        )
        axes[i].axis("off")

    plt.suptitle(
        f"Qualitative Analysis: Universal Patch Attack ({config['data']['dataset']})",
        fontsize=16,
    )
    plt.tight_layout()

    if results_path:
        prefix = config["model"].get("prefix", "test")
        dataset = config["data"]["dataset"]
        arch = config["model"]["architecture"]
        fig_path = os.path.join(
            results_path, f"{prefix}_{dataset}_{arch}_patch_viz.png"
        )
        plt.savefig(fig_path, dpi=300)
        print(f"🖼️ Visualization saved to: {fig_path}")

    plt.show()
