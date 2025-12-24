"""
Unified Adversarial Patch Service.
Handles optimization, quantitative ASR evaluation, and qualitative grid visualization.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from services.attacks import UniversalPatchAttack
from services.storage_manager import manage_checkpoint
from services.dataloaders.factory import get_dataloaders
from services.models.factory import get_model


def run_patch_analysis(args, config):
    """
    Executes the full L-0 attack pipeline:
    1. Setup -> 2. Optimization -> 3. Quantitative ASR -> 4. Visualization
    """
    device = torch.device(
        args.device
        if args.device
        else (
            config["project"]["device"] if torch.cuda.is_available() else "cpu"
        )
    )
    storage_path = args.storage_path or config["project"]["storage_path"]
    dataset = args.dataset or config["data"]["dataset"]
    arch = args.model or config["model"]["architecture"]
    prefix = config["model"].get("prefix", "test2")

    # 1. Resource Setup
    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=config["data"].get("batch_size", 32)
    )

    # Load Models
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

    # 2. Patch Optimization / Loading
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

    # 3. Quantitative Evaluation (Attack Success Rate)
    # We evaluate on both models to compare their resistance to the patch
    print("\n📊 Starting Quantitative Patch Analysis...")
    asr_clean = _calculate_asr(patch_handler, model_clean, test_loader, device)
    asr_robust = _calculate_asr(
        patch_handler, model_robust, test_loader, device
    )

    print(f"\n[RESULTS] Universal Patch Performance:")
    print(f"   -> ASR on Standard Model: {asr_clean:.2f}%")
    print(f"   -> ASR on Robust Model:   {asr_robust:.2f}%")

    # 4. Qualitative Visualization
    print("\n🎨 Generating qualitative grid visualization...")
    _plot_results(
        patch_handler, model_clean, model_robust, test_loader, config
    )


def _calculate_asr(handler, model, loader, device):
    """
    Calculates the Attack Success Rate (ASR).
    ASR = (Successful Attacks) / (Samples initially correctly classified)
    """
    model.eval()
    total_initially_correct = 0
    successful_attacks = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating ASR", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Get initial predictions
            clean_preds = model(images).argmax(dim=1)
            correct_mask = clean_preds == labels

            num_correct = correct_mask.sum().item()
            if num_correct == 0:
                continue

            total_initially_correct += num_correct

            # Apply patch to correctly classified samples only
            patched_images = handler.apply_patch(images[correct_mask])
            patched_preds = model(patched_images).argmax(dim=1)

            # Attack is successful if prediction is no longer the ground truth
            successful_attacks += (
                (patched_preds != labels[correct_mask]).sum().item()
            )

    return (
        (successful_attacks / total_initially_correct) * 100
        if total_initially_correct > 0
        else 0.0
    )


def _plot_results(handler, m_clean, m_robust, loader, config):
    """Internal helper for publication-ready plotting."""
    m_clean.eval()
    m_robust.eval()

    # Retrieve one batch
    images, labels = next(iter(loader))
    images, labels = (
        images[:4].to(handler.device),
        labels[:4].to(handler.device),
    )

    # Generate adversarial samples
    patched = handler.apply_patch(images)
    with torch.no_grad():
        preds_c = m_clean(patched).argmax(dim=1)
        preds_r = m_robust(patched).argmax(dim=1)

    # Plotting Grid 1x4
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for i in range(4):
        # Denormalize and convert to HWC for matplotlib
        img = np.transpose(patched[i].detach().cpu().numpy(), (1, 2, 0))
        axes[i].imshow(np.clip(img, 0, 1))

        # Color logic based on robust model performance
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
    plt.show()
