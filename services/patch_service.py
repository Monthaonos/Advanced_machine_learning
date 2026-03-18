"""
Phase 3: Universal adversarial patch (L0) analysis and visualization.
"""

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

GTSRB_LABELS = {
    0: "20km/h",
    1: "30km/h",
    2: "50km/h",
    3: "60km/h",
    4: "70km/h",
    5: "80km/h",
    6: "End 80km/h",
    7: "100km/h",
    8: "120km/h",
    9: "No passing",
    10: "No passing (>3.5t)",
    11: "Priority at next junc",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5t forbidden",
    17: "No entry",
    18: "General caution",
    19: "Curve left",
    20: "Curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Ice/snow",
    31: "Wild animals",
    32: "End speed + passing limits",
    33: "Turn right",
    34: "Turn left",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout",
    41: "End no passing",
    42: "End no passing (>3.5t)",
}

CIFAR10_LABELS = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}


def _get_label_map(dataset: str) -> dict:
    """Return the label mapping for a given dataset."""
    if dataset == "gtsrb":
        return GTSRB_LABELS
    elif dataset == "cifar10":
        return CIFAR10_LABELS
    return {}


def run_patch_analysis(args, config):
    """
    Execute the L0 patch attack pipeline:
    1. Load clean and robust model checkpoints.
    2. Train or load a universal adversarial patch.
    3. Compute Attack Success Rate (ASR) for both models.
    4. Generate qualitative visualization grid.
    """
    device = torch.device(
        args.device
        if args.device
        else (
            config["project"]["device"] if torch.cuda.is_available() else "cpu"
        )
    )

    storage_path = args.storage_path
    results_path = args.results_path

    dataset = args.dataset or config["data"]["dataset"]
    arch = args.model or config["model"]["architecture"]
    prefix = args.prefix or config["model"].get("prefix", "test")

    train_loader, test_loader, num_classes, in_channels = get_dataloaders(
        dataset, batch_size=config["data"].get("batch_size", 32)
    )

    model_clean = get_model(arch, dataset, num_classes, in_channels).to(device)
    model_robust = get_model(arch, dataset, num_classes, in_channels).to(device)

    manage_checkpoint(
        model_clean, storage_path,
        f"{prefix}_{dataset}_{arch}_clean.pth", device,
    )
    manage_checkpoint(
        model_robust, storage_path,
        f"{prefix}_{dataset}_{arch}_robust.pth", device,
    )

    # Patch optimization or loading
    patch_handler = UniversalPatchAttack(model_clean, config["patch_attack"])
    patch_file = f"{prefix}_{dataset}_{arch}_patch.pth"
    patch_path = os.path.join(storage_path, patch_file)

    if not os.path.exists(patch_path) or args.force_retrain:
        print(f"[*] Training universal patch for {arch}...")
        patch_handler.train_patch(train_loader)
        torch.save(patch_handler.patch.data, patch_path)
    else:
        print(f"[*] Loading existing patch from {patch_path}")
        patch_handler.patch.data = torch.load(
            patch_path, weights_only=True
        ).to(device)

    # Quantitative evaluation
    print("\n[*] Computing Attack Success Rate...")
    asr_clean = _calculate_asr(patch_handler, model_clean, test_loader, device)
    asr_robust = _calculate_asr(patch_handler, model_robust, test_loader, device)

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

    os.makedirs(results_path, exist_ok=True)

    output_filename = f"{prefix}_{dataset}_{arch}_patch_report.csv"
    save_results(pd.DataFrame(patch_stats), results_path, output_filename)
    print(f"[+] Patch report saved to: {results_path}/{output_filename}")

    # Qualitative visualization
    print("\n[*] Generating visualization grid...")
    _plot_results(
        patch_handler, model_clean, model_robust,
        test_loader, config, results_path, prefix,
    )


def _calculate_asr(handler, model, loader, device):
    """
    Compute Attack Success Rate (ASR).

    ASR = (correctly classified samples that are fooled by the patch)
        / (total correctly classified samples)
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
    handler, m_clean, m_robust, loader,
    config, results_path=None, prefix="test",
):
    """Generate a 1x4 qualitative grid showing patch attack predictions."""
    m_clean.eval()
    m_robust.eval()

    dataset = config["data"]["dataset"]
    label_map = _get_label_map(dataset)

    images, labels = next(iter(loader))
    images, labels = (
        images[:4].to(handler.device),
        labels[:4].to(handler.device),
    )

    patched = handler.apply_patch(images)
    with torch.no_grad():
        preds_c = m_clean(patched).argmax(dim=1)
        preds_r = m_robust(patched).argmax(dim=1)

    fig, axes = plt.subplots(1, 4, figsize=(24, 8))

    for i in range(4):
        img = np.transpose(patched[i].detach().cpu().numpy(), (1, 2, 0))
        axes[i].imshow(np.clip(img, 0, 1))

        gt_name = label_map.get(labels[i].item(), str(labels[i].item()))
        std_name = label_map.get(preds_c[i].item(), str(preds_c[i].item()))
        rob_name = label_map.get(preds_r[i].item(), str(preds_r[i].item()))

        color = "green" if preds_r[i] == labels[i] else "red"

        axes[i].set_title(
            f"GT: {gt_name}\nStd: {std_name}\nRob: {rob_name}",
            color=color,
            fontsize=10,
            fontweight="bold",
            pad=20,
        )
        axes[i].axis("off")

    plt.suptitle(
        f"Universal Patch Attack ({dataset.upper()})",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if results_path:
        arch = config["model"]["architecture"]
        fig_path = os.path.join(
            results_path, f"{prefix}_{dataset}_{arch}_patch_viz.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[+] Visualization saved to: {fig_path}")

    plt.show()
