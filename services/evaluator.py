import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Dict, Any


def run_evaluation_suite(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    dataloader: torch.utils.data.DataLoader,
    attack_configs: Dict[str, Dict[str, Any]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Execute a sequence of adversarial attacks defined in attack_configs and measure performance.

    This function iterates over the dataset for each attack configuration.
    It supports both 'Clean' evaluation (no attack) and adversarial attacks (FGSM, PGD, etc.).

    Args:
        model (nn.Module): The model to evaluate (must include normalization wrapper).
        model_name (str): Name of the model (for logging purposes).
        dataset_name (str): Name of the dataset (for logging purposes).
        dataloader (DataLoader): Iterator over the test set (images in [0, 1]).
        attack_configs (dict): Dictionary defining attacks.
            Example:
            {
                "Clean": {"fn": None, "kwargs": {}},
                "FGSM": {"fn": fgsm_attack, "kwargs": {"epsilon": 0.01}},
                "PGD":  {"fn": pgd_attack,  "kwargs": {"epsilon": 0.03, "num_steps": 10}}
            }
        device (torch.device): Computation device (CPU/GPU).

    Returns:
        List[Dict]: A list of result dictionaries, suitable for creating a Pandas DataFrame.
    """
    results = []
    loss_fn = nn.CrossEntropyLoss()

    # Ensure model is in evaluation mode (disables Dropout, freezes Batch Norm stats)
    model.eval()

    # Loop over each attack configuration
    for attack_name, config in attack_configs.items():
        print(f"   -> Running Attack: {attack_name}...")

        attack_fn = config.get("fn")
        attack_args = config.get("kwargs", {})

        correct = 0
        total = 0
        total_loss = 0.0

        # Use tqdm for progress tracking (Critical for slow attacks like PGD)
        # leave=False ensures the bar clears after completion to keep logs clean
        loader_bar = tqdm(
            dataloader, desc=f"      Evaluating {attack_name}", leave=False
        )

        for images, labels in loader_bar:
            images, labels = images.to(device), labels.to(device)

            # 1. Generate Adversarial Examples
            if attack_fn is None:
                # Clean evaluation
                inputs = images
            else:
                # Adversarial evaluation
                # Note: attack_fn handles gradient calculation internally
                inputs = attack_fn(model, images, labels, **attack_args)

            # 2. Inference (No Gradient Needed for metrics)
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                # Update stats
                batch_size = labels.size(0)
                total += batch_size
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * batch_size

        # Calculate metrics
        acc = 100 * correct / total
        avg_loss = total_loss / total

        # Log structure for DataFrame
        results.append(
            {
                "Model": model_name,
                "Dataset": dataset_name,
                "Attack": attack_name,
                "Epsilon": attack_args.get("epsilon", 0.0),
                "Steps": attack_args.get("num_steps", 0),  # 0 for FGSM/Clean
                "Accuracy": acc,
                "Loss": avg_loss,
            }
        )

        # Immediate feedback in console
        print(f"      [Result] Acc: {acc:.2f}% | Loss: {avg_loss:.4f}")

    return results
