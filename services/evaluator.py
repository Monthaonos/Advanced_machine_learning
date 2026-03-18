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
    Run a sequence of adversarial attacks and measure model performance.

    Args:
        model: The model to evaluate (must include normalization wrapper).
        model_name: Name identifier for logging.
        dataset_name: Dataset identifier for logging.
        dataloader: Test set iterator (images in [0, 1]).
        attack_configs: Attack name -> {"fn": callable, "kwargs": dict} mapping.
        device: Computation device.

    Returns:
        List of result dictionaries suitable for a Pandas DataFrame.
    """
    results = []
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    for attack_name, config in attack_configs.items():
        print(f"   -> Running attack: {attack_name}...")

        attack_fn = config.get("fn")
        attack_args = config.get("kwargs", {})

        correct = 0
        total = 0
        total_loss = 0.0

        loader_bar = tqdm(
            dataloader, desc=f"      {attack_name}", leave=False
        )

        for images, labels in loader_bar:
            images, labels = images.to(device), labels.to(device)

            if attack_fn is None:
                inputs = images
            else:
                inputs = attack_fn(model, images, labels, **attack_args)

            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                batch_size = labels.size(0)
                total += batch_size
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * batch_size

        acc = 100 * correct / total
        avg_loss = total_loss / total

        results.append(
            {
                "Model": model_name,
                "Dataset": dataset_name,
                "Attack": attack_name,
                "Epsilon": attack_args.get("epsilon", 0.0),
                "Steps": attack_args.get("num_steps", 0),
                "Accuracy": acc,
                "Loss": avg_loss,
            }
        )

        print(f"      Acc: {acc:.2f}% | Loss: {avg_loss:.4f}")

    return results
