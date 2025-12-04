import torch
import torch.nn as nn
from tqdm import tqdm


def run_evaluation_suite(
    model, model_name, dataset_name, dataloader, attack_configs, device
):
    """
    Execute a sequence of attacks defined in attack_configs.

    Args:
        attack_configs (dict): dictionnary of configs.
        Ex: {
            "Clean": {"fn": None, "kwargs": {}},
            "FGSM_Small": {"fn": fgsm_attack, "kwargs": {"epsilon": 0.01}},
            "PGD_Strong": {"fn": pgd_attack, "kwargs": {"epsilon": 0.03, "steps": 20}}
        }
    """
    results = []
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    # Iteration over each attack
    for attack_name, config in attack_configs.items():
        print(f"   -> Exécution : {attack_name}...")

        attack_fn = config.get("fn")
        attack_args = config.get("kwargs", {})

        correct = 0
        total = 0
        total_loss = 0.0

        # We go through the dataset
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Attack generation
            if attack_fn is None:
                # Clean case
                inputs = images
            else:
                # Adversarial case (FGSM, MIM, PGD...)
                # Dynamical use of kwargs
                inputs = attack_fn(model, images, labels, **attack_args)

            # Inference
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)

        acc = 100 * correct / total
        avg_loss = total_loss / total

        # We save the results
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

    return results
