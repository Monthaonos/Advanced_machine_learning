import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from typing import List, Optional, Union
from services.attacks import pgd_attack


def train_models(
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    alpha: float,
    num_steps: int,
    train_prob: float = 1.0,
    random_start: bool = True,
    pgd_robust: bool = False,
    scheduler: Optional[LRScheduler] = None,
    epochs: int = 10,
    device: Optional[Union[torch.device, str]] = None,
) -> List[float]:
    """
    Executes the training pipeline for standard or robust (Adversarial) models.

    Args:
        train_dataloader (DataLoader): Training data distribution.
        model (nn.Module): The neural network architecture to optimize.
        loss_fn (nn.Module): Objective function (typically CrossEntropyLoss).
        optimizer (Optimizer): Gradient descent algorithm (e.g., SGD, Adam).
        epsilon (float): Maximum L-infinity perturbation budget.
        alpha (float): Step size for each iteration of the PGD attack.
        num_steps (int): Number of iterations for adversarial example generation.
        train_prob (float): Probability (0 to 1) of using adversarial samples per batch.
        random_start (bool): If True, adds random noise to PGD initialization.
        pgd_robust (bool): If True, enables Adversarial Training (AT) logic.
        scheduler (Optional[LRScheduler]): Learning rate adjustment strategy.
        epochs (int): Total number of full passes over the training set.
        device (Optional[Union[torch.device, str]]): Computation hardware (CPU/CUDA/MPS).

    Returns:
        List[float]: Evolution of the average training loss per epoch.
    """

    # --- 1. Device Orchestration ---
    # Automatically infers the device from model parameters if not explicitly provided.
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model.to(device)

    loss_history = []

    # --- 2. Main Training Loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Progress bar initialization for visual monitoring
        epoch_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )

        for batch_idx, (x, y) in enumerate(epoch_bar, start=1):
            x, y = x.to(device), y.to(device)

            # --- 3. Adversarial Example Generation (Inner Loop) ---
            # If robust training is active, we stochastically decide to attack the current batch.
            # Reference: Madry et al., 2018 (Min-Max Optimization).
            should_attack = torch.rand(1).item() <= train_prob

            if pgd_robust and should_attack:
                # IMPORTANT: Switch to eval mode to freeze BatchNorm/Dropout statistics
                # during adversarial perturbation generation.
                model.eval()

                x_adv = pgd_attack(
                    model=model,
                    images=x,
                    labels=y,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                    random_start=random_start,
                )

                # Switch back to training mode for weights update
                model.train()
                inputs = x_adv
            else:
                inputs = x

            # --- 4. Optimization Step (Outer Loop) ---
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_fn(y_hat, y)

            # Backpropagation and weights update
            loss.backward()
            optimizer.step()

            # Logging updates
            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # --- 5. End of Epoch Procedures ---
        epoch_loss = running_loss / len(train_dataloader)
        loss_history.append(epoch_loss)

        # Scheduler update (e.g., StepLR or MultiStepLR)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(
                f"📉 Epoch {epoch + 1} Complete | Avg Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}"
            )

    return loss_history
