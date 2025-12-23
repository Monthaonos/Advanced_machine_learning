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
    Performs standard or adversarial training of the model.
    """

    # 1. Device Management
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model.to(device)

    loss_history = []

    for epoch in range(epochs):
        model.train()

        # Progress bar
        epoch_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )
        running_loss = 0.0

        # --- Batch Loop ---
        for batch_idx, (x, y) in enumerate(epoch_bar, start=1):
            x = x.to(device)
            y = y.to(device)

            # Determine if this specific batch should be attacked (stochastic AT)
            should_attack = torch.rand(1).item() <= train_prob

            if pgd_robust and should_attack:
                model.eval()

                # Generate adversarial examples
                x_adv = pgd_attack(
                    model=model,
                    images=x,
                    labels=y,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                    random_start=random_start,
                )

                model.train()
                inputs = x_adv
            else:
                inputs = x

            # Standard Optimization Step
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            # Logging
            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # End of Epoch Stats
        epoch_loss = running_loss / len(train_dataloader)
        loss_history.append(epoch_loss)

        # --- Scheduler Step ---
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(
                f"📉 End of Epoch {epoch + 1} -> LR updated to: {current_lr:.6f}"
            )

    return loss_history
