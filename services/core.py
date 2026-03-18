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
    Training loop for standard or adversarial (Madry et al.) models.

    When pgd_robust=True, each batch is stochastically converted to adversarial
    examples with probability train_prob, implementing the min-max optimization.

    Returns:
        Average training loss per epoch.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model.to(device)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        epoch_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )

        for batch_idx, (x, y) in enumerate(epoch_bar, start=1):
            x, y = x.to(device), y.to(device)

            # Stochastic adversarial example generation (inner maximization)
            should_attack = torch.rand(1).item() <= train_prob

            if pgd_robust and should_attack:
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
                model.train()
                inputs = x_adv
            else:
                inputs = x

            # Outer minimization step
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        epoch_loss = running_loss / len(train_dataloader)
        loss_history.append(epoch_loss)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(
                f"Epoch {epoch + 1} | Avg Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}"
            )

    return loss_history
