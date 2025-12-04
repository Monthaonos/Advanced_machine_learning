import torch
from torch import nn
from tqdm import tqdm
from services.attacks import pgd_attack


def train_models(
    train_dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    alpha: float,
    num_steps: int,
    prob: float = 1.0,
    random_start: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    epochs: int = 10,
    pgd_robust: bool = False,
    device: torch.device | str | None = None,
):
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.to(device)

    for epoch in range(epochs):
        model.train()

        epoch_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(epoch_bar, start=1):
            x = x.to(device)
            y = y.to(device)

            # We randomly attack the model
            should_attack = torch.bernoulli(torch.tensor(prob)).item() == 1.0

            if pgd_robust and should_attack:
                x_adv = pgd_attack(
                    model=model,
                    images=x,
                    labels=y,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                )
                inputs = x_adv
            else:
                inputs = x

            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / batch_idx

            epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return
