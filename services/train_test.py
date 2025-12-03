import torch
from torch import nn
from tqdm import tqdm
from .pgd_attack import pgd_attack


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
            should_attack = torch.bernoulli(torch.tensor(prob)).item()

            if pgd_robust and should_attack:
                x_adv = pgd_attack(
                    model=model,
                    loss_fn=loss_fn,
                    x=x,
                    y=y,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                    random_start=random_start,
                    clamp_min=clamp_min,
                    clamp_max=clamp_max,
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


def test_models(
    test_dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device | str | None = None,
):
    """
    Évaluation sur données propres (pas d'attaque).
    Retourne (loss_moyenne, accuracy).
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        test_bar = tqdm(test_dataloader, desc="Test (clean)", leave=False)
        for x, y in test_bar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                test_bar.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"}
                )

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def test_models_adversarial(
    test_dataloader,
    model: nn.Module,
    loss_fn: nn.Module,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    device: torch.device | str | None = None,
):
    """
    Évaluation sur des exemples adversariaux générés par PGD à partir des données de test.
    Retourne (loss_moyenne, accuracy) sur les x_adv.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    test_bar = tqdm(test_dataloader, desc="Test (adversarial PGD)", leave=False)

    for x, y in test_bar:
        x = x.to(device)
        y = y.to(device)

        # On génère les x_adv AVEC gradient par rapport à x, mais
        # en gardant le modèle en mode eval (BatchNorm/Dropout figés).
        x_adv = pgd_attack(
            model=model,
            loss_fn=loss_fn,
            x=x,
            y=y,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            random_start=random_start,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )

        with torch.no_grad():
            logits = model(x_adv)
            loss = loss_fn(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                test_bar.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"}
                )

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc
