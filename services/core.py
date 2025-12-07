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
    # --- AJOUT 1 : L'argument scheduler (optionnel) ---
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
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

        # --- Boucle des Batchs (L'entraînement pur) ---
        for batch_idx, (x, y) in enumerate(epoch_bar, start=1):
            x = x.to(device)
            y = y.to(device)

            should_attack = torch.bernoulli(torch.tensor(prob)).item() == 1.0

            if pgd_robust and should_attack:
                model.eval()  # PGD se fait souvent en mode eval (optionnel mais propre)
                x_adv = pgd_attack(
                    model=model,
                    images=x,
                    labels=y,
                    epsilon=epsilon,
                    alpha=alpha,
                    num_steps=num_steps,
                )
                model.train()  # On repasse en train pour la backward pass
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

        # --- AJOUT 2 : Mise à jour du Scheduler (Fin d'époque) ---
        if scheduler is not None:
            # On dit au scheduler : "Une époque est passée, ajuste le LR !"
            scheduler.step()

            # (Juste pour l'affichage) On récupère la nouvelle valeur pour vérifier
            current_lr = scheduler.get_last_lr()[0]
            # On utilise print ou tqdm.write pour ne pas casser la barre de progression
            tqdm.write(
                f"📉 Fin Époque {epoch + 1} -> Learning Rate ajusté à : {current_lr:.6f}"
            )

    return
