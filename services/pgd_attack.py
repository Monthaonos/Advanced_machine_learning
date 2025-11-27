import torch
from torch import nn, Tensor


def pgd_attack(
    model: nn.Module,
    loss_fn: nn.Module,
    x: Tensor,
    y: Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> Tensor:
    """
    Projected Gradient Descent (PGD) attack (L-infinity).
    x: batch d'images (tensor [B, ...])
    y: labels (tensor [B])
    """

    if random_start:
        noise = (2 * torch.rand_like(x) - 1.0) * epsilon
        x_adv = torch.clamp(x + noise, clamp_min, clamp_max)
    else:
        x_adv = x.clone()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)

        with torch.enable_grad():
            logits = model(x_adv)
            loss = loss_fn(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv + alpha * torch.sign(grad)

        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        x_adv = torch.clamp(x_adv, clamp_min, clamp_max)

        x_adv = x_adv.detach()

    return x_adv
