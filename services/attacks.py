"""
Adversarial attack implementations: FGSM, PGD, MIM, and Universal Patch.

References:
    - Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
    - Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
    - Dong et al., "Boosting Adversarial Attacks with Momentum" (2018)
    - Brown et al., "Adversarial Patch" (2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    **kwargs,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM).

    Single-step attack: x_adv = x + epsilon * sign(grad_x L(f(x), y))

    Args:
        model: Target neural network.
        images: Input images in [0, 1].
        labels: Ground truth labels.
        epsilon: L-infinity perturbation budget.

    Returns:
        Adversarial images clamped to [0, 1].
    """
    if epsilon == 0:
        return images

    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    perturbed_image = images + epsilon * images.grad.data.sign()
    return torch.clamp(perturbed_image, 0, 1)


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = 2 / 255,
    num_steps: int = 10,
    **kwargs,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD).

    Iterative FGSM with random initialization and L-inf projection.

    Args:
        model: Target model.
        images: Clean images in [0, 1].
        labels: True labels.
        epsilon: Maximum L-infinity perturbation.
        alpha: Step size per iteration.
        num_steps: Number of attack iterations.

    Returns:
        Adversarial images in [0, 1].
    """
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)
    original_images = images.clone().detach()

    # Random initialization within epsilon-ball
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()

        # Project back into epsilon-ball around original images
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images


def mim_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = 2 / 255,
    num_steps: int = 10,
    decay: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Momentum Iterative Method (MIM).

    PGD variant with momentum accumulation for more stable and transferable attacks.

    Args:
        decay: Momentum decay factor (mu).
        (Other args same as PGD)

    Returns:
        Adversarial images in [0, 1].
    """
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)
    original_images = images.clone().detach()

    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    momentum = torch.zeros_like(images).detach().to(images.device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = images.grad.data
        # L1-normalize gradient for stable momentum accumulation
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad / (grad_norm + 1e-12)

        # Momentum update: g_{t+1} = mu * g_t + grad / ||grad||_1
        momentum = momentum * decay + grad

        images = images.detach() + alpha * momentum.sign()

        # L-infinity projection
        delta = torch.clamp(images - original_images, -epsilon, epsilon)
        images = torch.clamp(original_images + delta, 0, 1)

    return images


class UniversalPatchAttack:
    """
    Universal adversarial patch optimization (L0-norm attack).

    Optimizes a localized, image-agnostic patch via gradient ascent to maximize
    misclassification across the entire dataset. Uses spatial invariance
    (random positioning within the central region) and gradient accumulation
    for stable convergence.
    """

    def __init__(self, model: nn.Module, patch_config: dict):
        self.model = model
        self.config = patch_config

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.image_size = 32
        self.patch_size = int(self.image_size * self.config["scale"])

        self.patch = nn.Parameter(
            torch.rand(
                (3, self.patch_size, self.patch_size), device=self.device
            )
        )

    def apply_patch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Overlay the adversarial patch onto images within the central region.

        The patch is randomly positioned within the [25%, 75%] central zone
        to enforce spatial invariance during optimization.
        """
        x_adv = images.clone().to(self.device)

        low = int(self.image_size * 0.25)
        high = max(low, int(self.image_size * 0.75) - self.patch_size)

        x = random.randint(low, high)
        y = random.randint(low, high)

        x_adv[:, :, y : y + self.patch_size, x : x + self.patch_size] = (
            self.patch
        )

        return torch.clamp(x_adv, 0, 1)

    def train_patch(self, train_loader: torch.utils.data.DataLoader):
        """
        Optimize the patch via gradient ascent with gradient accumulation.

        Objective: argmax_P E_{(x,y)~D} [L(f(apply_patch(x, P)), y)]
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        optimizer = optim.Adam([self.patch], lr=self.config["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        accumulation_steps = 4
        num_steps = self.config["number_of_steps"]
        current_step = 0

        iter_loader = iter(train_loader)

        print(f"[*] Optimizing {self.patch_size}x{self.patch_size} patch...")

        while current_step < num_steps:
            optimizer.zero_grad()
            step_loss_sum = 0

            for _ in range(accumulation_steps):
                try:
                    images, labels = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(train_loader)
                    images, labels = next(iter_loader)

                images, labels = images.to(self.device), labels.to(self.device)
                patched_images = self.apply_patch(images)
                outputs = self.model(patched_images)

                # Gradient ascent: negate loss to maximize misclassification
                loss = -criterion(outputs, labels) / accumulation_steps
                loss.backward()
                step_loss_sum += -loss.item() * accumulation_steps

            optimizer.step()
            self.patch.data.clamp_(0, 1)

            if current_step % 50 == 0:
                print(
                    f"    Step [{current_step}/{num_steps}] | "
                    f"Avg Loss: {step_loss_sum / accumulation_steps:.4f}"
                )

            current_step += 1

    def save(self, storage_path: str, filename: str):
        """Save the optimized patch tensor to disk."""
        os.makedirs(storage_path, exist_ok=True)
        save_path = os.path.join(storage_path, filename)
        torch.save(self.patch.data.cpu(), save_path)
        print(f"[+] Patch saved to: {save_path}")
