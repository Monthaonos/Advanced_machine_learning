import torch
import torch.nn as nn
from typing import Optional
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
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM).

    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014).

    Args:
        model (nn.Module): The neural network model (should include normalization wrapper).
        images (torch.Tensor): Input images in range [0, 1].
        labels (torch.Tensor): Ground truth labels.
        epsilon (float): Perturbation magnitude (L-infinity constraint).
        **kwargs: Placeholder for compatibility with other attack signatures.

    Returns:
        torch.Tensor: Adversarial images clamped to [0, 1].
    """
    # Optimization: If no perturbation is allowed, return original images immediately.
    if epsilon == 0:
        return images

    # Clone and detach to create a new leaf node for the computation graph.
    # This prevents modifying the original tensor in-place.
    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    images.requires_grad = True

    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    # Backward pass to calculate gradients w.r.t input
    model.zero_grad()
    loss.backward()

    # Collect the element-wise sign of the data gradient
    data_grad = images.grad.data

    # Craft adversarial image: x_adv = x + epsilon * sign(gradient)
    perturbed_image = images + epsilon * data_grad.sign()

    # Clip to maintain valid pixel range [0, 1]
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
    Generates adversarial examples using Projected Gradient Descent (PGD).
    This is essentially an iterative version of FGSM with random initialization.

    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017).

    Args:
        model (nn.Module): The target model.
        images (torch.Tensor): Clean images [0, 1].
        labels (torch.Tensor): True labels.
        epsilon (float): Maximum perturbation (L-infinity norm).
        alpha (float): Step size for each iteration. Default: 2/255.
        num_steps (int): Number of attack iterations. Default: 10.

    Returns:
        torch.Tensor: Adversarial images [0, 1].
    """
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)

    # Keep original images for projection step (epsilon constraint)
    original_images = images.clone().detach()

    # Random Start (Exploration):
    # Initialize with uniform random noise inside the epsilon ball [-eps, +eps].
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Iterative update: move by alpha in gradient direction
        adv_images = images + alpha * images.grad.sign()

        # Projection Step:
        # 1. Calculate perturbation (adv - original)
        # 2. Clip perturbation to [-epsilon, epsilon]
        eta = torch.clamp(
            adv_images - original_images, min=-epsilon, max=epsilon
        )

        # 3. Apply perturbation to original image and enforce [0, 1] pixel range
        # .detach() is critical to truncate the graph and prevent memory leaks
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
    Generates adversarial examples using Momentum Iterative Method (MIM).
    Uses momentum to stabilize update directions and escape local maxima.

    Reference: Dong et al., "Boosting Adversarial Attacks with Momentum" (2018).

    Args:
        decay (float): Momentum decay factor (mu). Default: 1.0.
        (Other args same as PGD)

    Returns:
        torch.Tensor: Adversarial images [0, 1].
    """
    if epsilon == 0:
        return images

    images = images.to(images.device)
    labels = labels.to(images.device)
    original_images = images.clone().detach()

    # Random start (improves robustness against defense)
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, 0, 1)

    # Initialize momentum buffer
    momentum = torch.zeros_like(images).detach().to(images.device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = images.grad.data

        # Normalize gradient by L1 norm (Mean Absolute Value)
        # This stabilizes the magnitude of updates across iterations
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad / (
            grad_norm + 1e-12
        )  # Add small epsilon to avoid div by zero

        # Accumulate momentum: g_{t+1} = mu * g_t + (grad / ||grad||_1)
        momentum = momentum * decay + grad

        # Update image using sign of momentum
        images = images.detach() + alpha * momentum.sign()

        # Projection step (L-infinity constraint)
        delta = torch.clamp(images - original_images, -epsilon, epsilon)
        images = torch.clamp(original_images + delta, 0, 1)

    return images


class UniversalPatchAttack:
    """
    Implementation of the Universal Adversarial Patch optimization.

    This class optimizes a localized patch (L0-norm attack) to be effective
    across an entire dataset and at multiple spatial locations. It uses
    Gradient Ascent to maximize the classification loss.
    """

    def __init__(self, model: nn.Module, patch_config: dict):
        """
        Initializes the adversarial patch optimization environment.

        Args:
            model (nn.Module): The target neural network to attack.
            patch_config (dict): Configuration containing 'scale', 'learning_rate',
                                 and 'number_of_steps'.
        """
        self.model = model
        self.config = patch_config

        # Device management: ensures compatibility with CUDA, MPS, and CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Dimensionality setup: patch size is proportional to image size (default 32x32)
        self.image_size = 32
        self.patch_size = int(self.image_size * self.config["scale"])

        # Learnable parameter initialization: Random noise in the [0, 1] range
        self.patch = nn.Parameter(
            torch.rand(
                (3, self.patch_size, self.patch_size), device=self.device
            )
        )

    def apply_patch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Overlays the adversarial patch onto a batch of images, strictly
        restricted to the central region where traffic signs are located.

        This prevents the optimizer from getting 'lost' in the background,
        ensuring every update contributes to the attack success.

        Args:
            images (torch.Tensor): Input batch of shape (N, C, H, W).

        Returns:
            torch.Tensor: The batch of images with the localized adversarial patch.
        """
        x_adv = images.clone().to(self.device)

        # Define the central boundaries (25% to 75% of the image size)
        # For GTSRB, this targets the actual traffic sign location.
        low = int(self.image_size * 0.25)
        high = max(low, int(self.image_size * 0.75) - self.patch_size)

        # Spatial Invariance: Random positioning within the sign-containing region
        x = random.randint(low, high)
        y = random.randint(low, high)

        # Apply the learned patch parameters to the specified spatial region
        x_adv[:, :, y : y + self.patch_size, x : x + self.patch_size] = (
            self.patch
        )

        # Maintain numerical stability within the valid [0, 1] pixel range
        return torch.clamp(x_adv, 0, 1)

    def train_patch(self, train_loader: torch.utils.data.DataLoader):
        """
        Executes the optimization loop using Gradient Ascent with Gradient Accumulation.

        Mathematical objective: argmax_P E_{(x,y)~D} [L(f(x + P), y)]
        This version averages gradients over multiple batches to stabilize the
        universal property and prevent loss oscillations.

        Args:
            train_loader (DataLoader): Training data distribution for optimization.
        """
        # Set model to evaluation mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Adam optimizer focused solely on the patch pixels
        optimizer = optim.Adam([self.patch], lr=self.config["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        # Stabilization: Gradient accumulation parameters
        accumulation_steps = 4  # Average over 4 batches (128 samples)
        num_steps = self.config["number_of_steps"]
        current_step = 0

        # Use a persistent iterator for consistent data flow across steps
        iter_loader = iter(train_loader)

        print(
            f"[*] Optimizing {self.patch_size}x{self.patch_size} patch (Stabilized & Centralized)..."
        )

        # Optimization loop across the data distribution
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

                # Training with random positioning inside the central zone
                patched_images = self.apply_patch(images)

                # Forward pass through the frozen model
                outputs = self.model(patched_images)

                # Gradient Ascent logic: Maximize classification loss
                # Divided by accumulation_steps to maintain correct gradient scale
                loss = -criterion(outputs, labels) / accumulation_steps
                loss.backward()

                step_loss_sum += -loss.item() * accumulation_steps

            optimizer.step()

            # Box constraint projection: Keep patch values within [0, 1]
            self.patch.data.clamp_(0, 1)

            if current_step % 50 == 0:
                print(
                    f"    Step [{current_step}/{num_steps}] | Average Loss: {step_loss_sum / accumulation_steps:.4f}"
                )

            current_step += 1

    def save(self, storage_path: str, filename: str):
        """
        Serializes the optimized patch tensor to disk.

        Args:
            storage_path (str): Target directory.
            filename (str): Name of the .pth file.
        """
        os.makedirs(storage_path, exist_ok=True)
        save_path = os.path.join(storage_path, filename)

        # Saving to CPU ensures compatibility during future loading
        torch.save(self.patch.data.cpu(), save_path)
        print(f"[+] Patch saved successfully at: {save_path}")
