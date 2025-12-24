import torch
import torch.nn as nn
from typing import Optional
import torch.optim as optim
import os


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
    The patch is a localized sticker optimized to maximize the model's loss
    across a distribution of images (CIFAR-10 or GTSRB).
    """

    def __init__(self, model: nn.Module, config: dict):
        """
        Initializes the patch optimization environment.
        :param model: The target neural network (frozen during optimization).
        :param config: Dictionary containing 'patch_attack' parameters from config.toml.
        """
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Determine patch size based on scale parameter (e.g., 0.3 for 30% coverage)
        # For 32x32 images, scale 0.3 results in a ~10x10 patch.
        self.patch_size = int(32 * config["scale"])

        # Initialize patch with random values in [0, 1] as a learnable parameter
        self.patch = nn.Parameter(
            torch.rand(
                (3, self.patch_size, self.patch_size), device=self.device
            )
        )

    def apply_patch(
        self, images: torch.Tensor, x_coord: int = 11, y_coord: int = 11
    ) -> torch.Tensor:
        """
        Applies the current patch onto a batch of images at fixed coordinates.
        :param images: Input batch (N, C, H, W).
        :param x_coord: Top-left horizontal position.
        :param y_coord: Top-left vertical position.
        :return: Patched images clamped to valid range.
        """
        x_adv = images.clone()
        # Overlay the patch on the target region
        x_adv[
            :,
            :,
            y_coord : y_coord + self.patch_size,
            x_coord : x_coord + self.patch_size,
        ] = self.patch
        return torch.clamp(x_adv, 0, 1)

    def train_patch(self, train_loader: torch.utils.data.DataLoader):
        """
        Optimization loop for the universal patch.
        Objective: Maximize CrossEntropy loss over the training set.
        Goal: argmax_P E_{(x,y)~D} [L(f(x + P), y)]
        """
        self.model.eval()  # Ensure model is in eval mode (frozen weights)

        # Setup optimizer based on config (e.g., "Adam")
        optimizer = optim.Adam([self.patch], lr=self.config["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        num_steps = self.config["number_of_steps"]
        print(
            f"Starting Universal Patch optimization for {num_steps} iterations..."
        )

        for step in range(num_steps):
            try:
                images, labels = next(iter(train_loader))
            except StopIteration:
                break  # End of data

            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            # Apply patch and forward pass
            patched_images = self.apply_patch(images)
            outputs = self.model(patched_images)

            # Minimize negative loss to achieve gradient ascent (maximize error)
            loss = -criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Ensure patch pixels remain within valid [0, 1] range
            self.patch.data.clamp_(0, 1)

            if step % 50 == 0:
                print(f"Step [{step}/{num_steps}] | Loss: {-loss.item():.4f}")

    def save(self, storage_path: str, filename: str):
        """Persists the optimized patch tensor to the checkpoints directory."""
        os.makedirs(storage_path, exist_ok=True)
        save_path = os.path.join(storage_path, filename)
        torch.save(self.patch.data, save_path)
        print(f"Patch saved successfully at: {save_path}")
