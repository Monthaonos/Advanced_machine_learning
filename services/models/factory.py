import torch
import torch.nn as nn
from .simple_cnn import SimpleCNN
from .resnet import ResNet18
from .wideresnet import WideResNet

# --- Dataset Statistics (RGB Space) ---
# CIFAR-10: Standard statistics calculated over the training set.
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# GTSRB: Specific statistics reflecting the varied lighting of German roads.
GTSRB_MEAN = [0.3337, 0.3064, 0.3171]
GTSRB_STD = [0.2672, 0.2564, 0.2629]


class NormalizeLayer(nn.Module):
    """
    On-the-fly normalization layer.

    This module transforms input tensors from the [0, 1] range to a
    standardized distribution (Mean=0, Std=1). Integrated into the
    model sequential flow, it allows adversarial attacks to optimize
    directly in the valid image space [0, 1].
    """

    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        # Buffers are stored in state_dict but excluded from gradient updates.
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def get_model(
    model_name: str,
    dataset_name: str,
    num_classes: int,
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Unified Model Factory with integrated normalization.

    Args:
        model_name (str): Architecture identifier ('simple_cnn', 'resnet18', 'wideresnet').
        dataset_name (str): Dataset identifier for normalization stats selection.
        num_classes (int): Number of output neurons (classes).
        in_channels (int): Input image channels (default=3 for RGB).
        **kwargs: Additional hyperparameters for complex architectures (e.g., WideResNet).

    Returns:
        nn.Module: Sequential model (Normalization -> Base Architecture).
    """
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    # --- 1. Base Architecture Selection ---
    if model_name == "simple_cnn":
        base_model = SimpleCNN(
            num_classes=num_classes, in_channels=in_channels
        )
    elif model_name in ["resnet", "resnet18"]:
        base_model = ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet":
        # Extract WideResNet specific parameters with default safety
        depth = kwargs.get("depth", 28)
        widen_factor = kwargs.get("widen_factor", 10)
        drop_rate = kwargs.get("drop_rate", 0.3)
        base_model = WideResNet(
            num_classes=num_classes,
            depth=depth,
            widen_factor=widen_factor,
            dropRate=drop_rate,
            in_channels=in_channels,
        )
    else:
        raise ValueError(
            f"Model architecture '{model_name}' is not supported."
        )

    # --- 2. Normalization Strategy Selection ---
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "gtsrb":
        mean, std = GTSRB_MEAN, GTSRB_STD
    else:
        # Fallback to generic normalization (Identity-like)
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # --- 3. Unified Sequential Wrapper ---
    # Constructing a single object for simplified inference and attack pipelines.
    final_model = nn.Sequential(NormalizeLayer(mean, std), base_model)

    return final_model
