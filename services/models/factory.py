import torch
import torch.nn as nn
from .simple_cnn import SimpleCNN
from .resnet import ResNet18
from .wideresnet import WideResNet

# Valeurs standard (ImageNet/CIFAR) - À adapter selon tes besoins précis
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# Pour GTSRB, souvent on utilise [0.5, 0.5, 0.5] ou on calcule les stats réelles
GTSRB_MEAN = [0.3337, 0.3064, 0.3171]
GTSRB_STD = [0.2672, 0.2564, 0.2629]


class NormalizeLayer(nn.Module):
    """
    Standardizes the input data (assumed to be in [0, 1])
    using dataset-specific mean and std.
    This is added as a layer so attacks can optimize on [0, 1] input space.
    """

    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        # make them buffers so they are saved with state_dict but not trained
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def get_model(
    model_name: str,
    dataset_name: str,
    num_classes: int,
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory to get a model normalized for the specific dataset.
    """
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    # 1. Select the base architecture
    if model_name == "simple_cnn":
        base_model = SimpleCNN(
            num_classes=num_classes, in_channels=in_channels
        )
    elif model_name in ["resnet", "resnet18"]:
        base_model = ResNet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet":
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
        raise ValueError(f"Model {model_name} not supported.")

    # 2. Select Normalization stats
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "gtsrb":
        mean, std = GTSRB_MEAN, GTSRB_STD
    else:
        # Default fallback (e.g., ImageNet or 0.5) if dataset is unknown
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # 3. Wrap the model: Normalize -> Architecture
    final_model = nn.Sequential(NormalizeLayer(mean, std), base_model)

    return final_model
