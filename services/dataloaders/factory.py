from .cifar10_loader import get_cifar10_loaders
from .gtsrb_loader import get_gtsrb_loaders
from typing import Tuple
from torch.utils.data import DataLoader


def get_dataloaders(
    dataset_name: str, batch_size: int = 128, root: str = "./data"
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Factory to return dataloaders and dataset info based on name.

    Returns:
        train_loader, test_loader, num_classes, in_channels
    """
    name = dataset_name.lower()

    if name == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=batch_size, root=root
        )
        num_classes = 10
        in_channels = 3

    elif name == "gtsrb":
        train_loader, test_loader = get_gtsrb_loaders(
            batch_size=batch_size, root=root
        )
        num_classes = 43
        in_channels = 3

    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Available: cifar10, gtsrb"
        )

    # On renvoie bien 4 valeurs maintenant
    return train_loader, test_loader, num_classes, in_channels
