from .cifar10_loader import get_cifar10_loaders
from .gtsrb_loader import get_gtsrb_loaders
from typing import Tuple
from torch.utils.data import DataLoader


def get_dataloaders(
    dataset_name: str, batch_size: int = 128, root: str = "./data"
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Unified Factory to initialize DataLoaders and retrieve dataset metadata.

    Args:
        dataset_name (str): The identifier of the dataset (e.g., 'cifar10', 'gtsrb').
        batch_size (int): Number of samples per batch.
        root (str): Root directory for dataset storage.

    Returns:
        Tuple[DataLoader, DataLoader, int, int]:
            (train_loader, test_loader, num_classes, in_channels)

    Raises:
        ValueError: If the requested dataset name is not implemented.
    """
    name = dataset_name.lower()

    if name == "cifar10":
        # Standard RGB dataset with 10 balanced classes
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=batch_size, root=root
        )
        num_classes = 10
        in_channels = 3

    elif name == "gtsrb":
        # Traffic sign recognition dataset with 43 imbalanced classes
        train_loader, test_loader = get_gtsrb_loaders(
            batch_size=batch_size, root=root
        )
        num_classes = 43
        in_channels = 3

    else:
        # Strict validation to ensure pipeline integrity
        supported = ["cifar10", "gtsrb"]
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Available options: {', '.join(supported)}"
        )

    return train_loader, test_loader, num_classes, in_channels
