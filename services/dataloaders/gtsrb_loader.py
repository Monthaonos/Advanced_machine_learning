import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_gtsrb_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for the GTSRB dataset.
    """

    # --- Training Transformations ---
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    )

    # --- Test Transformations ---
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    # --- Dataset Loading ---
    train_set = torchvision.datasets.GTSRB(
        root=root, split="train", download=True, transform=train_transform
    )
    test_set = torchvision.datasets.GTSRB(
        root=root, split="test", download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
