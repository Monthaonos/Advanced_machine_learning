import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_cifar10_loaders(
    batch_size: int = 64, num_workers: int = 4, root: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Standardized CIFAR-10 data pipeline initialization.

    Args:
        batch_size (int): Number of images processed per iteration.
        num_workers (int): Multi-process data loading settings for optimized I/O.
        root (str): Directory for raw dataset storage.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and Testing DataLoaders.
    """

    # --- Training Pipeline: Geometric Augmentation and Conversion ---
    # Reflection padding and random cropping are standard for CIFAR-10
    # to prevent overfitting and ensure spatial invariance.
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # --- Testing Pipeline: Consistency and Validation ---
    # No augmentation is applied during inference to maintain evaluation integrity.
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # --- Dataset Instantiation ---
    train_set = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train
    )

    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test
    )

    # --- Hardware-Optimized DataLoader Creation ---
    # pin_memory=True speeds up the transfer from CPU to GPU/MPS.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid incomplete batches for gradient stability
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
