import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_cifar10_loaders(
    batch_size: int = 64, num_workers: int = 6, root: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for the CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        root (str): Root directory where dataset is stored.
    """

    # --- Training Transformations (Augmentation ONLY) ---
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # --- Test Transformations ---
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # --- Dataset Loading ---
    train_set = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train
    )

    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test
    )

    # --- DataLoader Creation ---
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
