import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_gtsrb_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    German Traffic Sign Recognition Benchmark (GTSRB) pipeline initialization.

    Args:
        batch_size (int): Size of the mini-batches for optimization.
        num_workers (int): Number of parallel processes for data loading.
        root (str): Root directory for dataset storage.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and Testing DataLoaders.
    """

    # --- Training Pipeline: Normalization and Domain-Specific Augmentation ---
    # Resizing is mandatory as GTSRB original images have variable dimensions.
    # RandomAffine and ColorJitter simulate environmental noise (perspective, lighting).
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

    # --- Testing Pipeline: Static Validation ---
    # Identical resizing ensures compatibility with the model's input layer.
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    # --- Dataset Instantiation ---
    # split="train" and split="test" are used to partition the data according to the challenge rules.
    train_set = torchvision.datasets.GTSRB(
        root=root, split="train", download=True, transform=train_transform
    )

    test_set = torchvision.datasets.GTSRB(
        root=root, split="test", download=True, transform=test_transform
    )

    # --- DataLoader Instantiation ---
    # drop_last=True is used in training to maintain constant batch sizes for gradient stability.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
