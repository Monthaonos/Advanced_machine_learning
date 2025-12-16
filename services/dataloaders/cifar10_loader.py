import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_cifar10_loaders(
    batch_size: int = 64, num_workers: int = 6
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les dataloaders pour CIFAR-10.

    CORRECTION ROBUSTESSE :
    La normalisation a été retirée d'ici pour être placée à l'intérieur du modèle.
    Les images sortent donc en [0, 1].

    Args:
        batch_size: Taille du lot (batch).
        num_workers: Nombre de processus.

    Returns:
        Un tuple (train_loader, test_loader).
    """

    # NOTE : Les stats sont conservées ici pour référence, mais NE SONT PLUS utilisées
    # dans le loader. Vous devez les utiliser dans la première couche de votre modèle.
    # stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # --- Transformations pour l'entraînement (Augmentation UNIQUEMENT) ---
    transform_train = transforms.Compose(
        [
            # 1. Augmentation: Recadrage aléatoire avec padding
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            # 2. Augmentation: Retournement horizontal aléatoire
            transforms.RandomHorizontalFlip(),
            # 3. Conversion en tenseur (Met les pixels en [0, 1])
            transforms.ToTensor(),
        ]
    )

    # --- Transformations pour le test ---
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # PAS DE NORMALISATION ICI
        ]
    )

    # --- Chargement des Datasets ---
    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # --- Création des DataLoaders ---
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
