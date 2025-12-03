import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple


def get_cifar10_loaders(
    batch_size: int = 64, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les dataloaders pour CIFAR-10 avec la normalisation standard
    et l'augmentation des données pour l'entraînement (version WideResNet).

    Args:
        batch_size: Taille du lot (batch) pour les DataLoader.
        num_workers: Nombre de processus pour le chargement des données.

    Returns:
        Un tuple contenant (train_loader, test_loader).
    """

    # Statistiques de normalisation (Moyenne, Écart-type) sur le set d'entraînement CIFAR-10
    # Ces valeurs sont essentielles pour centrer les données autour de 0.
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # --- Transformations pour l'entraînement (Augmentation + Normalisation) ---
    transform_train = transforms.Compose(
        [
            # 1. Augmentation: Recadrage aléatoire avec padding
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            # 2. Augmentation: Retournement horizontal aléatoire
            transforms.RandomHorizontalFlip(),
            # 3. Conversion en tenseur
            transforms.ToTensor(),
            # 4. Normalisation (L'étape cruciale pour la convergence)
            transforms.Normalize(*stats),
        ]
    )

    # --- Transformations pour le test (Juste Normalisation) ---
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ]
    )

    # --- Chargement des Datasets ---
    # Télécharge les données si elles ne sont pas dans le dossier './data'
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
        shuffle=True,  # Important pour l'entraînement
        num_workers=num_workers,
        pin_memory=True,  # Accélère le transfert vers le GPU
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # Pas de mélange pour le test
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
