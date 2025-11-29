import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size: int = 64, num_workers: int = 2):
    """
    Crée les dataloaders pour CIFAR-10 (Version Large / WideResNet).
    Inclut la normalisation standard pour une meilleure convergence.
    """

    # Statistiques standards calculées sur le set d'entrainement de CIFAR-10
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Entraînement : Augmentation + Normalisation
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),  # <--- L'ajout crucial
        ]
    )

    # Test : Juste Normalisation
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*stats),  # <--- L'ajout crucial
        ]
    )

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
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
