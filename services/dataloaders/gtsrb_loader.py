import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_gtsrb_loaders(batch_size=128):
    # --- Augmentation pour le Train ---
    # On veut simuler des conditions réelles (angle, zoom, lumière)
    # ATTENTION : Pas de RandomHorizontalFlip pour les panneaux !
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            # 1. Légère rotation (-10 à +10 degrés), translation et zoom
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            # 2. Changement de luminosité/contraste (très important pour l'extérieur)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # PAS DE NORMALISATION ICI (car gérée dans le modèle pour PGD)
        ]
    )

    # --- Juste Resize/Tensor pour le Test ---
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    train_set = torchvision.datasets.GTSRB(
        root="./data", split="train", download=True, transform=train_transform
    )
    test_set = torchvision.datasets.GTSRB(
        root="./data", split="test", download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
