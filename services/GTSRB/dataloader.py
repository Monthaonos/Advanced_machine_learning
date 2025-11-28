import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_gtsrb_loaders(batch_size=128):
    # Juste Resize et ToTensor. Les valeurs restent entre [0, 1]
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Ou plus grand si tu veux
            transforms.ToTensor(),
        ]
    )

    train_set = torchvision.datasets.GTSRB(
        root="./data", split="train", download=True, transform=transform
    )
    test_set = torchvision.datasets.GTSRB(
        root="./data", split="test", download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader
