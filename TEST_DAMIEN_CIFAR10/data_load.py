from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


def load_data_CIFAR10(batch_size = 64):
    # --- ÉTAPE 1 : LES TRANSFORMATIONS ---
    # Les images arrivent sous forme "PIL Image" (valeurs 0-255).
    # Le réseau a besoin de Tenseurs (valeurs 0-1 ou -1 à 1).

    transform_train = transforms.Compose([
        # 1. Data Augmentation (Optionnel mais recommandé pour la robustesse !)
        transforms.RandomHorizontalFlip(), # Retourne l'image horizontalement au hasard
        transforms.RandomCrop(32, padding=4), # Découpe un bout au hasard (bouge l'image)

        # 2. Conversion tenseur
        transforms.ToTensor(), # Convertit l'image (H, W, C) 0-255 en Tensor (C, H, W) 0-1

        # 3. Normalisation
        # (image - mean) / std.
        # Pour CIFAR10, les moyennes (mean) et écart-types (std) sont connus.
        # Cela centre les données autour de 0 pour aider le réseau.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Pour le test, on ne fait pas d'augmentation (on veut évaluer la vraie image)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # --- ÉTAPE 2 : LE TÉLÉCHARGEMENT ---

    trainset = datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train
    )

    testset = datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test
    )

    # --- ÉTAPE 3 : LE DATALOADER ---

    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,      
        num_workers=2    
    )

    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False,    
        num_workers=2
    )

    return train_loader, test_loader