import torch
import torch.nn as nn
import torch.nn.functional as F


# --- AJOUT : La classe de Normalisation (La même que pour WideResNet) ---
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- AJOUT ICI : Normalisation Interne (Stats CIFAR-10) ---
        self.normalize = Normalization(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        # ----------------------------------------------------------

        # --- Block 1: Apprendre les formes simples ---
        # Entrée: 32x32 -> Sortie: 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # --- Block 2: Apprendre les motifs complexes ---
        # Entrée: 16x16 -> Sortie: 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # --- Block 3: Apprendre les concepts abstraits ---
        # Entrée: 8x8 -> Sortie: 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # --- Classifier ---
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # --- AJOUT ICI : On normalise d'abord ---
        x = self.normalize(x)
        # ----------------------------------------

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
