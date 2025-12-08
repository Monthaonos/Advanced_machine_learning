import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Block 1: Apprendre les formes simples ---
        # Entrée: 32x32 -> Sortie: 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # On augmente la profondeur
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Division de la taille par 2
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
        # Dropout pour éviter le par-coeur (Overfitting)
        self.dropout = nn.Dropout(0.5)

        # Calcul taille: 256 canaux * 4 pixels * 4 pixels = 4096
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Aplatir
        x = torch.flatten(x, 1)

        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Encore un peu de dropout avant la fin
        x = self.fc2(x)

        return x
