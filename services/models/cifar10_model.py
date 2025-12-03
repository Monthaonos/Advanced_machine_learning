import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Block 1: feature extraction ---
        # 3 input channels (RGB) → 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # --- Block 2: deeper features ---
        # 32 → 64 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # --- Block 3: even deeper features ---
        # 64 → 128 feature maps
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # --- Classifier head ---
        # After max pooling: spatial size becomes 16×16
        self.fc = nn.Linear(128 * 16 * 16, num_classes)

    def forward(self, x):
        # --- Convolutional feature extractor ---
        # Each block: Conv → BatchNorm → ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # --- Spatial downsampling ---
        # Reduces size from 32×32 to 16×16
        x = F.max_pool2d(x, kernel_size=2)

        # --- Flatten for the classifier ---
        x = torch.flatten(x, 1)

        # --- Final classification layer ---
        logits = self.fc(x)

        return logits
