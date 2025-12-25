import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Modular Convolutional Neural Network designed for 32x32 image classification.

    Architecture features:
    - Three-stage convolutional backbone with Batch Normalization.
    - Increasing filter depth (32 -> 256) to capture hierarchical features.
    - Fully connected classifier with Dropout regularization for robustness.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """
        Initializes the SimpleCNN architecture.

        Args:
            num_classes (int): Dimensionality of the output logit vector.
            in_channels (int): Number of input color channels (e.g., RGB = 3).
        """
        super(SimpleCNN, self).__init__()

        # --- Feature Extraction Backbone ---
        # Successive spatial downsampling via MaxPool2d (32x32 -> 16x16 -> 8x8 -> 4x4)
        self.features = nn.Sequential(
            # Stage 1: Initial filtering
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2: Intermediate abstraction
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3: High-level feature mapping
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Dense Classification Head ---
        # Includes Dropout to mitigate overfitting during adversarial training.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass logic.

        Args:
            x (torch.Tensor): Input batch of shape (N, C, 32, 32).

        Returns:
            torch.Tensor: Unnormalized class logits.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten spatial dimensions to vector
        x = self.classifier(x)
        return x
