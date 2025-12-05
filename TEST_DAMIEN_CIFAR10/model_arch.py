import torch
import torch.nn as nn 
import torch.nn.functional as F


class CIFAR10(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Block 1: feature extraction ---
        # 3 input channels (RGB) → 32 feature maps

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)


        # --- Block 2: deeper features ---
        # 32 → 64 feature maps

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)


        # --- Block 3: even deeper features ---
        # 64 → 128 feature maps

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)


        # --- Block 4: even deeper features ---
        # 128 → 256 feature maps

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)


        # --- Block 5: Bottleneck ---
        # 256 -> 64 (Compression)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=1) 
        self.bn5 = nn.BatchNorm2d(64) 

        # 64 -> 64 (Traitement spatial)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64) 

        # 64 -> 128 (Expansion)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(128)


        # --- Output layer ---
        # After max pooling: spatial size becomes ?×?

        self.fc = nn.Linear(128 * 8 * 8, num_classes)



    def forward(self, x):

        # --- Convolutional feature extractor ---
        # Each block: Conv → BatchNorm → ReLU

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.max_pool2d(x, kernel_size=2) # 32x32 -> 16x16
        resnet = x

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.bn7(self.conv7(x))

        x += resnet

        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=2) # 16x16 -> 8x8

        # --- Flatten for the classifier ---

        x = torch.flatten(x, 1)

        # --- Final classification layer ---

        res = self.fc(x)

        return res
