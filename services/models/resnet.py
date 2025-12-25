import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Fundamental Residual Block for ResNet architectures.

    This block consists of two 3x3 convolutional layers with Batch Normalization.
    A skip connection (shortcut) is used to add the input directly to the
    output, mitigating the vanishing gradient problem in deep networks.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        Initializes the BasicBlock.

        Args:
            in_planes (int): Number of input feature maps.
            planes (int): Number of output feature maps for the convolutions.
            stride (int): Stride for the first convolution (controls downsampling).
        """
        super(BasicBlock, self).__init__()

        # First convolutional layer (downsampling occurs here if stride > 1)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Identity shortcut connection
        # If input/output dimensions differ, we use a 1x1 conv to align them.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicBlock.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Element-wise addition of the residual (shortcut)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Residual Network (ResNet) specialized for small-scale image classification.

    Specific modifications for 32x32 resolution:
    - Replaced the initial 7x7 convolution with a 3x3 kernel.
    - Removed the initial MaxPool layer to prevent premature information loss.
    """

    def __init__(
        self,
        block: nn.Module,
        num_blocks: list,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        """
        Initializes the ResNet architecture.

        Args:
            block (nn.Module): The block type to use (e.g., BasicBlock).
            num_blocks (list): List containing the number of blocks per stage.
            num_classes (int): Number of target output classes.
            in_channels (int): Number of input image channels (RGB = 3).
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Entry layer: Preserves spatial dimensions (32x32)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Stage definitions: Successive downsampling to extract high-level features
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global Average Pooling and final Linear Classifier
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: nn.Module, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """
        Helper function to stack residual blocks into a layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass of the ResNet.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Average pooling reduces spatial dimensions (4x4) to (1x1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    """
    Factory function for the ResNet-18 variant.

    Architecture summary:
    - 4 stages, each containing 2 BasicBlocks.
    - Optimized for robust training on small-scale datasets.
    """
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
    )
