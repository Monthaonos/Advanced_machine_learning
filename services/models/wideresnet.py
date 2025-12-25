import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Wide Residual Network Basic Block using Pre-activation logic.

    Sequence: BN -> ReLU -> Conv (3x3).
    This variant is highly effective for deep wide architectures as it
    stabilizes the distribution of activations before convolutions.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropRate: float = 0.0,
    ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes

        # Shortcut connection to handle dimension/stride mismatch
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-activation residual link.
        """
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        # First convolution stage
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        # Optional Dropout for regularization
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        # Second convolution stage
        out = self.conv2(out)

        # Residual addition (Skip Connection)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    Utility block to stack multiple BasicBlocks into a single stage.
    """

    def __init__(
        self,
        nb_layers: int,
        in_planes: int,
        out_planes: int,
        block: nn.Module,
        stride: int,
        dropRate: float = 0.0,
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet architecture tailored for CIFAR-10 and GTSRB.

    The depth and width parameters define the model capacity.
    A common robust configuration is WRN-28-10 (Depth 28, Widen Factor 10).
    The formula for layers per stage is: $n = \frac{depth - 4}{6}$.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        widen_factor: int = 10,
        dropRate: float = 0.3,
        in_channels: int = 3,
    ):
        super(WideResNet, self).__init__()
        n = (depth - 4) / 6
        nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Initial convolution before the residual stages
        self.conv1 = nn.Conv2d(
            in_channels,
            nStages[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Stage 1: 32x32 resolution
        self.block1 = NetworkBlock(
            n, nStages[0], nStages[1], BasicBlock, 1, dropRate
        )

        # Stage 2: 16x16 resolution
        self.block2 = NetworkBlock(
            n, nStages[1], nStages[2], BasicBlock, 2, dropRate
        )

        # Stage 3: 8x8 resolution
        self.block3 = NetworkBlock(
            n, nStages[2], nStages[3], BasicBlock, 2, dropRate
        )

        # Final Batch Normalization and Linear Classifier
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)
        self.nChannels = nStages[3]

        # Weight initialization (Kaiming Normal for ReLU-based networks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        # Final average pooling (8x8 -> 1x1)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
