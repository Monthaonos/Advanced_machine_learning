import torch
import torch.nn as nn
import torch.nn.functional as F


# --- AJOUT : Classe de Normalisation ---
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # On enregistre mean et std comme des "buffers" persistants.
        # Cela permet à PyTorch de gérer automatiquement le déplacement vers GPU/CPU
        # quand on fait model.to(device).
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
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

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class Network(nn.Module):
    def __init__(
        self, depth=28, widen_factor=10, dropRate=0.3, num_classes=10
    ):
        super(Network, self).__init__()

        # --- AJOUT : Normalisation Interne (Stats CIFAR-10) ---
        # Le modèle s'attend désormais à recevoir des images en [0, 1]
        self.normalize = Normalization(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        # ------------------------------------------------------

        nChannels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # 1st block
        self.block1 = self._make_layer(
            block, nChannels[0], nChannels[1], n, 1, dropRate
        )
        # 2nd block
        self.block2 = self._make_layer(
            block, nChannels[1], nChannels[2], n, 2, dropRate
        )
        # 3rd block
        self.block3 = self._make_layer(
            block, nChannels[2], nChannels[3], n, 2, dropRate
        )

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

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

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- AJOUT : On applique la normalisation en TOUT PREMIER ---
        out = self.normalize(x)
        # ------------------------------------------------------------

        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# Fonction helper pour instancier facilement
def get_model(device):
    model = Network(depth=28, widen_factor=10, dropRate=0.3, num_classes=10)
    return model.to(device)
