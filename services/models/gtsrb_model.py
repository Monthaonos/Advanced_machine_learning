import torch
import torch.nn as nn
import torchvision


class GTSRBModel(nn.Module):
    def __init__(self):
        super(GTSRBModel, self).__init__()
        # On charge un ResNet18 (léger et efficace)
        self.base_model = torchvision.models.resnet18(pretrained=False)
        # On adapte la sortie pour les 43 classes de panneaux GTSRB
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 43)

        # Paramètres de normalisation GTSRB (ou ImageNet standard)
        self.register_buffer(
            "mu", torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "sigma", torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # 1. L'attaque envoie x dans [0, 1]
        # 2. On normalise manuellement ici
        x_norm = (x - self.mu) / self.sigma
        # 3. On passe au ResNet
        return self.base_model(x_norm)
