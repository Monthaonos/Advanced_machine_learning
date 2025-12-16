import torch
import torch.nn as nn
import torchvision


class GTSRBModel(nn.Module):
    def __init__(self, pretrained=False):  # J'ai ajouté l'option au cas où
        super(GTSRBModel, self).__init__()

        # 1. Charger le squelette ResNet18
        # pretrained=True peut accélérer la convergence, mais False est ok
        self.base_model = torchvision.models.resnet18(pretrained=pretrained)

        # --- FIX CRITIQUE POUR PETITES IMAGES (32x32) ---
        # On remplace la grosse conv 7x7 par une 3x3 qui ne réduit pas la taille (stride=1)
        self.base_model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # On supprime le MaxPool initial qui détruit trop d'infos sur des petites images
        self.base_model.maxpool = nn.Identity()
        # ------------------------------------------------

        # 2. Adapter la sortie (43 classes pour GTSRB)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 43)

        # 3. Normalisation intégrée (Top pratique pour les attaques !)
        self.register_buffer(
            "mu", torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "sigma", torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x est attendu dans [0, 1]
        x_norm = (x - self.mu) / self.sigma
        return self.base_model(x_norm)
