import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- ZONE DE CONFIGURATION UTILISATEUR ---
# Remplace par ton import de modèle réel !
# from models import TonModele (Exemple)
# Si tu ne veux pas tester le modèle tout de suite, laisse MODEL_ARCH = None
MODEL_ARCH = None
BATCH_SIZE = 4
# -----------------------------------------


def check_data_statistics():
    print("\n--- 2. DIAGNOSTIC DES DONNÉES (Le Suspect N°1) ---")

    # Définition de la normalisation standard CIFAR-10
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Cas A : Sans Normalisation (Juste ToTensor)
    transform_raw = transforms.Compose([transforms.ToTensor()])

    # Cas B : Avec Normalisation (Ce que le modèle attend probablement)
    transform_norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*stats)]
    )

    # On télécharge un tout petit bout de CIFAR10 pour tester
    print("Chargement des données de test...")
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_norm
    )
    loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # On regarde le premier batch
    images, labels = next(iter(loader))

    v_min = images.min().item()
    v_max = images.max().item()
    v_mean = images.mean().item()

    print(f"📊 Statistiques d'un batch d'images envoyé au modèle :")
    print(f"   Min     : {v_min:.4f}")
    print(f"   Max     : {v_max:.4f}")
    print(f"   Moyenne : {v_mean:.4f}")

    print("\n🧐 ANALYSE DU MÉDECIN :")
    if v_min >= 0 and v_max <= 1.0:
        print(
            "⚠️ ALERTE : Tes images sont entre [0, 1]. Elles ne sont PAS normalisées."
        )
        print(
            "   Si ton modèle a été entraîné avec Normalize(), il va échouer (Acc ~10-20%)."
        )
    elif v_min < 0:
        print(
            "✅ OK : Tes images ont des valeurs négatives (ex: -1.8). Elles sont bien normalisées."
        )
        print(
            "   Si le modèle échoue encore, le problème vient des poids du modèle, pas des données."
        )

    return images, labels


def check_model_prediction(images, labels):
    print("\n--- 3. DIAGNOSTIC MODÈLE (Inférence Rapide) ---")

    if MODEL_ARCH is None:
        print(
            "ℹ️ Pas de classe de modèle fournie dans le script. On saute cette étape."
        )
        print(
            "   (Importe ta classe 'ResNet18' ou autre au début du fichier pour tester)"
        )
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    try:
        model = MODEL_ARCH()  # Instanciation
        # Ici on suppose que tu as le fichier .pth en local pour le test
        # model.load_state_dict(torch.load("ton_modele.pth"))
        model.to(device)
        model.eval()

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        print(f"Labels réels : {labels.cpu().numpy()}")
        print(f"Prédictions  : {predicted.cpu().numpy()}")

        acc = (predicted == labels).sum().item() / len(labels)
        print(f"Précision sur ce mini-batch : {acc * 100:.0f}%")

    except Exception as e:
        print(f"❌ Erreur lors du chargement/inférence modèle : {e}")


if __name__ == "__main__":
    # 2. Check Data
    imgs, lbls = check_data_statistics()

    # 3. Check Model (Optionnel)
    check_model_prediction(imgs, lbls)
