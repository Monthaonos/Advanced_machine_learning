import torch
import torch.nn as nn
import csv
import os
from datetime import datetime

# Imports
from services.dataloaders.gtsrb_loader import get_gtsrb_loaders
from services.models.gtsrb_model import GTSRBModel
from services.pgd_attack import pgd_attack

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
PATH_CLEAN = "checkpoints/gtsrb/gtsrb_clean.pth"
PATH_ROBUST = "checkpoints/gtsrb/gtsrb_robust.pth"
RESULTS_FILE = "results/attack_results_gtsrb.csv"

# Paramètres d'attaque globaux
EPSILON = 8 / 255
ALPHA = 2 / 255
NUM_STEPS = 20

# ... (Tes fonctions fgsm_attack, mim_attack et evaluate_model restent inchangées) ...
# Je remets ici brièvement les définitions pour que le code soit exécutable si tu le copies-colles,
# mais tu peux garder les tiennes.


def fgsm_attack(model, images, labels, eps):
    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    perturbed_image = images + eps * data_grad.sign()
    return torch.clamp(perturbed_image, 0, 1)


def mim_attack(model, images, labels, eps, alpha, iters, decay=1.0):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    original_images = images.clone().detach()
    images = images + torch.empty_like(images).uniform_(-eps, eps)
    images = torch.clamp(images, 0, 1)
    momentum = torch.zeros_like(images).detach().to(DEVICE)
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = momentum * decay + grad
        images = images.detach() + alpha * momentum.sign()
        delta = torch.clamp(images - original_images, -eps, eps)
        images = torch.clamp(original_images + delta, 0, 1)
    return images


def evaluate_model(model, test_loader, attack_name):
    model.eval()
    correct = 0
    total = 0
    # print(f"   ... Attaque en cours : {attack_name}") # Commenté pour moins de spam console

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if attack_name == "Clean":
            inputs = images
        elif attack_name == "FGSM":
            inputs = fgsm_attack(model, images, labels, eps=EPSILON)
        elif attack_name == "PGD":
            inputs = pgd_attack(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                x=images,
                y=labels,
                epsilon=EPSILON,
                alpha=ALPHA,
                num_steps=NUM_STEPS,
            )
        elif attack_name == "MIM":
            inputs = mim_attack(
                model, images, labels, eps=EPSILON, alpha=ALPHA, iters=NUM_STEPS
            )

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# --- FONCTION DE SAUVEGARDE ---
def save_results_to_csv(records, filename):
    """
    Enregistre une liste de dictionnaires dans un fichier CSV.
    Crée le fichier avec les en-têtes s'il n'existe pas.
    """
    file_exists = os.path.isfile(filename)

    # Définition des colonnes du CSV
    fieldnames = [
        "Date",
        "Model_Type",
        "Attack",
        "Accuracy",
        "Epsilon",
        "Alpha",
        "Steps",
        "Dataset",
    ]

    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Écrire l'en-tête seulement si le fichier est nouveau
        if not file_exists:
            writer.writeheader()

        writer.writerows(records)

    print(f"\n[INFO] Résultats sauvegardés dans : {filename}")


# --- MAIN ---
if __name__ == "__main__":
    print(f"--- Évaluation de Robustesse Avancée ({DEVICE}) ---")
    _, test_loader = get_gtsrb_loaders(batch_size=BATCH_SIZE)

    # Chargement
    models = {}
    print("Chargement modèle Classique...")
    net_c = GTSRBModel().to(DEVICE)
    net_c.load_state_dict(torch.load(PATH_CLEAN, map_location=DEVICE))
    models["Classique"] = net_c

    print("Chargement modèle Robuste...")
    net_r = GTSRBModel().to(DEVICE)
    net_r.load_state_dict(torch.load(PATH_ROBUST, map_location=DEVICE))
    models["Robuste"] = net_r

    attacks_list = ["Clean", "FGSM", "PGD", "MIM"]

    # Liste pour accumuler les données à sauvegarder
    records_to_save = []

    # Timestamp actuel pour identifier ce run
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- BOUCLE D'ÉVALUATION ---
    results_for_print = {name: [] for name in models.keys()}

    for atk in attacks_list:
        print(f"\n[{atk.upper()}]")
        for model_name, model in models.items():
            acc = evaluate_model(model, test_loader, atk)
            results_for_print[model_name].append(acc)

            print(f"   -> {model_name} : {acc:.2f}%")

            # Création de l'entrée pour le fichier CSV
            record = {
                "Date": current_time,
                "Model_Type": model_name,
                "Attack": atk,
                "Accuracy": round(acc, 2),
                "Epsilon": f"{EPSILON:.4f}",  # Pratique pour voir 0.0314 au lieu de 8/255
                "Alpha": f"{ALPHA:.4f}",
                "Steps": NUM_STEPS if atk in ["PGD", "MIM"] else 1,  # 1 step pour FGSM
                "Dataset": "GTSRB",
            }
            records_to_save.append(record)

    # --- AFFICHAGE TABLEAU CONSOLE ---
    print("\n" + "=" * 65)
    header = (
        f"{'Attaque':<12} | "
        + " | ".join([f"{k:<15}" for k in models.keys()])
        + " | Gain Robustesse"
    )
    print(header)
    print("-" * 65)

    for i, atk in enumerate(attacks_list):
        row = f"{atk:<12} | "
        vals = []
        for name in models.keys():
            val = results_for_print[name][i]
            vals.append(val)
            row += f"{val:.2f}%          | "

        gain = vals[1] - vals[0]  # Suppose que vals[1] est Robuste et vals[0] Classique
        row += f"{gain:+.2f}%"
        print(row)
    print("=" * 65)

    # --- SAUVEGARDE FINALE ---
    save_results_to_csv(records_to_save, RESULTS_FILE)
