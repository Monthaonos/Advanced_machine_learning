import torch
import os
import numpy.random as rd

# On réutilise votre fonction d'attaque PGD définie précédemment
# Assurez-vous qu'elle est bien importée ou copiée ici
from attacks import pgd_attack 
from data_load import load_data_CIFAR10
from model_arch import CIFAR10

def evaluate_model(model, loader, device, attack_func=None):
    """
    Evalue un modèle.
    Si attack_func est None : Evalue sur des données propres.
    Si attack_func est fourni : Génère l'attaque à la volée et évalue dessus.
    """
    model.eval() # Mode évaluation (fige le BatchNorm/Dropout)
    correct = 0
    total = 0
    
    # Pour l'attaque, on a besoin des gradients par rapport à l'image, 
    # mais pas par rapport aux poids du modèle.
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Génération de l'attaque (Si demandée)
        if attack_func is not None:
            # Important : PGD a besoin de calculer des gradients sur l'input
            # On active temporairement les gradients juste pour la génération
            with torch.enable_grad(): 
                eps = 0.33
                alpha = 2/255
                steps = rd.randint(2, 16)
                inputs = attack_func(model, inputs, labels, eps, alpha, steps)

        # 2. Prédiction (Mode inférence classique)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy




def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparaison sur : {device}")

    PATH = os.path.abspath("TEST_DAMIEN_CIFAR10/trained_models")

    # 1. Chargement des Données
    # On utilise load_data_CIFAR10 défini précédemment
    # batch_size=64 ou 128
    _, test_loader = load_data_CIFAR10(batch_size=100) 

    # 2. Chargement des Architectures
    # On instancie deux modèles vides
    model_std = CIFAR10().to(device)
    model_rob = CIFAR10().to(device)

    # 3. Chargement des Poids
    # (Adaptez les chemins vers vos fichiers .pth)
    try:
        model_std.load_state_dict(torch.load(os.path.join(PATH,"CIFAR10_natural_e24_b64.pth"), map_location=device))
        model_rob.load_state_dict(torch.load(os.path.join(PATH, "CIFAR10_robustpgd_032255growingsteps_e24_b64.pth"), map_location=device))
        print("Modèles chargés avec succès.")
    except FileNotFoundError:
        print("Erreur : Fichiers .pth introuvables. Vérifiez les chemins.")
        return

    print("-" * 50)
    print(f"{'Métrique':<25} | {'Modèle Standard':<15} | {'Modèle Robuste':<15}")
    print("-" * 50)

    # --- TEST 1 : Données Propres (Clean) ---
    acc_std_clean = evaluate_model(model_std, test_loader, device, attack_func=None)
    acc_rob_clean = evaluate_model(model_rob, test_loader, device, attack_func=None)
    
    print(f"{'Clean Accuracy':<25} | {acc_std_clean:.2f}%          | {acc_rob_clean:.2f}%")

    # --- TEST 2 : Données Attaquées (PGD) ---
    acc_std_adv = evaluate_model(model_std, test_loader, device, attack_func=pgd_attack)
    acc_rob_adv = evaluate_model(model_rob, test_loader, device, attack_func=pgd_attack)

    print(f"{'Adversarial Accuracy':<25} | {acc_std_adv:.2f}%          | {acc_rob_adv:.2f}%")
    print("-" * 50)


if __name__ == '__main__':
    # Lancez la comparaison
    compare_models()