import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, steps=40):
    """
    Implémentation de l'attaque PGD (Projected Gradient Descent).
    Args:
        images: Le batch d'images originales
        labels: Les vraies étiquettes
        eps: La force maximale de l'attaque (le rayon de la boule)
        alpha: La taille du pas à chaque itération
        steps: Le nombre d'itérations de l'attaque
    """
    # 1. On crée une copie détachée pour ne pas modifier l'original tout de suite
    # On ajoute un bruit aléatoire initial (Recommandé par le papier pour explorer le voisinage)
    delta = torch.zeros_like(images).uniform_(-eps, eps)
    delta = torch.clamp(delta, min=-eps, max=eps)
    delta.requires_grad = True # Important : on veut le gradient par rapport à ce bruit !

    # 2. Boucle itérative (L'attaque)
    for _ in range(steps):
        # On passe l'image piégée dans le modèle
        # Note : On additionne l'image originale (fixe) et le bruit (variable)
        outputs = model(images + delta)
        
        loss = F.cross_entropy(outputs, labels)

        # On calcule le gradient de la loss par rapport au bruit (delta)
        # create_graph=False économise la mémoire car on ne veut pas backpropager à travers l'attaque
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        # 3. Mise à jour du bruit (Ascension de Gradient)
        # On va DANS le sens du gradient pour AUGMENTER l'erreur
        delta.data = delta.data + alpha * grad.sign()

        # 4. Projection (On reste dans la boule epsilon)
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        
        # 5. Contrainte d'image valide (On reste entre 0 et 1 ou -1 et 1 selon votre normalisation)
        # Ici on suppose que vos images normalisées sont grosso modo entre -2 et 2, 
        # mais idéalement on clippe par rapport à l'image dénormalisée. 
        # Pour faire simple ici, on projette juste le delta.
        
    # On retourne l'image modifiée, détachée du graphe (c'est une nouvelle constante pour l'entraînement)
    return (images + delta).detach()