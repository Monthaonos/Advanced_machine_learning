import torch
import torch.nn as nn
import torch.optim as optim
from attacks import pgd_attack

def model_train(model: nn.Module, epochs: int, lr: float, train_loader, test_loader, robust: bool):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    model = model.to(device)

    # --- BOUCLE D'ENTRAINEMENT ---

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train() # Mode entraînement (active BatchNorm et Dropout)
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # A. PHASE D'ATTAQUE (Si activée)

            if robust == True:
                # On remplace les images propres par des images piégées
                # Le modèle doit apprendre sur le PIRE cas
                model.eval() # On fige le modèle pour calculer l'attaque proprement

                eps = 0.3                          # 1/ on choisi un epsilon adaptatif
                alpha = 2/255
                #steps = min(max(0, epoch - 4), 15)  # 2/ on choisi un nombre de pas adaptatif
                steps = max(0, epoch - 3) - max(0, epoch - 17) - max(0, epoch - 19) + max(0, epoch - 21) - max(0, epoch - 23) + max(0, epoch - 25) - max(0, epoch - 27)
                inputs = pgd_attack(model, inputs, labels, eps, alpha, steps)
                model.train() # On remet en mode train pour apprendre

            # B. PHASE D'APPRENTISSAGE CLASSIQUE
            print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]}, eps: {eps}, steps: {steps}")

            optimizer.zero_grad()               # 1. Reset gradients
            outputs = model(inputs)             # 2. Forward
            loss = criterion(outputs, labels)   # 3. Calcul Loss
            loss.backward()                     # 4. Backward
            optimizer.step()                    # 5. Update Poids

            # Stats pour l'affichage
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99: # Affiche toutes les 100 mini-batchs
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], Loss: {running_loss/100:.4f}, Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0

        scheduler.step()


    # --- 7. EVALUATION FINALE ---
    print("Entraînement terminé. Évaluation sur le test set...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # Pas besoin de gradients pour le test (économie mémoire)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Précision finale sur les 10000 images de test: {100 * correct / total:.2f} %')

    # Sauvegarder le modèle
    torch.save(model.state_dict(), f"{model._get_name()}_robustpgd_0332255growingdecreasingsteps_e24_b64.pth")  #robustpgd_0322557_e25_b64


