import torch
import torch.nn as nn
import torch.optim as optim


def model_train(model: nn.Module, epochs: int, lr: float, train_loader, test_loader, robust: bool):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")
    
    # --- BOUCLE D'ENTRAINEMENT ---

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train() # Mode entraînement (active BatchNorm et Dropout)
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            ## A. PHASE D'ATTAQUE (Si activée)
            #
            #if robust == True:
            #    # On remplace les images propres par des images piégées
            #    # Le modèle doit apprendre sur le PIRE cas
            #    model.eval() # On fige le modèle pour calculer l'attaque proprement
            #    inputs = pgd_attack(model, inputs, labels)
            #    model.train() # On remet en mode train pour apprendre

            # B. PHASE D'APPRENTISSAGE CLASSIQUE

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
    torch.save(model.state_dict(), f"./trained_models/{model._get_name}_non_robust.pth")