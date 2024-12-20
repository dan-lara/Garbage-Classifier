import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from transformers import ResNetForImageClassification
import matplotlib.pyplot as plt

# Chemin vers le dataset
dataset_path = "IA_Dataset"
# Chemin vers le fichier du modèle sauvegardé
model_path = "resnet_model.pth"

# Préparer les transformations des images
transform = transforms.Compose([
    transforms.ToTensor()  # Convertir en tenseur
])

# Charger les données d'entraînement
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Préparer le modèle ResNet
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 8)  # 8 classes

# Vérifier si un modèle existe déjà et charger les poids
if os.path.exists(model_path):
    print(f"Chargement des poids depuis '{model_path}'...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    print("Aucun modèle existant trouvé, entraînement à partir de zéro.")

# Configurer l'entraînement
device = torch.device('cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialiser des listes pour stocker les pertes
train_losses = []

# Boucle d'entraînement
epochs = 10  # Ajustez selon vos besoins
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Début de l'Epoch {epoch+1}/{epochs}")
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Entraînement
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits  # Extraire les logits pour la perte
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:  # Afficher la progression toutes les 10 batchs
            print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)  # Stocker la perte de l'époch
    print(f"Epoch {epoch+1} terminée. Loss moyenne : {epoch_loss:.4f}")

# Sauvegarder le modèle après les nouvelles epochs
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé sous '{model_path}'")

# Tracer la courbe de perte
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, marker='o', label='Train Loss')
plt.title("Évolution de la perte durant l'entraînement")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

# Sauvegarder la figure
plt.savefig("loss_curve.png")
