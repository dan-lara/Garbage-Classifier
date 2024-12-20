# Importations nécessaires
import torch
from transformers import ResNetForImageClassification
from torchvision import transforms
from PIL import Image

# Charger le modèle sauvegardé
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 8)  # 8 classes
model.load_state_dict(torch.load('resnet_model.pth', map_location=torch.device('cpu')), strict=False)

model.eval()

# Préparer les transformations pour une image
transform = transforms.Compose([
    transforms.ToTensor()  # Pas de normalisation, car le dataset est déjà normalisé
])

# Définir les classes correspondant à vos dossiers
classes = ["Batterie", "Carton", "Metal", "Organique", "Papier", "Plastique", "Verre", "Vetements"]

# Fonction pour effectuer une prédiction
def predict_image(image_path):
    # Charger et prétraiter l'image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension batch

    # Faire une prédiction
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)  # Convertir les logits en probabilités
        print("Probabilités par classe :", probabilities.numpy())  # Afficher les probabilités pour chaque classe
        _, predicted = torch.max(logits, 1)
        return classes[predicted.item()]  # Retourner la classe prédite

# Tester une image
image_path = '/home/polytech/Bureau/Projet_IA/IA_Dataset/Verre/Verre_1453.jpg'  # Remplacez par le chemin de votre image
classe_predite = predict_image(image_path)
print(f"L'image appartient à la classe : {classe_predite}")
