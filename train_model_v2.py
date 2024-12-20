import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm 
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from transformers import ResNetForImageClassification
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Configuration et chemins
DATASET_PATH = "IA_Dataset"
MODEL_PATH = "resnet_waste_classification_model2.pth"
METRICS_PATH = "./metrics/"
metrics_dir = os.path.expanduser("./metrics/")
os.makedirs(metrics_dir, exist_ok=True)

def set_seed(seed=42):
    """Définir les graines pour la reproductibilité."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(dataset):
    """Calculer les poids de classe pour gérer les déséquilibres."""
    # Extraire tous os rótulos do dataset
    labels = [label for _, label in dataset]
    
    # Converter para numpy array
    import numpy as np
    labels_array = np.array(labels)
    
    # Obter classes únicas
    unique_classes = np.unique(labels_array)
    
    # Calcular pesos de classe
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels_array)
    
    return torch.tensor(class_weights, dtype=torch.float)

def create_data_loaders(dataset_path, transform, batch_size=32, val_split=0.15, test_split=0.15):
    """Créer des dataloaders avec séparation train/val/test."""
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    total_size = len(full_dataset)
    train_size = int((1 - val_split - test_split) * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def calculate_metrics(labels, predictions):
    """Calcula métricas como F1, precisão e revocação."""
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    return f1, precision, recall

def validate_model(model, val_loader, criterion, device):
    """Valider le modèle et calculer les métriques."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels)
            all_predictions.extend(predicted)
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    # Calcular métricas adicionais
    f1, precision, recall = calculate_metrics(torch.stack(all_labels), torch.stack(all_predictions))
    
    return avg_loss, accuracy, f1, precision, recall

def plot_confusion_matrix(model, val_loader, classes, device, epoch):
    """Gérer la matrice de confusion et la tracer."""
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(all_labels, all_predictions, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    # Tracer et sauvegarder
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    plt.title(f"Matrice de Confusion (Époque {epoch+1})")
    plt.tight_layout()
    cm_path = os.path.join(metrics_dir, f"confusion_matrix_epoch_{epoch+1}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {cm_path}")

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch):
    """Tracer et sauvegarder les courbes de perte et d'accuracy."""
    plt.figure(figsize=(12, 6))

    # Courbe de perte
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="x")
    plt.title("Évolution des Pertes")
    plt.xlabel("Époque")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)

    # Courbe d'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(val_accuracies, label="Validation Accuracy", marker="x")
    plt.title("Évolution de l'Accuracy")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Sauvegarder le graphique
    metrics_path = os.path.join(metrics_dir, f"metrics_epoch_{epoch+1}.png")
    plt.tight_layout()
    plt.savefig(metrics_path)
    plt.close()
    print(f"Graphiques de métriques sauvegardés: {metrics_path}")

custom_classes = ["Batterie", "Carton", "Metal", "Organique", "Papier", "Plastique", "Verre", "Vetements"]

def main():
    set_seed()
    
    # Configuration du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Transformations pour augmentation et prétraitement
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Transformations d'images prêtes.")
    # Créer les dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(DATASET_PATH, transform)
    print("Dataloaders prêts.")

    # Modèle ResNet
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    num_classes = len(train_loader.dataset.dataset.classes)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    print("Modèle ResNet prêt.")
    train_losses, val_losses, train_accuracies = [], [], []
    val_accuracies, val_f1_scores = [], []
    plot_confusion_matrix(model, val_loader=val_loader, classes=custom_classes, device=device, epoch=-1)
    plot_metrics(train_losses, val_losses,train_accuracies,val_accuracies,epoch=-1)
    
    # Charger un modèle existant si disponible
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Modèle pré-existant chargé.")
    
    model.to(device)
    
    # Calcul des poids de classe
    class_weights = get_class_weights(train_loader.dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer et scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    print("Initialisation de l'entraînement.")



    # Entraînement
    epochs = 3
    train_losses, val_losses, train_accuracies = [], [], []
    val_accuracies, val_f1_scores = [], []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc="En entraînement")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item())
        
        train_accuracy = 100 * train_correct / train_total
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss, val_accuracy, val_f1, val_precision, val_recall = validate_model(model, val_loader, criterion, device)
        
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print(f"Val F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # Sauvegarder les métriques et la matrice de confusion
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch)
        plot_confusion_matrix(model, val_loader, train_loader.dataset.dataset.classes, device, epoch)
        
        # Scheduler et sauvegarde du meilleur modèle
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, MODEL_PATH)
    
    # Tracer les courbes finales
    plot_confusion_matrix(model, val_loader=val_loader, classes=custom_classes, device=device, epoch=-1)
    plot_metrics(train_losses, val_losses,train_accuracies,val_accuracies,epoch=-1)
    
    # Évaluation finale
    final_test_loss, final_test_accuracy = validate_model(model, test_loader, criterion, device)
    print(f"\nRésultats finaux sur le jeu de test:")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy: {final_test_accuracy:.2f}%")

if __name__ == "__main__":
    main()