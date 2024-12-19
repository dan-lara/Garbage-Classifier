# EcoMindAI : Intelligence Artificielle au Service du Tri des Déchets

## Présentation du Projet

EcoMindAI est une initiative innovante qui combine un dataset exhaustif de classification des déchets avec un modèle d'intelligence artificielle haute performance, conçu pour révolutionner la gestion des déchets et promouvoir le développement durable. Ce projet représente une avancée significative dans l'application de l'apprentissage automatique aux défis environnementaux contemporains.

---

## Architecture Technique

### Dataset de Classification des Déchets
- Collection de plus de 20 000 images haute qualité
- 8 catégories distinctes de déchets, avec 2 500 images par classe
- Images prétraitées et normalisées (224x224 pixels)
- Compatible avec les principales bibliothèques de deep learning (PyTorch, TensorFlow, Keras)

### Modèle d'Intelligence Artificielle
- Architecture basée sur ResNet-50 v1.5
- Pré-entraîné sur ImageNet-1k
- Performance exceptionnelle avec 94% de précision sur le jeu de test
- Optimisé pour la classification en temps réel

---

## Catégories de Classification
1. 🔋 Batterie
2. 📦 Carton
3. 🔗 Métal
4. 🍓 Organique
5. 🗳️ Papier
6. 🧳 Plastique
7. 🫙 Verre
8. 👖 Vêtements

---

## Applications et Impact

### Solutions Pratiques
- Automatisation du tri des déchets
- Développement de systèmes de recyclage intelligents
- Applications mobiles de reconnaissance des déchets
- Outils éducatifs interactifs

### Impact Environnemental
- Amélioration de l'efficacité du recyclage
- Réduction des erreurs de tri
- Sensibilisation environnementale
- Contribution à l'économie circulaire

---

## Innovation Technologique

EcoMindAI se distingue par son simulateur intégré qui permet de comparer les performances de différentes technologies d'IA, notamment :
- Notre modèle ResNet-50 optimisé
- ResNet de base
- YOLO
- LLMs (Llama 3.2)

---

## Accessibilité et Support

Le projet est conçu pour être facilement accessible aux chercheurs, développeurs et organisations environnementales. Pour toute question ou collaboration, contactez l'équipe à :
daniel.ferreira_lara@etu.sorbonne-universite.fr

---

## Vision Future

EcoMindAI aspire à devenir un outil de référence dans la lutte contre la pollution et la promotion du recyclage, en combinant innovation technologique et conscience environnementale. Le projet continue d'évoluer pour répondre aux défis croissants de la gestion des déchets à l'échelle mondiale.

---

## Instructions pour le repo GitHub

Suivez les étapes ci-dessous pour installer les dépendances, exécuter l'application localement ou contribuer au projet.

---

## **Accéder au site**
Si vous ne souhaitez pas exécuter le projet en local, vous pouvez visiter directement notre application hébergée en ligne :
[**Lien vers le site**](https://ecomind-ai.streamlit.app/)

---

#### **1. Cloner le dépôt**
Clonez ce dépôt sur votre machine locale :
```bash
git clone https://github.com/dan-lara/Garbage-Classifier.git
cd Garbage-Classifier
```

#### **2. Installer les dépendances**
Ce projet utilise Python 3.8 ou supérieur. Assurez-vous d'avoir installé [Python](https://www.python.org/) avant de continuer.

1. Créez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate   # Sur Windows : venv\Scripts\activate
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

#### **3. Lancer l'application**
Exécutez l'application Streamlit en local :
```bash
streamlit run app.py
```
Cela ouvrira une interface web dans votre navigateur.

---

## **Contribuer**
Votre contribution est la bienvenue ! Voici comment proposer des modifications ou des améliorations :

1. **Forkez ce dépôt**
   Cliquez sur le bouton "Fork" en haut de la page GitHub.

2. **Créez une branche**
   ```bash
   git checkout -b nouvelle-fonctionnalite
   ```

3. **Faites vos modifications**
   Ajoutez des fonctionnalités, corrigez des bugs ou améliorez la documentation.

4. **Soumettez une Pull Request**
   Une fois vos modifications prêtes, poussez votre branche sur votre fork et soumettez une Pull Request depuis l'interface GitHub.

---

## Ressources Utiles

- **Dataset** : [Normalized Garbage Dataset for ResNet](https://www.kaggle.com/datasets/danielferreiralara/normalized-garbage-dataset-for-resnet)

- **Modèle ResNet-50 finetuné** : [Garbage Classifier ResNet-50 Fine-Tuning](https://huggingface.co/dan-lara/Garbage-Classifier-Resnet-50-Finetuning)

---

Merci de votre intérêt pour ce projet ! 🌱