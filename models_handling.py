import os
import requests
import time
import warnings
import torch
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from dotenv import load_dotenv
from transformers import (
    ResNetForImageClassification,
    YolosForObjectDetection,
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoImageProcessor,
    YolosImageProcessor
)

load_dotenv()

CUSTOM_MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH")
CUSTOM_MODEL_URL = os.getenv("CUSTOM_MODEL_URL")
LOGO_PATH = os.getenv('LOGO_PATH')
ICON_PATH = os.getenv('ICON_PATH')

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")  # Ignorer les avertissements de PyTorch

# Fonction pour charger les modèles
@st.cache_resource
def load_huggingface_model(model_name):
    """
    Charger un modèle Hugging Face à partir de son nom.
    """
    
    if model_name == "microsoft/resnet-50":
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    elif model_name == "hustvl/yolos-tiny":
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    else:
        model = AutoModelForImageClassification.from_pretrained(model_name)

    # extractor = AutoFeatureExtractor.from_pretrained(model_name)
    extractor = None
    if model_name == "hustvl/yolos-tiny":
        processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
    return model, extractor, processor

# Fonction pour charger les modèles
@st.cache_resource
def load_huggingface_resnet(model_name):
    """
    Charger un modèle Hugging Face à partir de son nom.
    """
    model = ResNetForImageClassification.from_pretrained(model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, extractor, processor


@st.cache_resource
def load_custom_model_web():
    """
    Load the custom ResNet model.
    """
    # URL of the model file
    model_url = CUSTOM_MODEL_URL
    local_model_path = "pytorch_model_web.pth"
    
    # Download the model if not already present locally
    if not os.path.exists(local_model_path):
        with st.spinner("Downloading the custom model..."):
            response = requests.get(model_url)
            response.raise_for_status()  # Raise an error if the download fails
            with open(local_model_path, "wb") as f:
                f.write(response.content)
    
    # Load the model
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 8)  # 8 classes
    model.load_state_dict(torch.load(local_model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model

@st.cache_resource
def load_custom_model():
    """
    Charger le modèle personnalisé ResNet.
    """
    try:
        model, _, _ = load_huggingface_model("microsoft/resnet-50")#AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 8)  # 8 classes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=device), strict=False)
        model.eval()
        return model
    except Exception as e:
        print(e)
        return load_custom_model_web()

# Définir les classes pour le modèle personnalisé
custom_classes = ["🔋 Batterie", "📦 Carton", "🔗 Metal", "🍓 Organique", "📰 Papier", "🧃 Plastique", "🫙 Verre", "👖 Vetements"]

    
# Préparer les transformations d'image pour le modèle personnalisé
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# custom_transform = transforms.Compose([
#     transforms.RandomRotation(20),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
def tensor_to_pil(tensor):
    """
    Convertit le tenseur normalisé en image PIL.
    Ici, pas d'inverse normalization. On utilise un min-max scaling pour exploiter toute la gamme [0,1].
    Le résultat sera visuellement très différent de l'original.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    scaled = (tensor - min_val) / (max_val - min_val + 1e-5)
    pil_image = transforms.ToPILImage()(scaled)
    return pil_image

# Fonction de prédiction pour le modèle personnalisé
def predict_custom(image):
    """
    Effectuer une prédiction avec le modèle personnalisé.
    """
    
    input_tensor = custom_transform(image).unsqueeze(0)  # Ajouter une dimension batch
    with torch.no_grad():
        outputs = custom_model(input_tensor)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy().flatten()
    _, predicted = torch.max(logits, 1)
    return custom_classes[predicted.item()], probabilities

def predict_resnet(image, model_name):
    """
    Effectuer une prédiction avec un modèle Hugging Face.
    """
    model, extractor, processor = load_huggingface_resnet(model_name)

    # processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy().flatten()
    predicted_label = model.config.id2label[predicted_label]
    return predicted_label, probabilities

def predict_yolo(image, model_name):
    """
    Effectuer une prédiction avec un modèle Hugging Face.
    """
    model, extractor, processor = load_huggingface_model(model_name)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    max_score = 0
    predicted_label = None
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # print(
        #     f"Detected {model.config.id2label[label.item()]} with confidence "
        #     f"{round(score.item(), 3)} at location {box}"
        # )
        if score.item() > max_score:
            predicted_label = model.config.id2label[label.item()]
            max_score = score.item()

    return predicted_label, max_score


def reponse(model, label, score, link, time: float = 0.0):
    block = st.container()
    with block:
        # st.header(model)
        st.markdown(f"## [{model}]({link})", unsafe_allow_html=False)
        cols = st.columns([2, 1])
        cols[0].metric("Classe prédite", label, delta=None, delta_color="normal", border=False)
        cols[1].metric("Confiance", f"{(score*100):.2f} %", delta=None, delta_color="normal", border=False)
        if time:
            # st.markdown(f"###### Temps d'inférence : {time:.2f} secondes", unsafe_allow_html=False)
            st.write(f"Temps d'inférence : {time:.2f} secondes")
            # st.caption(f"Temps d'inférence : {time:.2f} secondes")
    return block

def plot_prob_graphs(probabilities, top_k=7, color=[[0, "#A8E6A1"], [1, "#1B5E20"]]):
    try:

        df_probs = pd.DataFrame(probabilities[0:top_k], columns=["Classe", "Probabilité"])
        df_probs = df_probs.sort_values("Probabilité", ascending=True)
        
        fig = px.bar(
            df_probs, 
            x="Probabilité", 
            y="Classe", 
            orientation='h', 
            color="Probabilité",
            color_continuous_scale= color,
            barmode='group',
        )
        
        fig.update_layout(
            title="Classification des Déchets",
            title_font=dict(size=20, color="#1B5E20"),  # Verde escuro para o título
            font=dict(size=16, color="#1B5E20"),  # Fonte verde escura
            xaxis=dict(
                title="Probabilité", 
                title_font=dict(color="#1B5E20"), 
                tickfont=dict(color="#1B5E20")
            ),
            yaxis=dict(
                title="Classe", 
                title_font=dict(color="#1B5E20"), 
                tickfont=dict(color="#1B5E20")
            ),
            coloraxis_colorbar=dict(title="Probabilité", title_font=dict(color="#1B5E20")),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Installez Plotly pour afficher le graphique en barres.")

def rounded_image(image):
    """
    Ajouter des coins arrondis à une image.
    """
    image = image.convert("RGBA")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, *image.size), radius=10, fill=255)
    
    # Aplicar a máscara na imagem
    image = ImageOps.fit(image, mask.size)
    image.putalpha(mask)
    return image

# Charger les modèles
with st.spinner('Chargement des modèles en cours...'):
    resnet_model, resnet_extractor, resnet_processor = load_huggingface_model("microsoft/resnet-50")
    yolo_model, yolo_extractor, ryolo_processor  = load_huggingface_model("hustvl/yolos-tiny")
    custom_model = load_custom_model()