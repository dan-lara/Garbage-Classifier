from dotenv import load_dotenv
import streamlit as st
import os
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, AutoImageProcessor,ResNetForImageClassification, YolosForObjectDetection, YolosImageProcessor
from torchvision import transforms
from PIL import Image
import time
import warnings
import requests
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

from util import predict_class_llama

load_dotenv()
CUSTOM_MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH")
CUSTOM_MODEL_URL = os.getenv("CUSTOM_MODEL_URL")
st.set_page_config(
    page_title="Classification des Déchets - EcoMind AI",
    page_icon=":recycle:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/dan-lara/Garbage-Classifier',
        'Report a bug': "https://github.com/dan-lara/Garbage-Classifier/issues",
        'About': "# C'est le *meilleur AI trieur* de déchets que vous verrez aujourd'hui.!"
    }
)

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
    transforms.ToTensor(),
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


def reponse(model, label, score, link):
    block = st.container()
    with block:
        # st.header(model)
        st.markdown(f"## [{model}]({link})", unsafe_allow_html=False)
        cols = st.columns([2, 1])
        cols[0].metric("Classe prédite", label, delta=None, delta_color="normal", border=True)
        cols[1].metric("Confiance", f"{(score*100):.2f} %", delta=None, delta_color="normal", border=True)
        
    return block

# Charger les modèles
with st.spinner('Chargement des modèles en cours...'):
    resnet_model, resnet_extractor, resnet_processor = load_huggingface_model("microsoft/resnet-50")
    yolo_model, yolo_extractor, ryolo_processor  = load_huggingface_model("hustvl/yolos-tiny")
    custom_model = load_custom_model()

# Application principale
def main():
    LOGO_PATH = os.getenv('LOGO_PATH')
    ICON_PATH = os.getenv('ICON_PATH')
    st.logo(
        image = LOGO_PATH,
        icon_image = ICON_PATH,
        size = "large") 
    st.title("♻️ Classification des Déchets ")
    st.sidebar.title("Choisir Modèles : ")
    checks = []
    for model in ["resnet 50", "EcoMind AI", "yolos tiny", "llama-3 11b", "llama-3 90b"]:
        checks.append(st.sidebar.checkbox(model, key=model, value=True))
        # st.session_state.update({model: checks[-1]})
    # st.markdown("### Soumettez une image pour la classifier selon différents modèles d'IA")

    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image2 = tensor_to_pil(custom_transform(image))
        # image = image.resize((224, 224))

        col1, col2, col3 = st.columns([1, 1, 1])
        # Afficher l'image soumise
        with col1:
            st.header("Image soumise")
            with st.expander("Afficher les détails de l'image"):
                st.write("Dimensions de l'image :", image2.size)
                st.write("Mode de couleur :", image.mode)
            st.image(image2, caption="Image soumise", use_container_width=True)            
        
        if st.session_state["resnet 50"]:
            with col2:
                start_time = time.time()
                predicted_class, probabilities = predict_resnet(image2, "microsoft/resnet-50")
                inference_time = time.time() - start_time
                reponse("ResNet 50", predicted_class, max(probabilities), "https://huggingface.co/microsoft/resnet-50")
                st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
        
        if st.session_state["EcoMind AI"]:
            with col2:
                start_time = time.time()
                predicted_class, probabilities = predict_custom(image2)
                inference_time = time.time() - start_time
                reponse("EcoMind AI", predicted_class, max(probabilities), "https://huggingface.co/dan-lara/Garbage-Classifier-Resnet-50-Finetuning")
                sorted_probs = sorted(zip(custom_classes, probabilities), key=lambda x: x[1], reverse=True)
                st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
                with st.expander("Probabilités par classe", expanded=True):
                    for label, prob in sorted_probs:
                        st.markdown(f"### {label} : {(prob*100):.2f} %")
        
        if st.session_state["yolos tiny"]:
            with col3:
                start_time = time.time()
                predicted_class, probabilities = predict_yolo(image, "hustvl/yolos-tiny")
                inference_time = time.time() - start_time
                reponse("Yolos Tiny", predicted_class, probabilities, "https://huggingface.co/hustvl/yolos-tiny")
                st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")

        if st.session_state["llama-3 11b"]:
            with col3:
                start_time = time.time()
                predicted_class = predict_class_llama(image, "llama-3.2-11b-vision-preview")
                inference_time = time.time() - start_time
                # st.header("Llama 3.2 11b")
                st.markdown("## [Llama 3.2 11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)", unsafe_allow_html=False)
                st.metric("Classe prédite", predicted_class, delta=None, delta_color="normal", border=True)
                st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
        
        if st.session_state["llama-3 90b"]:
            with col3:
                start_time = time.time()
                predicted_class = predict_class_llama(image, "llama-3.2-90b-vision-preview")
                inference_time = time.time() - start_time
                # st.header("Llama 3.2 90B")
                st.markdown("## [Llama 3.2 90B](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)", unsafe_allow_html=False)
                st.metric("Classe prédite", predicted_class, delta=None, delta_color="normal", border=True)
                st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")

        
        

if __name__ == "__main__":
    main()

