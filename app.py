import os
from dotenv import load_dotenv
import streamlit as st

from groq_handling import predict_class_llama

load_dotenv()

LOGO_PATH = os.getenv('LOGO_PATH')
ICON_PATH = os.getenv('ICON_PATH')
ECOMIND_LOGO_PATH = os.getenv('ECOMIND_LOGO_PATH')

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
from models_handling import *

custom_classes = ["🔋 Batterie", "📦 Carton", "🔗 Metal", "🍓 Organique", "📰 Papier", "🧃 Plastique", "🫙 Verre", "👖 Vetements"]
columns_history = ["Nom Image", "EcoMind AI", "ResNet 50", "Yolos Tiny", "Llama 90B", "Date/Heure"]
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame(columns=["Nom Image", "EcoMind AI", "ResNet 50", "Yolos Tiny", "Llama 90B", "Date/Heure"])

# Application principale
def page_analyse():
    st.logo(
        image = LOGO_PATH,
        icon_image = ICON_PATH,
        size = "large"
    ) 
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
        predicted_classes = []
        col1, col2, col3 = st.columns([0.5, 1, 0.6])
        # Afficher l'image soumise
        with col1:
            st.header("Image soumise")
            st.write("Dimensions de l'image :", image.size) 
            st.image(rounded_image(image), caption="Image soumise", use_container_width=True) 
            st.write("Dimensions de l'image :", image2.size)           
            st.image(rounded_image(image2), caption="Image soumise", use_container_width=True) 
                      
        
        if st.session_state["resnet 50"]:
            with col2:
                with st.container(border=True):
                    start_time = time.time()
                    predicted_class, probabilities = predict_resnet(image2, "microsoft/resnet-50")
                    inference_time = time.time() - start_time
                    reponse("ResNet 50", predicted_class, max(probabilities),
                            "https://huggingface.co/microsoft/resnet-50", inference_time)
                    # st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
                    # sorted_probs = sorted(zip(resnet_model.config.id2label.values(), probabilities), key=lambda x: x[1], reverse=True)
                    # plot_prob_graphs(sorted_probs, color="Blues")
                    predicted_classes.append({"predict_class": predicted_class, "prob":max(probabilities)})

        if st.session_state["EcoMind AI"]:
            with col2:
                with st.container(border=True):
                    start_time = time.time()
                    predicted_class, probabilities = predict_custom(image2)
                    inference_time = time.time() - start_time
                    reponse("EcoMind AI", predicted_class, max(probabilities),
                            "https://huggingface.co/dan-lara/Garbage-Classifier-Resnet-50-Finetuning", inference_time)
                    # st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
                    sorted_probs = sorted(zip(custom_classes, probabilities*100), key=lambda x: x[1], reverse=True)
                    # st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
                    # with st.expander("Probabilités par classe", expanded=True):
                    #     for label, prob in sorted_probs:
                    #         st.markdown(f"### {label} : {(prob*100):.2f} %")
                    plot_prob_graphs(sorted_probs, color=[[0, "#A8E6A1"], [1, "#1B5E20"]])
                    predicted_classes.append({"predict_class": predicted_class, "prob":max(probabilities)})
        
        if st.session_state["yolos tiny"]:
            with col3:
                with st.container(border=True):
                    start_time = time.time()
                    predicted_class, probabilities = predict_yolo(image, "hustvl/yolos-tiny")
                    inference_time = time.time() - start_time
                    reponse("Yolos Tiny", predicted_class, probabilities,
                            "https://huggingface.co/hustvl/yolos-tiny", inference_time)
                    # st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
                    # sorted_probs = sorted(zip(yolo_model.config.id2label.values(), probabilities), key=lambda x: x[1], reverse=True)
                    # plot_prob_graphs(sorted_probs, color="Reds")
                    predicted_classes.append({"predict_class": predicted_class, "prob":(probabilities)})

        if st.session_state["llama-3 11b"]:
            with col3:
                with st.container(border=True):
                    start_time = time.time()
                    predicted_class = predict_class_llama(image, "llama-3.2-11b-vision-preview")
                    inference_time = time.time() - start_time
                    # st.header("Llama 3.2 11b")
                    st.markdown("## [Llama 3.2 11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)", unsafe_allow_html=False)
                    st.metric("Classe prédite", predicted_class, delta=None, delta_color="normal", border=False)
                    # st.markdown(f"###### Temps d'inférence : {inference_time:.2f} secondes", unsafe_allow_html=False)
                    st.write(f"Temps d'inférence : {inference_time:.2f} secondes")
                    # st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
        
        if st.session_state["llama-3 90b"]:
            with col3:
                with st.container(border=True):
                    start_time = time.time()
                    predicted_class = predict_class_llama(image, "llama-3.2-90b-vision-preview")
                    inference_time = time.time() - start_time
                    # st.header("Llama 3.2 90B")
                    st.markdown("## [Llama 3.2 90B](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)", unsafe_allow_html=False)
                    st.metric("Classe prédite", predicted_class, delta=None, delta_color="normal", border=False)
                    # st.markdown(f"###### Temps d'inférence : {inference_time:.2f} secondes", unsafe_allow_html=False)
                    st.write(f"Temps d'inférence : {inference_time:.2f} secondes")
                    # st.caption(f"Temps d'inférence : {inference_time:.2f} secondes")
                    predicted_classes.append({"predict_class": predicted_class})

        add_prediction_to_history(uploaded_file.name, predicted_classes)

def add_prediction_to_history(image_name, predicted_classes):
    """
    Ajoute une nouvelle prédiction à l'historique.
    """
    if "history" not in st.session_state:
        st.session_state["history"] = pd.DataFrame(columns=columns_history)

    new_entry = {
        "Nom Image": image_name,
        "EcoMind AI": predicted_classes[1],
        "ResNet 50": predicted_classes[0],
        "Yolos Tiny": predicted_classes[2],
        "Llama 90B": predicted_classes[3],
        "Date/Heure": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.session_state["history"] = pd.concat([st.session_state["history"], pd.DataFrame([new_entry])], ignore_index=True)

def page_historique():
    """
    Historique des Prédictions
    """
    # st.image(
    #     image="/home/portable016/magnet/SU/AI/app/ecomind_logo.png",
    #     width=800, 
    # )
    st.title("Historique des Prédictions")
    st.write("Retrouvez la liste des images soumises et leur classe prédite.")

    # Historique des prédictions
    hist_df = st.session_state["history"]
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True)

        # Bouton de téléchargement CSV
        csv_data = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger l'historique en CSV",
            data=csv_data,
            file_name="historique_predictions.csv",
            mime="text/csv"
        )

        # Bouton de réinitialisation
        if st.button("Réinitialiser l'historique"):
            st.session_state["history"] = pd.DataFrame(columns=columns_history)
            st.success("Historique réinitialisé !")
    else:
        st.info("Aucune prédiction n'a été enregistrée pour le moment.")

def page_a_propos():
    st.logo(
        image = LOGO_PATH,
        icon_image = ICON_PATH,
        size = "large"
    ) 
    st.title("À Propos")
    
    st.markdown("""
    # Comparaison entre Modèles de Classification des Déchets

    Bienvenue sur notre plateforme web, qui vous permet d'explorer et de comparer différents modèles d'intelligence artificielle pour la classification des déchets. Cette application interactive vise à présenter les performances et les caractéristiques des modèles les plus avancés dans ce domaine, en se basant sur le « Garbage Classification Dataset ».

    ## Modèles Comparés
    """)
    with st.container():
        col1, col2 = st.columns([1, 1])
        col1.markdown("""
        ### ResNet-50
        - **Architecture** : Modèle convolutionnel pré-entraîné sur ImageNet.
        - **Taux de Précision** : 92 % sur le jeu de test.
        - **Points Forts** : Performances élevées, polyvalence.
        - **Limitations** : Sensible à la qualité des images.
        - **Problème** : Généraliste avec 1000 classes d'images
        - **Lien vers le modèle** : [ResNet-50](https://huggingface.co/microsoft/resnet-50)

        ### ResNet-50 Fine-Tuné
        - **Architecture** : Modèle convolutionnel avec des connexions résiduelles, pré-entraiîné sur ImageNet.
        - **Taux de Précision** : 94 % sur le jeu de test.
        - **Points Forts** : Performances équilibrées, facile à déployer.
        - **Limitations** : Moins adapté à des classes hors dataset.
        - **Lien vers le modèle** : [Garbage Classifier ResNet-50 Fine-Tuning](https://huggingface.co/dan-lara/Garbage-Classifier-Resnet-50-Finetuning)
        """)
        col2.markdown("""
        ### YOLO (You Only Look Once) Tiny
        - **Architecture** : Modèle à détection rapide et précise.
        - **Taux de Précision** : Variable selon la version et les ajustements.
        - **Points Forts** : Idéal pour la détection en temps réel.
        - **Limitations** : Sensible à la qualité des images.
        - **Lien vers le modèle** : [YOLOs Tiny](https://huggingface.co/hustvl/yolos-tiny)

        ### Modèle Multimodal (LLM)
        - **Architecture** : Basé sur des modèles de langage étendus à la vision.
        - **Modèle** : Llama 3.2 11B et Llama 3.2 90B.
        - **Points Forts** : Compréhension contextuelle avancée.
        - **Limitations** : Consommation élevée de ressources.
        - **Lien vers le modèle** : [Llama 3.2 11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision) et [Llama 3.2 90B](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)
        """)
    st.markdown("""
    ## Fonctionnement du Site

    1. **Téléchargez le Dataset**
    - Accédez au dataset via [Normalized Garbage Dataset for ResNet](https://www.kaggle.com/datasets/danielferreiralara/normalized-garbage-dataset-for-resnet).

    2. **Chargez une Image**
    - Importez une photo de déchet via l'interface.

    3. **Obtenez les Résultats**
    - Visualisez les prédictions avec des scores de confiance et comparez les résultats entre modèles.

    4. **Explorez les Statistiques**
    - Consultez les graphiques et les rapports sur les performances des modèles par classe.

    ## Applications Potentielles

    - **Environnement** : Automatisation du tri des déchets.
    - **Recherche** : Développement de nouveaux modèles plus précis.
    - **Éducation** : Sensibilisation au recyclage via des outils interactifs.

    ## Découvrez Maintenant

    Visitez notre site et testez nos outils pour contribuer activement à une meilleure gestion des déchets et à la protection de notre environnement.         
    """
    )
    # st.markdown("Cette application a été développée par [Dan Lara](https://github.com/dan-lara/)")

def main():
    # st.image(os.getenv('LOGO_PATH'), width=1000)
    pg = st.navigation([
        st.Page(page_analyse,       icon=":material/search:",   title="Analyse"),
        st.Page(page_historique,    icon=":material/history:",   title="Historique"),
        st.Page(page_a_propos,       icon=":material/info:",     title="À Propos")
    ])
    pg.run()

if __name__ == "__main__":
    main()

