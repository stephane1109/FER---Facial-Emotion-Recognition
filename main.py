##########################################
# Projet : FER - Facial-Emotion-Recognition
# Auteur : Stéphane Meurisse
# Contact : stephane.meurisse@example.com
# Site Web : https://www.codeandcortex.fr
# LinkedIn : https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/
# Date : 4 octobre 2024
# Licence : Ce programme est un logiciel libre : vous pouvez le redistribuer selon les termes de la Licence Publique Générale GNU v3
##########################################

# pip install opencv-python streamlit fer pandas matplotlib tensorflow xlsxwriter
# pip install tensorflow-gpu ##### si gpu
# pip install watchdog
# pip install yt_dlp
# pip install altair vega_datasets
# pip install vl-convert-python
# pip install altair_saver

import os
import streamlit as st
import yt_dlp as youtube_dl
import cv2
from fer import Video, FER
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import logging
import xlsxwriter

# Configuration du logger pour Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Fonction pour afficher les logs dans Streamlit
def afficher_log(message):
    logger.info(message)
    st.text(message)

# Fonction pour télécharger la vidéo complète depuis YouTube avec yt_dlp
def telecharger_video_youtube(url, output_path):
    ydl_opts = {
        'format': 'mp4/best',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'noplaylist': True
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            afficher_log(f"Téléchargement de la vidéo depuis {url}...")
            ydl.download([url])
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None)
            fichier_video = os.path.join(output_path, f"{video_title}.mp4")
            afficher_log(f"Vidéo téléchargée et sauvegardée sous : {fichier_video}")
            return fichier_video
    except Exception as e:
        st.error(f"Erreur lors du téléchargement de la vidéo YouTube : {e}")
        return None

# Fonction pour dessiner les annotations manuellement sur une image (cadres et scores d'émotions)
def dessiner_annotations(image, emotions):
    height, width, _ = image.shape
    cv2.rectangle(image, (50, 50), (width - 50, height - 50), (0, 255, 0), 2)
    for idx, (emotion, score) in enumerate(emotions.items()):
        text = f"{emotion}: {score:.2f}"
        cv2.putText(image, text, (50, 50 + 30 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

# Fonction pour extraire une image par seconde, analyser les émotions et afficher les informations visuelles
def extraire_images_analyse_emotions(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps)

    current_frame = 0
    image_num = 0
    sous_repertoire_images = os.path.join(output_path, "images_annotées")
    if not os.path.exists(sous_repertoire_images):
        os.makedirs(sous_repertoire_images)

    afficher_log(f"Extraction des images toutes les secondes...")
    afficher_log(f"Vidéo: {fps} fps, {total_frames} frames, {total_frames / fps:.2f} seconds")

    detecteur_visage = FER(mtcnn=True)
    fichier_excel = os.path.join(output_path, "emotions_scores.xlsx")
    workbook = xlsxwriter.Workbook(fichier_excel)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Image')
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for idx, emotion in enumerate(emotions):
        worksheet.write(0, idx + 1, emotion)

    row = 1
    emotion_data = {emotion: [] for emotion in emotions}

    while current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % interval_frames == 0:
            image_num += 1
            emotion = detecteur_visage.detect_emotions(frame)
            if emotion:
                frame_annotated = dessiner_annotations(frame, emotion[0]['emotions'])
                image_path = os.path.join(sous_repertoire_images, f"image_{image_num}.png")
                cv2.imwrite(image_path, frame_annotated)

                emotion_scores = emotion[0]['emotions']
                worksheet.write(row, 0, f"image_{image_num}.png")
                for idx, emotion in enumerate(emotions):
                    worksheet.write(row, idx + 1, emotion_scores[emotion])
                    emotion_data[emotion].append(emotion_scores[emotion])
                row += 1
        current_frame += 1

    cap.release()
    workbook.close()

    return pd.DataFrame(emotion_data)

# Interface Streamlit
st.title("FER - Facial Emotion Recognition")
st.markdown("<h5 style='text-align: center;'>www.codeandcortex.fr</h5>", unsafe_allow_html=True)

# Répertoire de sortie pour les vidéos et résultats
st.header("Répertoire de sortie")
repertoire_sortie = st.text_input("Spécifiez le répertoire de sortie", "output")

if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Variables initiales
fichier_video = None

# Section 1 : Téléchargement de la vidéo YouTube
st.header("Étape 1 : Si vous n'avez pas encore de vidéo au format MP4, c'est ici !")
url_video = st.text_input("Entrez l'URL de la vidéo YouTube")
if st.button("Télécharger la vidéo depuis YouTube"):
    fichier_video = telecharger_video_youtube(url_video, repertoire_sortie)

# Section 2 : Drag and Drop
st.header("Étape 2 : Drag and Drop votre fichier MP4")
fichier_video_local = st.file_uploader("Ou déposez directement un fichier vidéo local (MP4)", type=["mp4"])
if fichier_video_local:
    fichier_video_path = os.path.join(repertoire_sortie, fichier_video_local.name)
    with open(fichier_video_path, "wb") as f:
        f.write(fichier_video_local.getbuffer())
    fichier_video = fichier_video_path
    st.write(f"Vidéo locale enregistrée avec succès : {fichier_video_path}")

# Section 3 : Extraction des images et analyse des émotions
if fichier_video:
    st.header("Étape 3 : Le script analysera une image par seconde pour détecter les émotions.")
    if st.button("Lancer l'analyse des émotions et extraire les images"):
        df = extraire_images_analyse_emotions(fichier_video, repertoire_sortie)

        # Si les données d'émotions sont disponibles, on peut afficher les graphiques
        if not df.empty:
            # Graphique 1 : Line Chart with Datum (secondes remplacées)
            st.header("Graphique 1 : Évolution des émotions - Line Chart ")
            seconds = list(range(len(df)))  # Remplacement par les secondes
            df_reset = df.reset_index().melt('index', var_name='Emotion', value_name='Score')
            line_chart = alt.Chart(df_reset).mark_line().encode(
                x=alt.X('index:Q', title='Secondes'),
                y=alt.Y('Score:Q', title='Score d\'émotion'),
                color='Emotion:N'
            )
            st.altair_chart(line_chart, use_container_width=True)
            line_chart.save(os.path.join(repertoire_sortie, "line_chart_emotions.png"))

            # Graphique 2 : Streamgraph
            st.header("Graphique 2 : Autre représentation du graphique 1 - Streamgraph des émotions")
            streamgraph = alt.Chart(df_reset).mark_area().encode(
                alt.X('index:Q', title='Secondes'),
                alt.Y('Score:Q', stack='center', title='Score d\'émotion'),
                alt.Color('Emotion:N'),
                tooltip=[alt.Tooltip('index:Q', title='Secondes'), alt.Tooltip('Score:Q', title='Score d\'émotion')]
            ).interactive()
            st.altair_chart(streamgraph, use_container_width=True)
            streamgraph.save(os.path.join(repertoire_sortie, "streamgraph_emotions.png"))

            # Graphique 3 : Bar Chart with Rounded Edges (Barres avec arrondis)
            st.header("Graphique 3 : Bar Chart")
            st.caption("Ce graphique affiche les scores d'émotions totaux pour chaque émotion. "
                       "Chaque barre représente le score total d'une émotion pour toutes les images extraites.")
            bar_chart = alt.Chart(df_reset).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x=alt.X('Emotion:N', title='Émotion'),
                y=alt.Y('Score:Q', title='Score total'),
                color='Emotion:N'
            )
            st.altair_chart(bar_chart, use_container_width=True)
            bar_chart.save(os.path.join(repertoire_sortie, "bar_chart_rounded_edges.png"))

            # Graphique 4 : Courbes lissées sur 5 secondes (Lissage sur 5 secondes)
            st.header("Graphique 4 : Courbes lissées sur 5 secondes")
            st.caption("Le DataFrame est lissé sur une fenêtre de 5 secondes avec .rolling(window=5).mean(). "
                       "Cela signifie que chaque valeur sera la moyenne des 5 dernières secondes pour lisser les données.")
            df_smoothed = df.rolling(window=5).mean()  # Lissage sur une fenêtre de 5 secondes
            df_reset_smoothed = df_smoothed.reset_index().melt('index', var_name='Emotion', value_name='Score')
            line_chart_smoothed = alt.Chart(df_reset_smoothed).mark_line().encode(
                x=alt.X('index:Q', title='Secondes'),
                y=alt.Y('Score:Q', title='Score d\'émotion lissé (5 secondes)'),
                color='Emotion:N'
            )
            st.altair_chart(line_chart_smoothed, use_container_width=True)
            line_chart_smoothed.save(os.path.join(repertoire_sortie, "emotions_lissees_5s.png"))

            # Graphique supplémentaire : Graphique de flux des émotions lissées sur 5 secondes (streamgraph)
            st.header("Graphique 5 : Autre représentation du graphique 4 - Streamgraph lissées sur 5 secondes")
            st.caption("Ce graphique de flux représente les données d'émotions lissées sur 5 secondes. "
                       "Il permet de visualiser les variations des émotions de manière plus fluide.")
            streamgraph_smoothed = alt.Chart(df_reset_smoothed).mark_area().encode(
                alt.X('index:Q', title='Secondes'),
                alt.Y('Score:Q', stack='center', title='Score d\'émotion lissé (5 secondes)'),
                alt.Color('Emotion:N')
            )
            st.altair_chart(streamgraph_smoothed, use_container_width=True)
            streamgraph_smoothed.save(os.path.join(repertoire_sortie, "streamgraph_lisse_5s.png"))

            afficher_log(f"Extraction et analyse des images terminées.")
            afficher_log(f"Graphiques exportés dans le répertoire : {repertoire_sortie}")
