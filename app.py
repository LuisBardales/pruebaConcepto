import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

st.title("🖐️ MediaPipe - Detección de Manos")
st.write("Sube una imagen para detectar puntos de la mano.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        results = hands.process(img_array)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img_array, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.image(img_array, caption="Landmarks detectados", use_column_width=True)
    else:
        st.warning("No se detectaron manos en la imagen.")