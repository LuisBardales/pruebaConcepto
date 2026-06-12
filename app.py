import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import urllib.request
import os
from PIL import Image

# ── Constantes ──────────────────────────────────────────────
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# ── Descarga del modelo si no existe ────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo MediaPipe Tasks..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

# ── Función para dibujar landmarks ──────────────────────────
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Convertir a proto para usar drawing_utils
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated,
            proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Etiqueta de mano (izq/der)
        h, w, _ = annotated.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        label = handedness[0].display_name
        cv2.putText(
            annotated,
            label,
            (int(min(x_coords) * w), int(min(y_coords) * h) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (88, 205, 54), 2, cv2.LINE_AA
        )

    return annotated

# ── UI Streamlit ─────────────────────────────────────────────
st.set_page_config(page_title="MediaPipe Tasks - Hand Landmarker", layout="centered")
st.title("🖐️ Hand Landmarker — MediaPipe Tasks")
st.caption("Detección de 21 puntos clave por mano usando la API moderna de MediaPipe.")

landmarker = load_model()

uploaded_file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)

    # Crear mp.Image desde numpy array
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    # Detección
    result = landmarker.detect(mp_image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_image, use_column_width=True)

    with col2:
        st.subheader("Landmarks detectados")
        if result.hand_landmarks:
            annotated = draw_landmarks_on_image(img_array, result)
            st.image(annotated, use_column_width=True)
            st.success(f"✅ {len(result.hand_landmarks)} mano(s) detectada(s)")

            # Tabla de handedness
            for i, h in enumerate(result.handedness):
                st.write(f"**Mano {i+1}:** {h[0].display_name} — confianza: `{h[0].score:.2f}`")
        else:
            st.image(pil_image, use_column_width=True)
            st.warning("⚠️ No se detectaron manos en la imagen.")