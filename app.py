import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
from PIL import Image, ImageDraw

# ── Constantes ──────────────────────────────────────────────
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Conexiones entre landmarks (21 puntos de la mano)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # Pulgar
    (0,5),(5,6),(6,7),(7,8),         # Índice
    (0,9),(9,10),(10,11),(11,12),    # Medio
    (0,13),(13,14),(14,15),(15,16),  # Anular
    (0,17),(17,18),(18,19),(19,20),  # Meñique
    (5,9),(9,13),(13,17),            # Nudillos
]

HAND_COLORS = {
    "Left":  {"joint": (255, 80,  80),  "line": (200, 40,  40)},
    "Right": {"joint": (80,  180, 255), "line": (40,  120, 220)},
}

# ── Carga del modelo ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Descargando modelo (~4 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)

# ── Dibujo con Pillow (sin cv2 ni libGL) ─────────────────────
def draw_landmarks_pil(pil_image, detection_result):
    img = pil_image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness  = detection_result.handedness[idx][0].display_name
        colors      = HAND_COLORS.get(handedness, HAND_COLORS["Right"])
        pts         = [(lm.x * w, lm.y * h) for lm in hand_landmarks]

        # Líneas de conexión
        for a, b in HAND_CONNECTIONS:
            draw.line([pts[a], pts[b]], fill=colors["line"], width=3)

        # Puntos articulares
        r = 5
        for x, y in pts:
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=colors["joint"], outline="white", width=1)

        # Etiqueta (izq/der)
        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        draw.text((min_x, max(min_y - 22, 0)), handedness,
                  fill=colors["joint"])

    return img

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="Hand Landmarker", layout="centered")
st.title("🖐️ Hand Landmarker — MediaPipe Tasks")
st.caption("Detección de 21 puntos clave por mano · sin dependencias de OpenCV")

landmarker = load_model()

uploaded_file = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_image))
    result    = landmarker.detect(mp_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(pil_image, use_column_width=True)

    with col2:
        st.subheader("Landmarks detectados")
        if result.hand_landmarks:
            annotated = draw_landmarks_pil(pil_image, result)
            st.image(annotated, use_column_width=True)
            st.success(f"✅ {len(result.hand_landmarks)} mano(s) detectada(s)")
            for i, hand in enumerate(result.handedness):
                st.write(f"**Mano {i+1}:** {hand[0].display_name} — "
                         f"confianza: `{hand[0].score:.2f}`")
        else:
            st.image(pil_image, use_column_width=True)
            st.warning("⚠️ No se detectaron manos en la imagen.")