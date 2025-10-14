import streamlit as st
import cv2
import numpy as np
from utils import extract_face_landmarks, compare_expressions
from PIL import Image

st.title("Expression Image Matcher")

# Upload reference image
st.sidebar.header("Reference Image")
ref_img_file = st.sidebar.file_uploader("Upload a reference face image", type=["jpg", "png"])
reference_landmarks = None
if ref_img_file:
    ref_img = np.array(Image.open(ref_img_file))
    reference_landmarks = extract_face_landmarks(ref_img)
    st.sidebar.image(ref_img, caption="Reference Image")

# Webcam capture
st.header("Match Your Expression")
run = st.checkbox("Turn on webcam")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0) if run else None

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    user_landmarks = extract_face_landmarks(frame)
    match_score = 0
    if reference_landmarks is not None and user_landmarks is not None:
        match_score = compare_expressions(reference_landmarks, user_landmarks)

    FRAME_WINDOW.image(frame_rgb)
    st.write(f"Match Score: {match_score:.2f}")
    if st.button("Stop Webcam"):
        run = False

if cap is not None:
    cap.release()
    cv2.destroyAllWindows()
