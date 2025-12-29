import streamlit as st
import face_recognition
import os
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="Celebrity Look-Alike", layout="centered")

st.title("🎭 Celebrity Look-Alike Finder")

UPLOAD_DIR = "data"

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        user_img = face_recognition.load_image_file(temp.name)
        user_enc = face_recognition.face_encodings(user_img)

    if not user_enc:
        st.error("No face detected.")
        st.stop()

    user_enc = user_enc[0]

    best_match = None
    best_distance = 1.0

    for person in os.listdir(UPLOAD_DIR):
        person_path = os.path.join(UPLOAD_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            known_img = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(known_img)
            if not enc:
                continue

            distance = np.linalg.norm(enc[0] - user_enc)
            if distance < best_distance:
                best_distance = distance
                best_match = person

    if best_match:
        st.success(f"🎉 You look like **{best_match}**")
    else:
        st.warning("No match found")
