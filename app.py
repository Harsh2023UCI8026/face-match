import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import os
import tempfile

st.set_page_config(page_title="Celebrity Look-Alike", layout="centered")

st.title("🎭 Celebrity Look-Alike Finder")

UPLOAD_DIR = "data"

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        user_img = face_recognition.load_image_file(temp.name)
        user_encoding = face_recognition.face_encodings(user_img)

    if not user_encoding:
        st.error("No face detected.")
        st.stop()

    user_encoding = user_encoding[0]

    best_match = None
    best_distance = 1.0

    for person in os.listdir(UPLOAD_DIR):
        person_dir = os.path.join(UPLOAD_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for img in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img)
            known_img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(known_img)

            if encodings:
                dist = np.linalg.norm(encodings[0] - user_encoding)
                if dist < best_distance:
                    best_distance = dist
                    best_match = person

    if best_match:
        st.success(f"🎉 You look like **{best_match}**!")
    else:
        st.warning("No matching face found.")



