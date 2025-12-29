import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile

st.set_page_config(page_title="Celebrity Look-Alike", layout="centered")

st.title("🎭 Celebrity Look-Alike Finder")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Finding your celebrity twin..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            img_path = tmp.name

        try:
            result = DeepFace.find(
                img_path=img_path,
                db_path="data",   # folder with actor images
                enforce_detection=False,
                model_name="VGG-Face"
            )

            if len(result) > 0 and len(result[0]) > 0:
                top = result[0].iloc[0]
                st.success("🎉 Match Found!")
                st.write(f"**Celebrity:** {top['identity'].split('/')[-2]}")
                st.write(f"**Similarity:** {round((1 - top['distance']) * 100, 2)}%")
            else:
                st.warning("No close match found.")

        except Exception as e:
            st.error("Something went wrong")
            st.code(str(e))



