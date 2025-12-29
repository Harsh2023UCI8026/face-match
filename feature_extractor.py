import os
import numpy as np
from deepface import DeepFace
from joblib import dump

DATASET_PATH = "data"   # folder containing face images

def extract_embeddings():
    embeddings = []
    names = []

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)

            try:
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=False
                )

                embeddings.append(result[0]["embedding"])
                names.append(person)

            except Exception as e:
                print(f"Skipped {img_path}: {e}")

    # 🔥 Compress embeddings
    embeddings = np.array(embeddings, dtype="float16")

    dump((embeddings, names), "image_embeddings.joblib", compress=3)
    print("✅ Embeddings saved as image_embeddings.joblib")


if __name__ == "__main__":
    extract_embeddings()


