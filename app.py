from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import tensorflow as tf
from pathlib import Path

from utils.image_preprocess import preprocess_image
from utils.symptom_encoder import encode_symptoms
from utils.fusion import fuse_predictions

# -------------------------
# Path Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# -------------------------
# App
# -------------------------
app = FastAPI(title="Skin AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Models
# -------------------------
try:
    image_model = tf.keras.models.load_model(
        MODEL_DIR / "best_model.h5",
        compile=False
    )
    symptom_model = tf.keras.models.load_model(
        MODEL_DIR / "checkbox_model_10class.h5",
        compile=False
    )
except Exception as e:
    raise RuntimeError(f"Model load error: {e}")

# -------------------------
# Load Classes
# -------------------------
with open(MODEL_DIR / "classes.json", "r", encoding="utf-8") as f:
    classes = json.load(f)

IMAGE_CLASSES = classes["image_model"]["class_names"]
SYMPTOM_CLASSES = classes["checkbox_model"]["class_names"]
IMAGE_ID2LABEL = classes["image_model"]["id_to_label"]

# -------------------------
# Root
# -------------------------
@app.get("/")
def root():
    return {
        "status": "Skin AI Backend is running",
        "image_classes": len(IMAGE_CLASSES),
        "symptom_classes": len(SYMPTOM_CLASSES)
    }

# -------------------------
# Predict
# -------------------------
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    symptoms: str = Form(""),
    areas: str = Form("")
):
    try:
        # ---------- Image ----------
        image_bytes = await image.read()
        img_tensor = preprocess_image(image_bytes)
        img_pred = image_model.predict(img_tensor, verbose=0)[0]

        # ---------- Symptoms ----------
        symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
        area_list = [a.strip() for a in areas.split(",") if a.strip()]

        sym_vec = encode_symptoms(
            selected_symptoms=symptom_list,
            selected_areas=area_list,
            schema_path=str(MODEL_DIR / "symptom_schema.json")
        )

        sym_pred = symptom_model.predict(sym_vec, verbose=0)[0]

        # ---------- Align classes (10 → 11) ----------
        sym_full = np.zeros(len(IMAGE_CLASSES), dtype="float32")
        for i, cls in enumerate(SYMPTOM_CLASSES):
            if cls in IMAGE_CLASSES:
                sym_full[IMAGE_CLASSES.index(cls)] = sym_pred[i]

        # ---------- Fusion ----------
        final_pred = fuse_predictions(
            image_pred=img_pred,
            symptom_pred=sym_full
        )

        top_idx = int(np.argmax(final_pred))

        return {
            "predicted_class": IMAGE_ID2LABEL[str(top_idx)],
            "confidence": float(final_pred[top_idx]),
            "all_scores": {
                IMAGE_ID2LABEL[str(i)]: float(final_pred[i])
                for i in range(len(IMAGE_CLASSES))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
