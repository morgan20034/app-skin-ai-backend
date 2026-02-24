from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
import os

from utils.image_preprocess import preprocess_image
from utils.symptom_encoder import encode_symptoms
from utils.fusion import fuse_predictions


app = FastAPI(title="Skin AI API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# PATH (แก้ให้ใช้ models อย่างเดียว)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SCHEMA_PATH = os.path.join(MODEL_DIR, "symptom_schema.json")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
DISEASES_PATH = os.path.join(MODEL_DIR, "diseases.json")

IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
SYMPTOM_MODEL_PATH = os.path.join(MODEL_DIR, "checkbox_model_10class.h5")

# -----------------------------
# GLOBAL VARIABLES
# -----------------------------
image_model = None
symptom_model = None
image_class_names = []
diseases_data = {}


# -----------------------------
# LOAD MODELS (สำคัญสำหรับ Railway)
# -----------------------------
@app.on_event("startup")
def load_models():
    global image_model, symptom_model, image_class_names, diseases_data

    print("🚀 Loading models... (Railway)")

    # ตรวจสอบไฟล์ก่อนโหลด
    required_files = [
        IMAGE_MODEL_PATH,
        SYMPTOM_MODEL_PATH,
        CLASSES_PATH,
        DISEASES_PATH,
        SCHEMA_PATH
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise RuntimeError(f"❌ Missing file: {file}")

    # โหลดโมเดล
    image_model = tf.keras.models.load_model(
        IMAGE_MODEL_PATH,
        compile=False
    )

    symptom_model = tf.keras.models.load_model(
        SYMPTOM_MODEL_PATH,
        compile=False
    )

    # โหลด classes
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_config = json.load(f)

    image_class_names = classes_config["image_model"]["class_names"]

    # โหลด diseases
    with open(DISEASES_PATH, "r", encoding="utf-8") as f:
        diseases_data.update(json.load(f))

    print("✅ Models loaded successfully")


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"message": "Skin AI API is running 🚀"}


# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    symptoms: str = Form(None),
    areas: str = Form(None)
):
    try:
        print("📥 Request received")

        # อ่านรูป
        image_bytes = await image.read()
        image_tensor = preprocess_image(image_bytes)

        image_probs = image_model.predict(image_tensor, verbose=0)
        image_probs = np.squeeze(image_probs)

        # parse symptoms
        selected_symptoms = []
        selected_areas = []

        if symptoms:
            try:
                selected_symptoms = json.loads(symptoms)
            except:
                selected_symptoms = symptoms.split(",")

        if areas:
            try:
                selected_areas = json.loads(areas)
            except:
                selected_areas = areas.split(",")

        print("Symptoms:", selected_symptoms)

        # RULE 1: ไม่เลือก symptom
        if len(selected_symptoms) == 0:
            return {
                "status": "unknown",
                "disease": diseases_data["unknown"]
            }

        # RULE 2: เลือกมั่วเยอะเกิน
        if len(selected_symptoms) >= 8:
            return {
                "status": "unknown",
                "disease": diseases_data["unknown"]
            }

        # encode symptom
        symptom_tensor = encode_symptoms(
            selected_symptoms,
            selected_areas,
            SCHEMA_PATH
        )

        symptom_probs = symptom_model.predict(symptom_tensor, verbose=0)
        symptom_probs = np.squeeze(symptom_probs)

        # fusion
        fused_probs = fuse_predictions(
            image_pred=image_probs,
            symptom_pred=symptom_probs
        )

        if fused_probs is None or np.sum(fused_probs) == 0:
            return {
                "status": "unknown",
                "disease": diseases_data["unknown"]
            }

        top_index = int(np.argmax(fused_probs))
        top_score = float(fused_probs[top_index])
        predicted_class = image_class_names[top_index]

        print("Final:", predicted_class, top_score)

        # confidence ต่ำ
        if top_score < 0.50:
            return {
                "status": "unknown",
                "disease": diseases_data["unknown"]
            }

        # normal skin
        if predicted_class == "normal_skin":
            return {
                "status": "normal",
                "disease": diseases_data["normal_skin"],
                "confidence": top_score
            }

        return {
            "status": "success",
            "disease": diseases_data.get(predicted_class, diseases_data["unknown"]),
            "confidence": top_score
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# RUN LOCAL / RAILWAY
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
