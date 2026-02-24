import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_image(
    image_bytes: bytes,
    target_size=(224, 224)
) -> np.ndarray:

    if image_bytes is None:
        raise ValueError("image_bytes is None")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    image = image.resize(target_size)

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    return image_array