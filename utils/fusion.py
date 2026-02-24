import numpy as np


def fuse_predictions(
    image_pred: np.ndarray,
    symptom_pred: np.ndarray
) -> np.ndarray:

    IMAGE_WEIGHT = 0.6
    SYMPTOM_WEIGHT = 0.4

    image_pred = np.asarray(image_pred, dtype="float32").flatten()
    symptom_pred = np.asarray(symptom_pred, dtype="float32").flatten()

    # ตรวจจำนวนคลาส
    if image_pred.shape[0] != 11:
        raise ValueError(
            f"Image model must have 11 classes, got {image_pred.shape}"
        )

    # align symptom -> เพิ่ม normal_skin
    if symptom_pred.shape[0] == 10:
        symptom_pred = np.insert(symptom_pred, 6, 0.0)

    # normalize
    image_pred = image_pred / (np.sum(image_pred) + 1e-8)

    if np.sum(symptom_pred) > 0:
        symptom_pred = symptom_pred / (np.sum(symptom_pred) + 1e-8)

    # ----------------------------------------
    # วิเคราะห์ confidence
    # ----------------------------------------
    image_top = np.argmax(image_pred)
    image_conf = image_pred[image_top]

    symptom_top = np.argmax(symptom_pred)
    symptom_conf = symptom_pred[symptom_top]

    NORMAL_SKIN_INDEX = 6

    # ----------------------------------------
    # ถ้าไม่มี symptom → ใช้ image อย่างเดียว
    # แต่ต้องมั่นใจสูงจริง
    # ----------------------------------------
    if np.sum(symptom_pred) == 0:

        # กันรูปมั่ว
        if image_conf < 0.75:
            return np.zeros_like(image_pred)

        return image_pred

    # ----------------------------------------
    # ผิวปกติ
    # ----------------------------------------
    if image_top == NORMAL_SKIN_INDEX and image_conf > 0.7:
        return image_pred

    # ----------------------------------------
    # ถ้า model ไม่เห็นตรงกัน
    # ----------------------------------------
    if image_top != symptom_top:
        if image_conf < 0.5 and symptom_conf < 0.5:
            return np.zeros_like(image_pred)

    # ----------------------------------------
    # Adaptive fusion
    # ----------------------------------------
    if symptom_conf > image_conf:
        IMAGE_WEIGHT = 0.4
        SYMPTOM_WEIGHT = 0.6

    if image_conf > 0.8:
        IMAGE_WEIGHT = 0.75
        SYMPTOM_WEIGHT = 0.25

    fused = (
        IMAGE_WEIGHT * image_pred +
        SYMPTOM_WEIGHT * symptom_pred
    )

    fused = fused / (np.sum(fused) + 1e-8)

    # ----------------------------------------
    # Final guard
    # ----------------------------------------
    if np.max(fused) < 0.45:
        return np.zeros_like(fused)

    return fused