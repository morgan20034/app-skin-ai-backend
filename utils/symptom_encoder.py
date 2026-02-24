import json
import numpy as np
from typing import List, Dict


def load_symptom_schema(schema_path: str) -> Dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_to_key_map(schema: Dict) -> Dict[str, str]:
    """
    map label_th -> key
    """
    mapping = {}
    for s in schema.get("symptoms", []):
        mapping[s["label_th"]] = s["key"]
    for a in schema.get("areas", []):
        mapping[a["label_th"]] = a["key"]
    return mapping


def encode_symptoms(
    selected_symptoms_th: List[str],
    selected_areas_th: List[str],
    schema_path: str
) -> np.ndarray:

    schema = load_symptom_schema(schema_path)
    label_map = build_label_to_key_map(schema)

    selected_symptoms = [label_map[s] for s in selected_symptoms_th if s in label_map]
    selected_areas = [label_map[a] for a in selected_areas_th if a in label_map]

    features = []

    # symptoms
    symptom_keys = [s["key"] for s in schema["symptoms"]]
    for key in symptom_keys:
        features.append(1.0 if key in selected_symptoms else 0.0)

    # areas
    area_keys = [a["key"] for a in schema["areas"]]
    for key in area_keys:
        features.append(1.0 if key in selected_areas else 0.0)

    features = np.array(features, dtype="float32")

    if features.shape[0] != schema["num_features"]:
        raise ValueError(
            f"Feature size mismatch: {features.shape[0]} != {schema['num_features']}"
        )

    return np.expand_dims(features, axis=0)
