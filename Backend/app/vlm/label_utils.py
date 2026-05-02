"""Ground-truth and evaluation."""

import json


def load_label(label_path: str) -> dict:
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_scene_damage(label_json: dict) -> str:
    features = label_json.get("features", {}).get("lng_lat", [])

    for feature in features:
        subtype = feature.get("properties", {}).get("subtype")
        if subtype == "destroyed":
            return "destroyed"

    for feature in features:
        subtype = feature.get("properties", {}).get("subtype")
        if subtype == "major-damage":
            return "major-damage"

    for feature in features:
        subtype = feature.get("properties", {}).get("subtype")
        if subtype == "minor-damage":
            return "minor-damage"

    return "no-damage"


def compare_with_ground_truth(model_result: dict, label_json: dict) -> dict:
    model_prediction = model_result.get("damage_level")
    ground_truth = label_to_scene_damage(label_json)

    return {
        "model_prediction": model_prediction,
        "ground_truth": ground_truth,
        "match": model_prediction == ground_truth,
    }
