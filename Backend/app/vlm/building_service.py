import base64
import json
import os
import random
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")


def image_to_data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png"
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def extract_json_text(response):
    text = (response.output_text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def call_building_vlm(pre_image_path, post_image_path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Compare these cropped pre-disaster and post-disaster building images. "
                            "Focus only on the building itself. Ignore surrounding trees, grass, roads, dirt, smoke, and burned land unless the structure itself is damaged. "
                            "If the building structure looks intact, return no-damage even if the surroundings changed. "
                            "Return JSON only with keys: damage_level, confidence, reasoning. "
                            "damage_level must be one of: no-damage, minor-damage, major-damage, destroyed."
                        ),
                    },
                    {"type": "input_image", "image_url": image_to_data_url(pre_image_path)},
                    {"type": "input_image", "image_url": image_to_data_url(post_image_path)},
                ],
            }
        ],
        temperature=0,
    )

    raw_text = extract_json_text(response)
    result = json.loads(raw_text)

    if "damage_level" not in result:
        result["damage_level"] = "no-damage"
    if "confidence" not in result:
        result["confidence"] = None
    if "reasoning" not in result:
        result["reasoning"] = ""

    return result


def assess_building_damage(pre_image_path, post_image_path, ground_truth=None):
    model_result = call_building_vlm(pre_image_path, post_image_path)
    prediction = model_result.get("damage_level")

    return {
        "model": model_result,
        "evaluation": {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "match": prediction == ground_truth if ground_truth is not None else None,
        },
        "inputs": {
            "pre_image": os.path.basename(pre_image_path),
            "post_image": os.path.basename(post_image_path),
        },
    }


def assess_building_damage_with_retry(pre_image_path, post_image_path, ground_truth=None, max_attempts=4, base_delay_seconds=0.5):
    last_error = None

    for attempt in range(max_attempts):
        try:
            return assess_building_damage(pre_image_path, post_image_path, ground_truth=ground_truth)
        except Exception as error:
            last_error = error
            if attempt == max_attempts - 1:
                break

            delay = base_delay_seconds * (2 ** attempt)
            delay += random.uniform(0.0, 0.2)
            print(f"Retry {attempt + 1}/{max_attempts - 1} after error: {error}")
            time.sleep(delay)

    raise RuntimeError(f"Building assessment failed after {max_attempts} attempts: {last_error}")
