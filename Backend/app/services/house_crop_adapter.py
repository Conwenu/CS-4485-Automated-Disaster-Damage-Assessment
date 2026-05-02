import os

from app.services.house_crop_preprocessing import get_building_pairs


DEFAULT_IMAGE_ROOT = os.path.join("data", "santa_rosa", "images")
DEFAULT_CROP_ROOT = os.path.join("data", "santa_rosa", "building_crops")


def normalize_building_pair(item):
    return {
        "id": item.get("building_id"),
        "scene_id": item.get("scene_id"),
        "city": item.get("city"),
        "pre_image_path": item.get("pre_crop"),
        "post_image_path": item.get("post_crop"),
        "ground_truth": item.get("subtype"),
    }


def validate_building_pair(pair):
    issues = []

    if not pair.get("id"):
        issues.append("missing id")
    if not pair.get("scene_id"):
        issues.append("missing scene_id")
    if not pair.get("pre_image_path"):
        issues.append("missing pre_image_path")
    if not pair.get("post_image_path"):
        issues.append("missing post_image_path")

    pre_path = pair.get("pre_image_path")
    post_path = pair.get("post_image_path")

    if isinstance(pre_path, str) and not os.path.exists(pre_path):
        issues.append(f"pre image not found: {pre_path}")
    if isinstance(post_path, str) and not os.path.exists(post_path):
        issues.append(f"post image not found: {post_path}")

    return len(issues) == 0, issues


def load_and_normalize_building_pairs(
    image_root=DEFAULT_IMAGE_ROOT,
    crop_root=DEFAULT_CROP_ROOT,
    scene_id=None,
    label_root=None,
):
    raw_pairs = get_building_pairs(
        image_root,
        crop_root,
        scene_id=scene_id,
        label_directory=label_root,
    )
    normalized = []

    for item in raw_pairs:
        normalized.append(normalize_building_pair(item))

    return normalized


def load_normalized_building_pairs_with_issues(
    image_root=DEFAULT_IMAGE_ROOT,
    crop_root=DEFAULT_CROP_ROOT,
    scene_id=None,
    label_root=None,
):
    normalized = load_and_normalize_building_pairs(
        image_root=image_root,
        crop_root=crop_root,
        scene_id=scene_id,
        label_root=label_root,
    )
    valid = []
    invalid = []

    for pair in normalized:
        is_valid, issues = validate_building_pair(pair)
        if is_valid:
            valid.append(pair)
        else:
            invalid.append({"pair": pair, "issues": issues})

    return valid, invalid
