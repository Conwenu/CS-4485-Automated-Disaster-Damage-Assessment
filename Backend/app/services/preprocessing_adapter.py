import os

from app.services.preprocessing import get_pairs


DEFAULT_IMAGE_ROOT = os.path.join("Backend", "data", "santa_rosa", "images")


def normalize_pair(raw_pair):
    return {
        "id": raw_pair.get("id"),
        "city": raw_pair.get("city"),
        "pre_image_path": raw_pair.get("pre"),
        "post_image_path": raw_pair.get("post"),
        "labels_xy": raw_pair.get("labels", []),
    }


def validate_pair(pair):
    issues = []

    if not pair.get("id"):
        issues.append("missing id")
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

    labels = pair.get("labels_xy", [])
    if not isinstance(labels, list):
        issues.append("labels_xy must be a list")

    return len(issues) == 0, issues


def load_and_normalize_pairs(image_root=DEFAULT_IMAGE_ROOT):
    raw_pairs = get_pairs(os.path.abspath(image_root))
    normalized = []

    for pair in raw_pairs:
        normalized.append(normalize_pair(pair))

    return normalized


def load_normalized_pairs_with_issues(image_root=DEFAULT_IMAGE_ROOT):
    normalized = load_and_normalize_pairs(image_root=image_root)
    valid = []
    invalid = []

    for pair in normalized:
        is_valid, issues = validate_pair(pair)
        if is_valid:
            valid.append(pair)
        else:
            invalid.append({"pair": pair, "issues": issues})

    return valid, invalid
