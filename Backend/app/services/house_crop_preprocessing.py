import json
from pathlib import Path

from PIL import Image

CROP_PADDING = 5


def resolve_label_path(image_path: str, label_directory: str | None = None):
    img_path = Path(image_path)
    label_name = img_path.name.replace(".png", ".json")
    candidate_paths = []

    if label_directory:
        candidate_paths.append(Path(label_directory) / label_name)

    candidate_paths.append(img_path.parent.parent / "labels" / label_name)
    candidate_paths.append(img_path.parents[2] / "labels" / label_name)
    candidate_paths.append(img_path.parents[2] / "labelsCapstone" / label_name)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Label file not found for image: {image_path}")


def get_label_data(image_path: str, label_directory: str | None = None):
    label_path = resolve_label_path(image_path, label_directory=label_directory)

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", {}).get("xy", [])
    extracted = []

    for obj in features:
        wkt = obj.get("wkt", ",")
        coords = wkt.replace("POLYGON ((", "").replace("))", "")
        subtype = obj.get("properties", {}).get("subtype", "unknown")
        extracted.append({
            "coords": coords,
            "subtype": subtype,
        })

    return extracted


def create_bounding_box(coords: str):
    points = coords.split(",")
    xs = []
    ys = []

    for point in points:
        parts = point.strip().split(" ")
        if len(parts) != 2:
            continue
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))

    if not xs or not ys:
        return None

    min_x = min(xs) - CROP_PADDING
    min_y = min(ys) - CROP_PADDING
    max_x = max(xs) + CROP_PADDING
    max_y = max(ys) + CROP_PADDING
    return (int(min_x), int(min_y), int(max_x), int(max_y))


def get_building_pairs(
    image_directory: str,
    output_crop_directory: str,
    scene_id: str | None = None,
    label_directory: str | None = None,
):
    valid_pairs = []
    grouped = {}
    path = Path(image_directory)
    crop_dir = Path(output_crop_directory)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for img_path in path.glob("*.png"):
        name = img_path.name
        if "wildfire" not in name:
            continue

        parts = name.split("_")
        city_and_disaster = parts[0]
        num = parts[1]
        time = parts[2]
        city = city_and_disaster.replace("-wildfire", "")
        pair_id = f"{city}-{num}"

        if scene_id and pair_id != scene_id:
            continue

        if pair_id not in grouped:
            grouped[pair_id] = {"city": city}

        if "pre" in time:
            grouped[pair_id]["pre"] = str(img_path)
        elif "post" in time:
            grouped[pair_id]["post"] = str(img_path)
            grouped[pair_id]["labels_data"] = get_label_data(
                str(img_path),
                label_directory=label_directory,
            )

    for pair_id, data in grouped.items():
        if "pre" not in data or "post" not in data or "labels_data" not in data:
            continue

        with Image.open(data["pre"]) as pre_image, Image.open(data["post"]) as post_image:
            for idx, item in enumerate(data["labels_data"]):
                bounding = create_bounding_box(item["coords"])
                if bounding is None:
                    continue

                pre_crop = pre_image.crop(bounding)
                post_crop = post_image.crop(bounding)

                pre_crop_path = crop_dir / f"{pair_id}_bldg{idx}_pre.png"
                post_crop_path = crop_dir / f"{pair_id}_bldg{idx}_post.png"

                pre_crop.save(pre_crop_path)
                post_crop.save(post_crop_path)

                valid_pairs.append({
                    "building_id": f"{pair_id}_bldg{idx}",
                    "scene_id": pair_id,
                    "city": data["city"],
                    "subtype": item["subtype"],
                    "pre_crop": str(pre_crop_path),
                    "post_crop": str(post_crop_path),
                })

    return valid_pairs
