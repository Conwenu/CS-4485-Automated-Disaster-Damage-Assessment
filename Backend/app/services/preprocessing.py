from pathlib import Path

from PIL import Image

from app.services.house_crop_preprocessing import (
    create_bounding_box,
    get_label_data,
)


def get_pairs(
    image_directory: str,
    output_crop_directory: str | None = None,
    scene_id: str | None = None,
    label_directory: str | None = None,
):
    valid_pairs = []
    grouped = {}
    image_path = Path(image_directory)
    crop_dir = Path(output_crop_directory) if output_crop_directory else None

    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_path.glob("*.png"):
        name = img_path.name
        if "wildfire" not in name:
            continue

        parts = name.split("_")
        city = parts[0].replace("-wildfire", "")
        num = parts[1]
        time = parts[2]
        pair_id = f"{city}-{num}"

        if scene_id and pair_id != scene_id:
            continue

        grouped.setdefault(pair_id, {"city": city})

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

                pair = {
                    "building_id": f"{pair_id}_bldg{idx}",
                    "scene_id": pair_id,
                    "city": data["city"],
                    "subtype": item["subtype"],
                    "bbox": list(bounding),
                }

                if crop_dir is not None:
                    pre_crop_path = crop_dir / f"{pair_id}_bldg{idx}_pre.png"
                    post_crop_path = crop_dir / f"{pair_id}_bldg{idx}_post.png"
                    pre_image.crop(bounding).save(pre_crop_path)
                    post_image.crop(bounding).save(post_crop_path)
                    pair["pre_crop"] = str(pre_crop_path)
                    pair["post_crop"] = str(post_crop_path)

                valid_pairs.append(pair)

    return valid_pairs
