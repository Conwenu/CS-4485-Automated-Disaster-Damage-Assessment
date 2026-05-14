import json
import re
from pathlib import Path

BASE = Path(__file__).parent
LABELS_DIR = BASE / "data" / "santa_rosa" / "labelsCapstone"
RESULTS_FILE = BASE / "data" / "santa_rosa" / "building_results.json"
OUTPUT_FILE = BASE.parent / "Frontend" / "public" / "bounding_boxes.json"

print("loading building_results.json ...")
with open(RESULTS_FILE, encoding="utf-8") as f:
    results = json.load(f)


scene_damage: dict[str, list[str]] = {}
for entry in results:
    scene_id = entry.get("scene_id", "")
    damage = entry.get("evaluation", {}).get("ground_truth","un-classified")
    scene_damage.setdefault(scene_id, []).append(damage)


def wkt_to_bbox(wkt: str) -> list[float] | None:
    """
    Extract all (x y) pairs from a WKT POLYGON string and return
    [minX, minY, maxX, maxY] in pixel space.
    """
    nums = re.findall(r"[-\d.]+\s+[-\d.]+", wkt)
    if not nums:
        return None
    xs, ys = [], []
    for pair in nums:
        parts = pair.split()
        if len(parts) == 2:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]

print(f"Processing label files in {LABELS_DIR} ...")
all_boxes: list[dict] = []
skipped = 0
 
label_files = sorted(LABELS_DIR.glob("*_pre_disaster.json"))
print(f"Found {len(label_files)} pre-disaster label files.")

for label_path in label_files:
    m = re.search(r"santa-rosa-wildfire_(\d+)_pre_disaster", label_path.name)
    if not m:
        skipped += 1
        continue
 
    tile_num  = m.group(1)
    image_id  = f"santa-rosa-{tile_num}"
    scene_id  = f"santa-rosa-{tile_num}"
 
    try:
        with open(label_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  SKIP {label_path.name}: {e}")
        skipped += 1
        continue

    features = data.get("features", {})
    xy_features = features.get("xy", [])
 
    if not xy_features:
        skipped += 1
        continue

    damage_list = scene_damage.get(scene_id, [])
 
    for i, feature in enumerate(xy_features):
        props = feature.get("properties", {})
        uid   = props.get("uid", f"{image_id}_bldg{i}")
        wkt   = feature.get("wkt", "")
 
        bbox = wkt_to_bbox(wkt)
        if bbox is None:
            continue

        damage = damage_list[i] if i < len(damage_list) else "un-classified"
 
        all_boxes.append({
            "building_id": f"{image_id}_{uid}",
            "subtype": damage,
            "bbox": bbox,
        })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_boxes, f)
 
print(f"\nDone!")
print(f"  Total bounding boxes: {len(all_boxes)}")
print(f"  Skipped files:        {skipped}")
print(f"  Output written to:    {OUTPUT_FILE}")