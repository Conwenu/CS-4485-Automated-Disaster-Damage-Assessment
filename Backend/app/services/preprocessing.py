import json
import os
from pathlib import Path

def get_label_xy(imagePath: str):
    imgPath = Path(imagePath)
    labelPath = imgPath.parent.parent / "labels" / imgPath.name.replace(".png", ".json")
    with open(labelPath, 'r') as p:
        data = json.load(p)
        features = data.get("features", {}).get("xy", [])
        rawCoords = []
        for obj in features:
            wkt = obj.get("wkt", ",")
            clean = wkt.replace("POLYGON ((", "").replace("))", "")
            rawCoords.append(clean)
        return rawCoords

def get_pairs(image_directory: str):
    groupIds = {}
    path = Path(image_directory)
    for imgPath in path.glob("*.png"):
        name = imgPath.name
        if "wildfire" not in name:
            continue
        parts = name.split("_")
        cityAndDisaster = parts[0]
        num = parts[1]
        time = parts[2]
        city = cityAndDisaster.replace("-wildfire", "")
        pair = f"{city}-{num}"
        if pair not in groupIds:
            groupIds[pair] = {'city': city}
        if "pre" in time:
            groupIds[pair]['pre'] = str(imgPath)
            groupIds[pair]['labels'] = get_label_xy(imgPath)
        elif "post" in time:
            groupIds[pair]["post"] = str(imgPath)
            if 'labels' not in groupIds[pair] or not groupIds[pair]['labels']:
                groupIds[pair]['labels'] = get_label_xy(imgPath)
    
    validPairs = []
    for pairs, data in groupIds.items():
        if "pre" in data and "post" in data:
            validPairs.append({
                "id": pairs,
                "city": data["city"],
                "pre": data["pre"],
                "post": data["post"],
                "labels": data.get("labels", [])
            })
    return validPairs


if __name__ == "__main__":
    script_dir = Path(__file__).parent 
    test_image_dir = script_dir.parent / "data" / "images" / "test" / "images"
    
    try:
        results = get_pairs(str(test_image_dir))
        print(f"Found {len(results)} valid pairs.\n")
        
        for sample in results:
            print(f"ID: {sample['id']} (City: {sample['city']})")
            
            print(f"\n[LABELS] Buildings found: {len(sample['labels'])}")
            for i, lbl in enumerate(sample['labels'], 1):
                print(f"  Label {i}: {lbl}")
            
            print("\n" + "-"*50 + "\n")
            break
            
    except Exception as e:
        print(f"Error: {e}")