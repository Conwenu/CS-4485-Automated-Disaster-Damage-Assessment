from fastapi import APIRouter, Query
from app.services.preprocessing import get_pairs
from pathlib import Path

router = APIRouter()

@router.get("/")
def bounding_boxes(image_id: str = Query(default="santa-rosa-00000000")):
    base = Path(__file__).parent.parent.parent
    image_dir = base / "data" / "images" / "test" / "images"
    crop_dir = base / "data" / "images" / "test" / "building_crops"

    results = get_pairs(str(image_dir), str(crop_dir))
    filtered = [r for r in results if r["building_id"].startswith(image_id)]
    return [
        {
            "building_id": r["building_id"],
            "subtype": r["subtype"],
            "bbox": r["bbox"],
        }
        for r in filtered
    ]