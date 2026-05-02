import os
import shutil
import uuid

from fastapi import APIRouter, File, UploadFile

from app.vlm.service import assess_damage

router = APIRouter(prefix="/vlm", tags=["VLM"])
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.get("/health")
async def health_check():
    return {"status": "ok", "module": "vlm"}


@router.post("/assess")
async def assess(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    label_file: UploadFile = File(...),
):
    run_id = uuid.uuid4().hex

    pre_path = os.path.join(TEMP_DIR, f"{run_id}_{pre_image.filename}")
    post_path = os.path.join(TEMP_DIR, f"{run_id}_{post_image.filename}")
    label_path = os.path.join(TEMP_DIR, f"{run_id}_{label_file.filename}")

    try:
        with open(pre_path, "wb") as f:
            shutil.copyfileobj(pre_image.file, f)
        with open(post_path, "wb") as f:
            shutil.copyfileobj(post_image.file, f)
        with open(label_path, "wb") as f:
            shutil.copyfileobj(label_file.file, f)

        return assess_damage(pre_path, post_path, label_path)
    finally:
        for path in (pre_path, post_path, label_path):
            if os.path.exists(path):
                os.remove(path)
