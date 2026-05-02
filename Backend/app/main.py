from fastapi import FastAPI

from app.vlm.router import router as vlm_router

app = FastAPI()
app.include_router(vlm_router)


@app.get("/health")
def health():
    return {"status": "ok"}
