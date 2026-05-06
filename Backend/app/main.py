from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
from datetime import datetime

from app.vlm.router import router as vlm_router
from app.api.endpoints.bounding_boxes import router as bounding_boxes_router


app = FastAPI(title="Disaster Assessment Chatbot API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://main.d2oxlq5059eehn.amplifyapp.com"
    ],  # no trailing slash
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vlm_router)
app.include_router(bounding_boxes_router, prefix="/api/bounding-boxes", tags=["bounding-boxes"])

@app.get("/greet/{name}")
def greet_user(name: str):
    greetings = ["Hello", "Hi", "Hey", "Greetings", "Howdy", "Salutations"]
    return {"message": f"{random.choice(greetings)}, {name}!"}

@app.get("/time")
def get_time():
    now = datetime.now()
    return {"current_time": now.strftime("%I:%M:%S %p")}

@app.get("/health")
def health():
    return {"status": "ok"}