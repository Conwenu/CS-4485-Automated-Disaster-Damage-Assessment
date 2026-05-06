import random
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints.bounding_boxes import router as bounding_boxes_router
from app.vlm.router import router as vlm_router

chat = None
query = None

try:
    from app.api.endpoints import chat, query
except ModuleNotFoundError:
    # Keep the VLM/backend app bootable even when optional chatbot deps
    # are not installed on a presentation machine.
    pass


app = FastAPI(title="Disaster Assessment Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vlm_router)
if chat is not None:
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
if query is not None:
    app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(
    bounding_boxes_router,
    prefix="/api/bounding-boxes",
    tags=["bounding-boxes"],
)


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