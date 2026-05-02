from fastapi import FastAPI, Request

from app.vlm.router import router as vlm_router
from app.api.endpoints.chat import router as ChatRouter
from app.api.endpoints.query import router as QueryRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.include_router(vlm_router)
app.include_router(ChatRouter, prefix="/chat", tags=["Chat"])
app.include_router(QueryRouter, prefix="/query", tags=["Query"])


templates = Jinja2Templates(directory="app/templates")

@app.get("/chat_interface", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the chat web interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}
