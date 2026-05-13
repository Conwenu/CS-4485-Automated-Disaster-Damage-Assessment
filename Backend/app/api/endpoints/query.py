"""Chat API routes.

Two endpoints: synchronous /ask and streaming /ask/stream.
Both route through ChatService — see app/services/chat_service.py.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.chat_service import ChatService
from app.models.models import QueryRequest

router = APIRouter()
chat_service = ChatService()


@router.post("/ask")
def ask(request: QueryRequest):
    """Synchronous chat endpoint. Returns a complete response as JSON."""
    try:
        return chat_service.process_query(
            query=request.query,
            session_id=request.session_id,
            history=(
                [m.model_dump() for m in request.history] if request.history else None
            ),
            pending_clarification=request.pending_clarification,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """Streaming chat endpoint. Returns Server-Sent Events.

    Event types in order:
      - token:    text chunks of the response
      - metadata: suggestions, ui_actions, and clarification state
      - done:     signals stream completion
    """
    try:
        return StreamingResponse(
            chat_service.process_query_stream(
                query=request.query,
                session_id=request.session_id,
                history=(
                    [m.model_dump() for m in request.history]
                    if request.history
                    else None
                ),
                pending_clarification=request.pending_clarification,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
