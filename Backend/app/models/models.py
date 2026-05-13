from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    relevant_data: List[Any]
    map_focus: Optional[dict] = None
    suggested_followups: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: str  # should be 'user' or 'bot'
    content: str


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    # dashboard_context: Optional[Any]
    history: Optional[List[ChatMessage]] = None
    pending_clarification: Optional[Dict[str, Any]] = None  # new


class TestResultResponse(BaseModel):
    intent_accuracy: str
    parameter_accuracy: str
    total_time_taken: str
    results: list
