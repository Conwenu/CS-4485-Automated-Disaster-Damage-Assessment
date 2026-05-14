"""Chat service orchestrator.

Stateless: the client sends history and any pending_clarification each turn.
One LLM call per turn (the parser). Everything else is deterministic Python.

Per-turn flow:
  1. Format history for the parser.
  2. Parser call: single pass returning a ParsedQuery.
  3. If there was a pending_clarification:
        a) If the parser says is_answer_to_pending=true, merge it.
        b) Otherwise the user pivoted — drop the pending state.
  4. Dispatch to the intent engine.
  5. If the engine asks for more info, return a clarification envelope.
  6. Otherwise build the final response.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator

from app.services.query_parser import QueryParser
from app.services.query_engine import QueryEngine
from app.services.intent_dispatcher import IntentDispatcher
from app.services.response_service import ResponseService
from app.services.external_knowledge_service import ExternalKnowledgeService
from app.models.schemas import ParsedQuery
from app.models.enums import ChatIntent
from app.data.data import retrieve_true_data
from app.rag.chains import stream_polished_answer

from app.config import settings

log = logging.getLogger(__name__)


OOS_REDIRECT_TEXT = (
    "I can only answer questions about the disaster damage dataset — "
    "damage levels, model predictions, accuracy, cities or buildings, "
    "and related disaster concepts. Try asking about a city, a damage "
    "level, or the model's performance."
)

OOS_SUGGESTIONS = [
    "Show the overall damage distribution",
    "How accurate is the model?",
    "Show me 5 destroyed buildings",
]


class ChatService:
    def __init__(self):
        self.qp = QueryParser()
        self.qe = QueryEngine(retrieve_true_data())
        self.dispatcher = IntentDispatcher(self.qe)
        self.responder = ResponseService(query_engine=self.qe)
        self.external = ExternalKnowledgeService()

    # ------------------------------------------------------------------
    # Non-streaming path (backward compatible)
    # ------------------------------------------------------------------
    def process_query(
        self,
        query: str,
        session_id: str,
        history: Optional[List[Dict[str, str]]] = None,
        pending_clarification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        formatted_history = self._format_history(history or [])

        parsed = self.qp.parse(
            query=query,
            history=formatted_history,
            pending_clarification=pending_clarification,
        )
        log.info(
            "parsed session=%s intent=%s follow_up=%s answer_to_pending=%s reasoning=%r",
            session_id,
            parsed.intent.value,
            parsed.is_follow_up,
            parsed.is_answer_to_pending,
            parsed.reasoning,
        )

        if pending_clarification:
            if parsed.is_answer_to_pending:
                parsed = self._merge_pending(pending_clarification, parsed, query)
            else:
                log.info(
                    "User pivoted away from pending clarification; dropping it. session=%s",
                    session_id,
                )

        final_query = parsed.rewritten_query or query

        dispatch_result = self.dispatcher.dispatch(parsed.model_dump(mode="python"))
        data = dispatch_result.get("data", {}) or {}

        # Engine-raised clarification path.
        if data.get("type") == "clarification_needed":
            message = data.get("message", "Could you clarify?")
            pending_out = data.get("pending_clarification")
            return self._build_response(
                original_query=query,
                final_query=final_query,
                parsed=parsed,
                response={"text": message, "suggestions": [], "ui_actions": []},
                requires_clarification=True,
                pending_clarification=pending_out,
            )

        # External knowledge (RAG).
        external_info = None
        external_note = None
        if parsed.needs_external_knowledge and parsed.external_query:
            enriched = parsed.external_query
            log.info("Calling external retrieval with query=%r", enriched)
            external_info = self.external.retrieve(enriched)
            log.info(
                "External retrieval returned: %s",
                repr(external_info)[:150] if external_info else "None",
            )
            if external_info is None:
                external_note = (
                    "I may not have information on that in my reference materials "
                    "(xBD documentation, FEMA guidelines, and Tubbs Fire reports). "
                )

        # OOS + external knowledge: return JUST the external answer.
        if parsed.intent.value == "OUT_OF_SCOPE" and external_info is not None:
            return self._build_response(
                original_query=query,
                final_query=final_query,
                parsed=parsed,
                response={
                    "text": external_info,
                    "suggestions": OOS_SUGGESTIONS,
                    "ui_actions": [],
                },
                requires_clarification=False,
                external_info=external_info,
            )

        # OOS with no external info — show redirect.
        if parsed.intent.value == "OUT_OF_SCOPE":
            return self._build_response(
                original_query=query,
                final_query=final_query,
                parsed=parsed,
                response={
                    "text": OOS_REDIRECT_TEXT,
                    "suggestions": OOS_SUGGESTIONS,
                    "ui_actions": [],
                },
                requires_clarification=False,
            )

        # Dataset response.
        response = self.responder.generate(
            query=final_query,
            parsed=parsed,
            result=dispatch_result,
            external_info=external_info,
            external_note=external_note,
        )

        return self._build_response(
            original_query=query,
            final_query=final_query,
            parsed=parsed,
            response=response,
            requires_clarification=False,
            external_info=external_info,
        )

    # ------------------------------------------------------------------
    # Streaming path (SSE)
    # ------------------------------------------------------------------
    async def process_query_stream(
        self,
        query: str,
        session_id: str,
        history: Optional[List[Dict[str, str]]] = None,
        pending_clarification: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        formatted_history = self._format_history(history or [])
        loop = asyncio.get_event_loop()

        # --- Parser ---
        parsed = await loop.run_in_executor(
            None,
            lambda: self.qp.parse(
                query=query,
                history=formatted_history,
                pending_clarification=pending_clarification,
            ),
        )
        log.info(
            "stream session=%s intent=%s reasoning=%r",
            session_id,
            parsed.intent.value,
            parsed.reasoning,
        )

        if pending_clarification and parsed.is_answer_to_pending:
            parsed = self._merge_pending(pending_clarification, parsed, query)

        final_query = parsed.rewritten_query or query

        # --- Dispatch ---
        dispatch_result = await loop.run_in_executor(
            None,
            lambda: self.dispatcher.dispatch(parsed.model_dump(mode="python")),
        )
        data = dispatch_result.get("data", {}) or {}

        # --- Engine-raised clarification ---
        if data.get("type") == "clarification_needed":
            yield self._sse("token", data.get("message", "Could you clarify?"))
            yield self._sse(
                "metadata",
                {
                    "suggestions": [],
                    "ui_actions": [],
                    "requires_clarification": True,
                    "pending_clarification": data.get("pending_clarification"),
                },
            )
            yield self._sse("done", {})
            return

        # --- OOS with no external knowledge — early return (no LLM polish) ---
        if (
            parsed.intent.value == "OUT_OF_SCOPE"
            and not parsed.needs_external_knowledge
        ):
            yield self._sse("token", OOS_REDIRECT_TEXT)
            yield self._sse(
                "metadata",
                {
                    "suggestions": OOS_SUGGESTIONS,
                    "ui_actions": [],
                    "requires_clarification": False,
                },
            )
            yield self._sse("done", {})
            return

        llm = self.external.get_generator_llm()

        # --------------------------------------------------------------
        # PURE EXTERNAL KNOWLEDGE (OOS + needs_external_knowledge)
        # --------------------------------------------------------------
        if (
            parsed.needs_external_knowledge
            and parsed.external_query
            and parsed.intent.value == "OUT_OF_SCOPE"
        ):
            enriched = parsed.external_query
            external_info = await loop.run_in_executor(
                None,
                self.external.retrieve,
                enriched,
            )

            if not external_info:
                yield self._sse("token", OOS_REDIRECT_TEXT)
                yield self._sse(
                    "metadata",
                    {
                        "suggestions": OOS_SUGGESTIONS,
                        "ui_actions": [],
                        "requires_clarification": False,
                    },
                )
                yield self._sse("done", {})
                return

            # Strip the "Sources: ..." line before polishing, then re-append.
            sources_line = ""
            base_for_polish = external_info
            if "\n\nSources:" in external_info:
                parts = external_info.rsplit("\n\nSources:", 1)
                base_for_polish = parts[0].strip()
                sources_line = "\n\nSources:" + parts[1]

            if settings.USE_LLM_RESPONSE_POLISH and llm:
                async for token in stream_polished_answer(
                    base_for_polish, final_query, llm
                ):
                    yield self._sse("token", token)
            else:
                for line in base_for_polish.split("\n"):
                    if line.strip():
                        yield self._sse("token", line + "\n")
                        await asyncio.sleep(0.03)

            if sources_line:
                yield self._sse("token", sources_line)

            yield self._sse(
                "metadata",
                {
                    "suggestions": OOS_SUGGESTIONS,
                    "ui_actions": [],
                    "requires_clarification": False,
                },
            )
            yield self._sse("done", {})
            return

        # --------------------------------------------------------------
        # DATASET RESPONSE (with optional mixed external knowledge)
        # --------------------------------------------------------------
        external_info = None
        if parsed.needs_external_knowledge and parsed.external_query:
            enriched = parsed.external_query
            external_info = await loop.run_in_executor(
                None,
                self.external.retrieve,
                enriched,
            )

        # Build the base text without polish — polish happens via streaming.
        response = self.responder.generate(
            query=final_query,
            parsed=parsed,
            result=dispatch_result,
            external_info=external_info,
            external_note=None,
            skip_polish=True,
        )

        base_text = response.get("text", "")

        if settings.USE_LLM_RESPONSE_POLISH and llm and base_text:
            async for token in stream_polished_answer(base_text, final_query, llm):
                yield self._sse("token", token)
        else:
            for line in base_text.split("\n"):
                if line.strip():
                    yield self._sse("token", line + "\n")
                    await asyncio.sleep(0.03)

        yield self._sse(
            "metadata",
            {
                "suggestions": response.get("suggestions", []),
                "ui_actions": response.get("ui_actions", []),
                "requires_clarification": False,
                "pending_clarification": None,
            },
        )
        yield self._sse("done", {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sse(event: str, data) -> str:
        """Format a Server-Sent Event."""
        if isinstance(data, str):
            payload = json.dumps({"text": data})
        else:
            payload = json.dumps(data)
        return f"event: {event}\ndata: {payload}\n\n"

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "(no prior turns)"
        return "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history
        )

    def _merge_pending(
        self,
        pending: Dict[str, Any],
        parsed: ParsedQuery,
        raw_answer: str,
    ) -> ParsedQuery:
        """Fold a pending clarification answer back into a complete ParsedQuery."""
        missing = pending.get("missing_param")
        params = pending.get("params") or {}
        intent_value = pending.get("intent")
        answer = (raw_answer or "").strip()

        merged: Dict[str, Any] = {
            "reasoning": f"Filled pending '{missing}' from user's answer.",
            "is_follow_up": True,
            "rewritten_query": parsed.rewritten_query,
            "is_answer_to_pending": True,
            "intent": ChatIntent(intent_value) if intent_value else parsed.intent,
            "city": params.get("city"),
            "cities": list(params.get("cities", []) or []),
            "id": params.get("id"),
            "ids": list(params.get("ids", []) or []),
            "scene_id": params.get("scene_id"),
            "top_k": params.get("top_k"),
            "damage_level": params.get("damage_level"),
            "confidence_threshold": params.get("confidence_threshold"),
            "direction": params.get("direction"),
            "status": params.get("status"),
            "needs_external_knowledge": False,
            "external_query": None,
            "needs_clarification": False,
            "clarification_question": None,
            "missing_param": None,
        }

        if missing == "top_k":
            merged["top_k"] = parsed.top_k or self._safe_int(answer)
        elif missing == "city":
            merged["city"] = parsed.city or answer
        elif missing == "cities":
            new_city = (
                (parsed.cities[0] if parsed.cities else None) or parsed.city or answer
            )
            if new_city and new_city not in merged["cities"]:
                merged["cities"].append(new_city)
        elif missing == "id":
            merged["id"] = parsed.id or answer
        elif missing == "ids":
            new_ids = parsed.ids or [p.strip() for p in answer.split(",") if p.strip()]
            merged["ids"] = list(dict.fromkeys(merged["ids"] + new_ids))
        elif missing == "scene_id":
            merged["scene_id"] = parsed.scene_id or answer
        elif missing == "damage_level":
            merged["damage_level"] = parsed.damage_level
        elif missing == "confidence_threshold":
            merged["confidence_threshold"] = (
                parsed.confidence_threshold
                if parsed.confidence_threshold is not None
                else self._safe_float(answer)
            )
        elif missing == "direction":
            merged["direction"] = parsed.direction or (
                "above" if "above" in answer.lower() else "below"
            )
        elif missing == "status":
            merged["status"] = parsed.status or (
                "failed"
                if "fail" in answer.lower()
                else ("ok" if "ok" in answer.lower() else None)
            )

        return ParsedQuery(**merged)

    @staticmethod
    def _safe_int(s: str) -> Optional[int]:
        try:
            return int((s or "").strip())
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _safe_float(s: str) -> Optional[float]:
        try:
            s = (s or "").strip()
            if s.endswith("%"):
                return float(s.rstrip("%")) / 100.0
            return float(s)
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _build_response(
        original_query: str,
        final_query: str,
        parsed: ParsedQuery,
        response: Dict[str, Any],
        requires_clarification: bool,
        pending_clarification: Optional[Dict[str, Any]] = None,
        external_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "original_query": original_query,
            "final_query": final_query,
            "is_follow_up": parsed.is_follow_up,
            "rewritten_query": parsed.rewritten_query,
            "parsed": parsed.model_dump(),
            "response": response,
            "requires_clarification": requires_clarification,
        }
        if pending_clarification is not None:
            payload["pending_clarification"] = pending_clarification
        if external_info is not None:
            payload["external_info"] = external_info
        return payload

    @staticmethod
    def _ensure_specific_query(query: str) -> str:
        """Safety net: prepend Tubbs Fire context to vague external queries.

        The parser SHOULD enrich queries per its prompt rules. This is a
        belt-and-suspenders backup. If you see the warning fire on real
        queries, that's data telling you the parser prompt needs work.
        """
        if not query:
            return query
        event_terms = {"tubbs", "santa rosa", "wildfire", "2017", "fire"}
        if not any(term in query.lower() for term in event_terms):
            log.warning("Parser produced vague external_query: %r — enriching", query)
            return f"2017 Tubbs Fire Santa Rosa: {query}"
        return query
