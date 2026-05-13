"""Pydantic schema for structured LLM output.

ParsedQuery is the strict contract between the parser and the rest of the
system. The LLM is forced into this shape via `.with_structured_output`,
so downstream code can trust every field's type.

Field order matters: `reasoning` appears first so the model is encouraged
to think step-by-step before committing to an intent.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from app.models.enums import ChatIntent


class ParsedQuery(BaseModel):
    # ----- Model's own reasoning (kept for debugging and CoT effect) -----
    reasoning: str = Field(
        description=(
            "One or two short sentences explaining WHY you chose this intent "
            "and HOW you interpreted the query. Always fill this in first. "
            "This forces step-by-step thinking and helps with debugging."
        )
    )

    # ----- Follow-up / context resolution -----
    is_follow_up: bool = Field(
        description=(
            "True if the query depends on prior conversation context "
            "(e.g., 'what about Miami?', 'compare it with X', '5' as an "
            "answer to 'how many?', or a lone city name after a question)."
        )
    )
    rewritten_query: Optional[str] = Field(
        None,
        description=(
            "If is_follow_up is true, a fully self-contained rewrite of the "
            "query with all referents resolved from history. Null otherwise."
        ),
    )
    is_answer_to_pending: bool = Field(
        False,
        description=(
            "True ONLY when the user is directly answering a pending "
            "clarification question from the previous assistant turn. "
            "A brand-new unrelated question sets this false even if there "
            "is pending clarification — the user has pivoted."
        ),
    )

    # ----- Intent -----
    intent: ChatIntent = Field(
        description="The intent category. Use OUT_OF_SCOPE only when nothing else fits."
    )

    # ----- Parameters (all optional; fill only when extractable) -----
    city: Optional[str] = Field(None, description="A single city name from the query.")
    cities: List[str] = Field(
        default_factory=list,
        description="Two or more cities (e.g., for COMPARE_LOCATIONS).",
    )
    id: Optional[str] = Field(None, description="A single building ID.")
    ids: List[str] = Field(
        default_factory=list, description="Two or more building IDs."
    )
    scene_id: Optional[str] = Field(None, description="A scene identifier.")
    top_k: Optional[int] = Field(None, description="Number of items requested.")
    damage_level: Optional[
        Literal["no-damage", "minor-damage", "major-damage", "destroyed"]
    ] = Field(None, description="Normalized damage level.")
    confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold between 0 and 1."
    )
    direction: Optional[Literal["above", "below"]] = Field(
        None, description="Direction for confidence outliers."
    )
    status: Optional[Literal["ok", "failed"]] = Field(
        None, description="Processing status filter."
    )

    # ----- External knowledge -----
    needs_external_knowledge: bool = Field(
        False,
        description=(
            "True if the query contains a general-knowledge component the "
            "dataset alone cannot answer (definitions, explanations, procedures)."
        ),
    )
    external_query: Optional[str] = Field(
        None,
        description=(
            "Only the general-knowledge portion of the query, rephrased for "
            "a knowledge-base lookup. Null if needs_external_knowledge is false."
        ),
    )

    # ----- Clarification -----
    needs_clarification: bool = Field(
        False,
        description=(
            "True only when a REQUIRED parameter for the chosen intent is "
            "missing AND cannot be inferred from history."
        ),
    )
    clarification_question: Optional[str] = Field(
        None, description="The question to show the user."
    )
    missing_param: Optional[str] = Field(
        None,
        description=(
            "Name of the missing parameter (e.g., 'city', 'top_k', 'ids'). "
            "Must match a field name on this schema."
        ),
    )
