from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import Optional
from app.config import settings


class QueryRewriter:

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model or settings.GOOGLE_LITE_MODEL,
            temperature=(
                temperature if temperature is not None else settings.TEMPERATURE
            ),
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self.prompt = PromptTemplate(
            input_variables=["history", "query"],
            template="""
You are rewriting a follow-up question to be fully self-contained.

Conversation History:
{history}

Current User Query:
{query}

Rules:
- Identify what the user is currently asking for (intent) from the query AND conversation.
- If the current query is a short answer (e.g., "5", "Santa Rosa"), it is providing a missing parameter for the previous question.
- Fill in ALL missing information from the conversation history: city, damage level, number of buildings, etc.
- Preserve the original intent from the previous assistant question or user request.
- Output a complete, standalone question.

Examples:
History: "User: Show top damaged buildings in Santa Rosa. Assistant: How many buildings would you like to see?"
Query: "5"
Rewritten: "Show the top 5 most damaged buildings in Santa Rosa"

History: "User: Compare Santa Rosa and another city. Assistant: Which other city?"
Query: "Oakland"
Rewritten: "Compare damage between Santa Rosa and Oakland"

Return ONLY the rewritten query.
""",
        )

    def rewrite(self, history: str, query: str) -> str:
        try:
            chain = self.prompt | self.llm

            response = chain.invoke({"history": history, "query": query})

            rewritten = response.content.strip()

            if not rewritten or len(rewritten) < 5:
                return query

            return rewritten

        except Exception:
            return query
