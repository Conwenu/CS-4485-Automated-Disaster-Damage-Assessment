from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from typing import Optional
from app.config import settings


class FollowUpDetector:

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
You are a classifier that determines whether a user query is a follow-up question.

A follow-up question depends on previous context.
A standalone question is fully self-contained.

Conversation History:
{history}

User Query:
{query}

Answer ONLY with:
YES or NO
""",
        )

    def is_follow_up(self, history: str, query: str) -> bool:
        try:
            chain = self.prompt | self.llm

            response = chain.invoke({"history": history, "query": query})

            answer = response.content.strip().upper()

            return answer == "YES"

        except Exception:
            return False
