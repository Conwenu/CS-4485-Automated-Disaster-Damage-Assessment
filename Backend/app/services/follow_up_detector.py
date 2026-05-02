from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


class FollowUpDetector:

    def __init__(self, model_name="llama3.1:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0)

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
"""
        )

    def is_follow_up(self, history: str, query: str) -> bool:
        try:
            chain = self.prompt | self.llm

            response = chain.invoke({
                "history": history,
                "query": query
            })

            answer = response.content.strip().upper()

            return answer == "YES"

        except Exception:
            return False
        