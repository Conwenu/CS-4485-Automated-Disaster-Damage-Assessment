from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class QueryRewriter:

    def __init__(self, model_name="llama3.1:8b"):
        self.llm = ChatOllama(model=model_name, temperature=0)

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
"""
        )

    def rewrite(self, history: str, query: str) -> str:
        try:
            chain = self.prompt | self.llm

            response = chain.invoke({
                "history": history,
                "query": query
            })

            rewritten = response.content.strip()

            if not rewritten or len(rewritten) < 5:
                return query

            return rewritten

        except Exception:
            return query
        