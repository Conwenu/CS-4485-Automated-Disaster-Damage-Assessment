import json
from langchain_ollama import OllamaLLM
from langchain_classic.schema import HumanMessage, SystemMessage

class JsonRepairTool:
    """
    A helper that returns a valid JSON string if possible.
    It first attempts to parse the input directly. If parsing fails,
    it uses an LLM to fix common JSON syntax errors.
    """
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0):
        self.llm = OllamaLLM(model=model_name, temperature=temperature)

    def ensure_json(self, input_str: str) -> str:
        try:
            parsed = json.loads(input_str)
            return json.dumps(parsed)
        except json.JSONDecodeError:
            pass  

        # Step 2: Attempt repair with LLM
        repaired = self._repair_with_llm(input_str)

        # Step 3: Final validation
        try:
            parsed = json.loads(repaired)
            return json.dumps(parsed)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to produce valid JSON even after LLM repair. "
                f"Original error: {e}\nRepaired output was:\n{repaired}"
            )

    def _repair_with_llm(self, invalid_json: str) -> str:
        system_msg = SystemMessage(
            content=(
                "You are a JSON syntax repair tool. "
                "Your ONLY task is to fix invalid JSON syntax and return a VALID JSON string. "
                "Do NOT add, remove, or interpret data. Do NOT change the structure or values. "
                "Only correct errors such as: trailing commas, missing quotes around keys, "
                "unescaped characters, or unbalanced brackets. "
                "If the input is already valid JSON, return it exactly. "
                "Output ONLY the raw JSON string, without any markdown, explanations, or extra text."
            )
        )
        human_msg = HumanMessage(content=f"Fix the JSON syntax in the following text:\n\n{invalid_json}")

        response = self.llm([system_msg, human_msg])
        return response.content.strip()