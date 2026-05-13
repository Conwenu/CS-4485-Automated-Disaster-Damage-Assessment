import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    MODEL_NAME: str = "llama3.1:8b"
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    GOOGLE_API_KEY: str = os.getenv("GEMINI_API_KEY", None)
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    GOOGLE_LITE_MODEL: str = os.getenv("GOOGLE_LITE_MODEL", "gemini-2.5-flash-lite")

    CONTEXT_INDEXING_DELAY_SECONDS = 1

    USE_LLM_RESPONSE_POLISH: bool = (
        os.getenv("USE_LLM_RESPONSE_POLISH", "false").lower() == "true"
    )

    EVAL_REQUEST_DELAY_SECONDS: float = float(
        os.getenv("EVAL_REQUEST_DELAY_SECONDS", "13")
    )


settings = Settings()
