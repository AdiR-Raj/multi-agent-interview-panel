import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def get_config_val(key: str, default: str = "") -> str:
    """Retrieves config from Streamlit secrets (if present) or environment variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets[key]
            if val:
                return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


class Settings:
    """Application settings loaded dynamically from Streamlit secrets or .env.

    LLM provider resolution order (Groq takes precedence):
      1. GROQ_API_KEY / GROQ_BASE_URL / GROQ_MODEL  (Groq)
      2. OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL  (OpenAI fallback)
    """

    # --- Groq-priority unified LLM properties ---

    @property
    def LLM_API_KEY(self) -> str:
        """Returns Groq API key if set, otherwise falls back to OpenAI key."""
        groq_key = get_config_val("GROQ_API_KEY", "")
        if groq_key:
            return groq_key
        return get_config_val("OPENAI_API_KEY", "")

    @property
    def LLM_BASE_URL(self) -> str:
        """Returns Groq base URL if GROQ_API_KEY is set, otherwise OpenAI base URL."""
        groq_key = get_config_val("GROQ_API_KEY", "")
        if groq_key:
            return get_config_val("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return get_config_val("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def LLM_MODEL(self) -> str:
        """Returns Groq model if GROQ_API_KEY is set, otherwise OpenAI model."""
        groq_key = get_config_val("GROQ_API_KEY", "")
        if groq_key:
            return get_config_val("GROQ_MODEL", "openai/gpt-oss-120b")
        return get_config_val("OPENAI_MODEL", "gpt-4o-mini")

    # --- Backwards-compatible OPENAI_* aliases (for health endpoint & tests) ---

    @property
    def OPENAI_API_KEY(self) -> str:
        return get_config_val("OPENAI_API_KEY", "")

    @property
    def OPENAI_BASE_URL(self) -> str:
        return get_config_val("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def OPENAI_MODEL(self) -> str:
        return get_config_val("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def HOST(self) -> str:
        return get_config_val("HOST", "0.0.0.0")

    @property
    def PORT(self) -> int:
        return int(get_config_val("PORT", "8000"))

    # Project directories
    DATA_DIR: Path = ROOT_DIR / "data"
    FRONTEND_DIR: Path = ROOT_DIR / "frontend"


settings = Settings()
