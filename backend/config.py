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
    """Application settings loaded dynamically from Streamlit secrets or .env."""

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
