import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Project directories
    DATA_DIR: Path = ROOT_DIR / "data"
    FRONTEND_DIR: Path = ROOT_DIR / "frontend"


settings = Settings()

