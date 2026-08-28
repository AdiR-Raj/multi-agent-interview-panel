import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path regardless of execution entrypoint
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings

app = FastAPI(
    title="Multi-Agent AI Interview Panel Simulator",
    description="Deterministic multi-agent interview panel evaluation with traceable evidence, structured debate, and non-averaging synthesis.",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend status for Cloud Run and monitoring."""
    return {
        "status": "healthy",
        "service": "Multi-Agent AI Interview Panel Simulator",
        "openai_base_url": settings.OPENAI_BASE_URL,
        "openai_model": settings.OPENAI_MODEL,
    }


# Fallback / root endpoint handling
index_html = settings.FRONTEND_DIR / "index.html"
if index_html.is_file():
    app.mount("/", StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {
            "status": "healthy",
            "service": "Multi-Agent AI Interview Panel Simulator",
            "health_endpoint": "/api/health",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )
