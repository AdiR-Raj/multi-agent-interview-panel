from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.config import settings

app = FastAPI(
    title="Multi-Agent AI Interview Panel Simulator",
    description="Deterministic multi-agent interview panel evaluation with traceable evidence and structured debate.",
    version="0.1.0",
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "healthy",
        "service": "Multi-Agent AI Interview Panel Simulator",
        "openai_base_url": settings.OPENAI_BASE_URL,
        "openai_model": settings.OPENAI_MODEL,
    }


# Mount frontend static directory if exists
if os.path.isdir(settings.FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)

