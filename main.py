"""FastAPI entry point for the Real-Time UPI Risk Intelligence Platform."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Real-Time UPI Transaction Risk Intelligence Platform",
    description="Risk scoring, decisioning, profiling, and explainability APIs for UPI transactions.",
    version="1.0.0",
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    """Return a small service descriptor."""
    return {
        "service": "Real-Time UPI Transaction Risk Intelligence Platform",
        "docs": "/docs",
        "health": "/health",
    }
