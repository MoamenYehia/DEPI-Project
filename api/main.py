"""
FastAPI application entry point.

Start server:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import anomalies, data_ingest, forecasting, insights


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI BI Agent API — started")
    yield
    print("AI BI Agent API — stopped")


app = FastAPI(
    title="AI Business Intelligence Agent",
    description=(
        "Autonomous BI system with predictive sales forecasting (Prophet + MLflow), "
        "anomaly detection, and RAG-powered strategic recommendations "
        "(LangChain + ChromaDB + Groq | Ollama | Azure OpenAI)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_ingest.router, prefix="/api", tags=["Data Ingestion"])
app.include_router(forecasting.router, prefix="/api", tags=["Forecasting (M2)"])
app.include_router(anomalies.router,   prefix="/api", tags=["Anomaly Detection (M2)"])
app.include_router(insights.router,    prefix="/api", tags=["AI Insights (M3)"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "service": "AI BI Agent API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    from core.config import settings

    return {
        "status":        "healthy",
        "llm_provider":  settings.LLM_PROVIDER.value,
        "data_dir":      str(settings.DATA_DIR),
        "processed_dir": str(settings.PROCESSED_DIR),
        "vector_db_dir": str(settings.VECTOR_DB_DIR),
    }
