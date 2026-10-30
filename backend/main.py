"""Main FastAPI Application Entrypoint

AI-Powered Resume Screening and Ranking System with Explainable AI.
Provides REST APIs for:
- Health check
- Job description upload and text extraction
- Multi-format resume upload (PDF, DOCX, TXT)
- Candidate ranking using ML scoring model
- Explainable AI with SHAP TreeExplainer and transparent fallback
- Bias mitigation and group fairness auditing
- Model and ranking evaluation metrics (Precision, Recall, F1, ROC-AUC, NDCG@K, MRR)
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend.main")

app = FastAPI(
    title="AI-Powered Resume Screening & Explainable AI API",
    description=(
        "REST API for Person 4: Explainability, Fairness, API & Frontend Engineer.\n\n"
        "Integrates SHAP for Explainable AI, responsible fairness checks, group parity analysis, "
        "and classification & ranking evaluation metrics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main router
app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=REPO_ROOT)
