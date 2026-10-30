"""FastAPI routes for Resume Screening, Ranking, SHAP Explainability, Fairness, and Evaluation."""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from backend.schemas import (
    CandidateExplanationResponse,
    CandidateRankingResponse,
    CandidateRankItem,
    EvaluationMetricsResponse,
    FairnessReportResponse,
    HealthResponse,
    JobDescriptionUploadResponse,
    ResumeUploadItem,
    ResumeUploadResponse,
)
from backend.services.evaluation import evaluation_service
from backend.services.explainability import explainability_service
from backend.services.fairness import fairness_service
from backend.services.ranking_service import ranking_service
from backend.services.resume_parser import resume_parser

logger = logging.getLogger(__name__)

router = APIRouter()

# In-Memory Session Storage (Temporary, safe session storage per backend lifecycle)
SESSION_STATE: Dict[str, Any] = {
    "active_jd": None,
    "resumes": {},  # candidate_id -> parsed profile dict
    "rankings": [],  # list of ranked candidate items
    "demographic_eval": {},  # candidate_id -> group_label (when provided)
    "evaluation_dataset": None,  # ground truth eval data
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per file
ALLOWED_JD_EXTENSIONS = {"txt", "pdf"}
ALLOWED_RESUME_EXTENSIONS = {"pdf", "docx", "doc", "txt"}


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Return application health status and explainability readiness."""
    return HealthResponse(
        status="operational",
        version="1.0.0",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        shap_ready=explainability_service.shap_available,
        active_session_candidates=len(SESSION_STATE["resumes"]),
        active_jd_present=SESSION_STATE["active_jd"] is not None,
    )


@router.post("/upload-jd", response_model=JobDescriptionUploadResponse, tags=["Job Description"])
async def upload_job_description(file: UploadFile = File(...)) -> JobDescriptionUploadResponse:
    """Upload and process a Job Description (TXT or PDF)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    ext = file.filename.lower().split(".")[-1]
    if ext not in ALLOWED_JD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '.{ext}'. Job description must be TXT or PDF.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024*1024)}MB).",
        )

    try:
        raw_text = resume_parser.extract_text_from_bytes(file_bytes, file.filename)
        cleaned_text = resume_parser.clean_text(raw_text)
        jd_id = f"jd_{uuid.uuid4().hex[:8]}"
        parsed_jd = ranking_service.parse_job_description(cleaned_text, jd_id=jd_id)

        # Store in session state
        SESSION_STATE["active_jd"] = parsed_jd
        # Invalidate previous rankings since JD changed
        SESSION_STATE["rankings"] = []

        preview = cleaned_text[:300] + ("..." if len(cleaned_text) > 300 else "")

        return JobDescriptionUploadResponse(
            message="Job description successfully parsed and set as active.",
            jd_id=jd_id,
            title=parsed_jd["title"],
            word_count=parsed_jd["word_count"],
            required_skills=parsed_jd["required_skills"],
            preview=preview,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error parsing job description: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process job description: {str(e)}")


@router.post("/upload-resumes", response_model=ResumeUploadResponse, tags=["Resumes"])
async def upload_resumes(files: List[UploadFile] = File(...)) -> ResumeUploadResponse:
    """Upload multiple resume files (PDF, DOCX, TXT). Validates each file and extracts structured data."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    items: List[ResumeUploadItem] = []
    success_count = 0
    fail_count = 0

    for file in files:
        if not file.filename:
            fail_count += 1
            items.append(
                ResumeUploadItem(
                    filename="unknown",
                    candidate_id="none",
                    candidate_name="Unknown",
                    status="failed",
                    error="Missing filename.",
                )
            )
            continue

        ext = file.filename.lower().split(".")[-1]
        if ext not in ALLOWED_RESUME_EXTENSIONS:
            fail_count += 1
            items.append(
                ResumeUploadItem(
                    filename=file.filename,
                    candidate_id="none",
                    candidate_name="Unknown",
                    status="failed",
                    error=f"Unsupported format '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_RESUME_EXTENSIONS))}",
                )
            )
            continue

        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            fail_count += 1
            items.append(
                ResumeUploadItem(
                    filename=file.filename,
                    candidate_id="none",
                    candidate_name="Unknown",
                    status="failed",
                    error="File size exceeds 10MB limit.",
                )
            )
            continue

        try:
            cand_id = f"cand_{uuid.uuid4().hex[:6]}"
            parsed = resume_parser.parse_resume(file_bytes, file.filename, cand_id)

            # Apply demographic anonymization to candidate display name to protect privacy
            # Candidate 1, Candidate 2, or masked name
            parsed["anonymized_display"] = f"Candidate {len(SESSION_STATE['resumes']) + 1}"

            SESSION_STATE["resumes"][cand_id] = parsed
            success_count += 1

            items.append(
                ResumeUploadItem(
                    filename=file.filename,
                    candidate_id=cand_id,
                    candidate_name=parsed["anonymized_display"],
                    status="processed",
                    extracted_skills=parsed["skills"],
                    experience_years=parsed["experience_years"],
                )
            )
        except ValueError as ve:
            fail_count += 1
            items.append(
                ResumeUploadItem(
                    filename=file.filename,
                    candidate_id="none",
                    candidate_name="Unknown",
                    status="failed",
                    error=str(ve),
                )
            )
        except Exception as e:
            fail_count += 1
            logger.error(f"Error parsing {file.filename}: {e}")
            items.append(
                ResumeUploadItem(
                    filename=file.filename,
                    candidate_id="none",
                    candidate_name="Unknown",
                    status="failed",
                    error=f"Internal extraction error: {str(e)}",
                )
            )

    return ResumeUploadResponse(
        message=f"Batch processed: {success_count} succeeded, {fail_count} failed.",
        total_uploaded=len(files),
        successful=success_count,
        failed=fail_count,
        items=items,
    )


@router.post("/rank-candidates", response_model=CandidateRankingResponse, tags=["Ranking"])
async def rank_candidates() -> CandidateRankingResponse:
    """Rank all currently uploaded candidates against the active job description."""
    jd = SESSION_STATE["active_jd"]
    if not jd:
        raise HTTPException(
            status_code=400,
            detail="No active job description found. Please upload a job description via /upload-jd first.",
        )

    resumes = list(SESSION_STATE["resumes"].values())
    if not resumes:
        raise HTTPException(
            status_code=400,
            detail="No resumes currently uploaded. Please upload candidate resumes via /upload-resumes first.",
        )

    ranked_list = ranking_service.rank_candidates(resumes, jd)
    SESSION_STATE["rankings"] = ranked_list

    rank_items = [
        CandidateRankItem(
            rank=r["rank"],
            candidate_id=r["candidate_id"],
            candidate_name=r.get("candidate_name", "Candidate"),
            match_score=r["match_score"],
            matched_skills=r["matched_skills"],
            missing_skills=r["missing_skills"],
            experience_years=r["experience_years"],
            education_level=r["education_level"],
            explanation_available=True,
            summary=r["summary"],
        )
        for r in ranked_list
    ]

    return CandidateRankingResponse(
        jd_title=jd["title"],
        total_candidates=len(rank_items),
        rankings=rank_items,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/candidate/{candidate_id}/explanation", response_model=CandidateExplanationResponse, tags=["Explainability"])
async def get_candidate_explanation(
    candidate_id: str,
    force_fallback: bool = Query(False, description="Force transparent fallback attribution instead of SHAP")
) -> CandidateExplanationResponse:
    """Generate SHAP or transparent fallback explainability for a specific candidate."""
    jd = SESSION_STATE["active_jd"]
    if not jd:
        raise HTTPException(
            status_code=400,
            detail="No active job description found. Upload a job description first.",
        )

    candidate = SESSION_STATE["resumes"].get(candidate_id)
    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate with ID '{candidate_id}' not found in active session.",
        )
    
    explanation = explainability_service.explain_candidate(
        candidate_profile=candidate,
        jd_profile=jd,
        force_fallback=force_fallback
    )

    return CandidateExplanationResponse(**explanation)


@router.get("/fairness-report", response_model=FairnessReportResponse, tags=["Fairness & Bias"])
async def get_fairness_report(
    threshold: float = Query(70.0, description="Match score cutoff for selection rate evaluation")
) -> FairnessReportResponse:
    """Generate fairness and bias mitigation report.

    Calculates group fairness metrics if lawful evaluation data exists.
    Otherwise reports demographic data unavailable status and debiasing guardrails.
    """
    ranked_candidates = SESSION_STATE["rankings"]
    candidate_scores = [
        {"candidate_id": r["candidate_id"], "match_score": r["match_score"]}
        for r in ranked_candidates
    ]

    # Demographic data for fairness evaluation (if provided/active)
    demo_data = SESSION_STATE.get("demographic_eval", {})

    report = fairness_service.evaluate_group_fairness(
        candidate_scores=candidate_scores,
        demographic_data=demo_data,
        selection_threshold=threshold,
    )

    # Also perform proxy audits on all uploaded resumes in session
    proxy_summary = {"total_resumes_checked": len(SESSION_STATE["resumes"]), "flagged_candidates": 0, "details": []}
    for cid, cand in SESSION_STATE["resumes"].items():
        audit = fairness_service.detect_proxy_features(cand["raw_text"])
        if audit["potential_proxies_detected"]:
            proxy_summary["flagged_candidates"] += 1
            proxy_summary["details"].append({
                "candidate_id": cid,
                "proxy_flags": audit["categories"],
            })

    report["proxy_detection_summary"] = proxy_summary

    return FairnessReportResponse(**report)


@router.get("/evaluation", response_model=EvaluationMetricsResponse, tags=["Evaluation"])
async def get_evaluation_metrics(
    k: int = Query(5, description="Top-K cutoff for ranking metrics"),
    threshold: float = Query(70.0, description="Score threshold for classification metrics")
) -> EvaluationMetricsResponse:
    """Return model evaluation metrics (Precision, Recall, F1, ROC-AUC, NDCG@K, Precision@K, MRR).

    Returns 'Evaluation data required' state if no ground-truth benchmark is active.
    """
    eval_dataset = SESSION_STATE.get("evaluation_dataset")

    if not eval_dataset:
        return EvaluationMetricsResponse(
            status="Evaluation data required",
            data_available=False,
            notes=[
                "No ground-truth evaluation dataset has been loaded.",
                "To inspect genuine evaluation metrics, load the synthetic benchmark dataset using POST /load-benchmark.",
                "Evaluation metrics are never fabricated without real or test ground-truth labels."
            ]
        )

    metrics = evaluation_service.compute_metrics(
        candidate_eval_items=eval_dataset.get("items", []),
        k=k,
        threshold=threshold,
        dataset_name=eval_dataset.get("name", "Benchmark Dataset")
    )

    return EvaluationMetricsResponse(**metrics)


@router.post("/load-benchmark", tags=["Evaluation & Demo"])
async def load_benchmark_evaluation_data() -> Dict[str, Any]:
    """Helper endpoint to load the pre-configured fictional benchmark dataset for college demonstration."""
    benchmark_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "sample_data",
        "benchmark_eval_data.json"
    )

    if not os.path.exists(benchmark_file):
        raise HTTPException(status_code=404, detail="Benchmark dataset file not found.")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    SESSION_STATE["evaluation_dataset"] = {
        "name": data.get("dataset_name", "Fictional Candidate Benchmark"),
        "items": data.get("candidates", [])
    }

    # Also load demographic labels into demographic_eval for fairness audit demo
    demo_dict = {}
    for cand in data.get("candidates", []):
        if "demographic_group" in cand:
            demo_dict[cand["candidate_id"]] = cand["demographic_group"]

    SESSION_STATE["demographic_eval"] = demo_dict

    return {
        "message": "Benchmark dataset successfully loaded into session.",
        "dataset_name": data.get("dataset_name"),
        "candidate_count": len(data.get("candidates", [])),
        "demographic_groups_loaded": len(set(demo_dict.values())),
    }


@router.post("/reset-session", tags=["System"])
async def reset_session() -> Dict[str, str]:
    """Reset session state, clearing uploaded resumes and job descriptions."""
    SESSION_STATE["active_jd"] = None
    SESSION_STATE["resumes"] = {}
    SESSION_STATE["rankings"] = []
    SESSION_STATE["demographic_eval"] = {}
    SESSION_STATE["evaluation_dataset"] = None
    return {"message": "Session state cleared successfully."}
