"""Tests for FastAPI backend REST API endpoints (Phase 10)."""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data
    assert "shap_ready" in data


def test_upload_jd_valid_txt():
    jd_content = b"Senior Software Engineer\nMust have Python, FastAPI, and SQL skills. 4 years exp."
    files = {"file": ("job_desc.txt", io.BytesIO(jd_content), "text/plain")}
    response = client.post("/upload-jd", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "jd_id" in data
    assert "python" in data["required_skills"]
    assert "fastapi" in data["required_skills"]


def test_upload_jd_invalid_extension():
    files = {"file": ("invalid.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    response = client.post("/upload-jd", files=files)
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_upload_resumes_valid():
    resume1 = b"Candidate Alice Smith\nPython and SQL developer with 5 years experience."
    resume2 = b"Candidate Bob Jones\nFastAPI and Docker engineer with 3 years experience."

    files = [
        ("files", ("alice.txt", io.BytesIO(resume1), "text/plain")),
        ("files", ("bob.txt", io.BytesIO(resume2), "text/plain")),
    ]
    response = client.post("/upload-resumes", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 2
    assert len(data["items"]) == 2


def test_rank_candidates_workflow():
    # 1. Reset
    client.post("/reset-session")

    # 2. Upload JD
    jd_content = b"Backend Python Dev\nNeed Python, FastAPI, and SQL with 4 years experience."
    client.post("/upload-jd", files={"file": ("jd.txt", io.BytesIO(jd_content), "text/plain")})

    # 3. Upload Resumes
    resume_content = b"Alice Smith\nPython, FastAPI, SQL expert with 5 years experience."
    client.post("/upload-resumes", files=[("files", ("alice.txt", io.BytesIO(resume_content), "text/plain"))])

    # 4. Rank Candidates
    rank_res = client.post("/rank-candidates")
    assert rank_res.status_code == 200
    rank_data = rank_res.json()
    assert rank_data["total_candidates"] == 1
    assert rank_data["rankings"][0]["match_score"] > 60.0

    cand_id = rank_data["rankings"][0]["candidate_id"]

    # 5. Candidate Explanation
    exp_res = client.get(f"/candidate/{cand_id}/explanation")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["candidate_id"] == cand_id
    assert len(exp_data["feature_contributions"]) == 6
    assert len(exp_data["human_readable_summary"]) > 0


def test_fairness_report_endpoint():
    response = client.get("/fairness-report")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "warnings" in data
    assert "limitations" in data


def test_evaluation_metrics_missing_data():
    client.post("/reset-session")
    response = client.get("/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Evaluation data required"
    assert data["data_available"] is False


def test_load_benchmark_and_evaluation():
    # Load benchmark
    load_res = client.post("/load-benchmark")
    assert load_res.status_code == 200
    assert load_res.json()["candidate_count"] > 0

    # Query evaluation
    eval_res = client.get("/evaluation")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert eval_data["status"] == "Available"
    assert eval_data["data_available"] is True
    assert eval_data["classification_metrics"]["precision"] > 0.0
    assert eval_data["ranking_metrics"]["ndcg_at_k"] > 0.0
