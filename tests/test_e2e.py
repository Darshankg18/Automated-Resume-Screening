"""End-to-end workflow integration test across all 4 phases (Person 4)."""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_full_recruitment_screening_workflow():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Step 1: Health Check
    health_res = client.get("/health")
    assert health_res.status_code == 200

    # Step 2: Upload Real Sample Job Description
    jd_file_path = os.path.join(repo_root, "sample_data", "job_descriptions", "senior_python_dev.txt")
    with open(jd_file_path, "rb") as f:
        jd_res = client.post("/upload-jd", files={"file": ("senior_python_dev.txt", f, "text/plain")})
    assert jd_res.status_code == 200
    jd_data = jd_res.json()
    assert "Senior Python Backend Engineer" in jd_data["title"]

    # Step 3: Upload Sample Resumes (Alice Smith, Bob Jones, Diana Kim)
    resumes_dir = os.path.join(repo_root, "sample_data", "resumes")
    files_to_upload = [
        "candidate_1_alice_smith_senior_dev.txt",
        "candidate_2_bob_jones_mid_dev.txt",
        "candidate_4_diana_kim_junior_dev.txt",
    ]

    upload_payload = []
    file_handles = []
    for rname in files_to_upload:
        fpath = os.path.join(resumes_dir, rname)
        fh = open(fpath, "rb")
        file_handles.append(fh)
        upload_payload.append(("files", (rname, fh, "text/plain")))

    try:
        batch_res = client.post("/upload-resumes", files=upload_payload)
        assert batch_res.status_code == 200
        batch_data = batch_res.json()
        assert batch_data["successful"] == 3
    finally:
        for fh in file_handles:
            fh.close()

    # Step 4: Rank Candidates
    rank_res = client.post("/rank-candidates")
    assert rank_res.status_code == 200
    rank_data = rank_res.json()
    assert rank_data["total_candidates"] == 3
    rankings = rank_data["rankings"]

    # Check rank ordering: Alice (senior) should be #1, Diana (junior) should be #3
    assert rankings[0]["rank"] == 1
    assert rankings[0]["match_score"] >= rankings[1]["match_score"] >= rankings[2]["match_score"]

    top_candidate_id = rankings[0]["candidate_id"]

    # Step 5: Candidate Explanation (SHAP)
    exp_res = client.get(f"/candidate/{top_candidate_id}/explanation")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["candidate_id"] == top_candidate_id
    assert exp_data["match_score"] > 70.0
    assert len(exp_data["positive_contributions"]) > 0
    assert len(exp_data["human_readable_summary"]) >= 3

    # Step 6: Fairness Report
    fairness_res = client.get("/fairness-report?threshold=70.0")
    assert fairness_res.status_code == 200
    fairness_data = fairness_res.json()
    assert "status" in fairness_data
    assert "limitations" in fairness_data
    assert len(fairness_data["limitations"]) > 0

    # Step 7: Load Benchmark and Check Evaluation Metrics
    bench_res = client.post("/load-benchmark")
    assert bench_res.status_code == 200

    eval_res = client.get("/evaluation?k=5&threshold=70.0")
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    assert eval_data["status"] == "Available"
    assert eval_data["data_available"] is True
    assert eval_data["classification_metrics"]["f1_score"] > 0.0
    assert eval_data["ranking_metrics"]["ndcg_at_k"] > 0.0
