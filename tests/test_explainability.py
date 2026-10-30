"""Tests for Explainable AI module (Phase 8 Module)."""

import pytest
from backend.services.ranking_service import ranking_service
from backend.services.explainability import explainability_service


@pytest.fixture
def test_setup():
    jd_text = (
        "Backend Python Engineer\n"
        "Requirements: 3+ years experience with Python, SQL, and Docker.\n"
        "Degree in Computer Science."
    )
    jd = ranking_service.parse_job_description(jd_text, "jd_exp")

    cand = {
        "candidate_id": "cand_exp_1",
        "candidate_name": "Alice Candidate",
        "raw_text": "Experienced Python Engineer with 4 years in Python and SQL. Missing Docker. Bachelor's degree.",
        "skills": ["python", "sql"],
        "experience_years": 4.0,
        "education": "Bachelor's",
        "word_count": 40,
    }
    return cand, jd


def test_explain_candidate_structure(test_setup):
    cand, jd = test_setup
    res = explainability_service.explain_candidate(cand, jd)

    assert "match_score" in res
    assert "base_score" in res
    assert "feature_contributions" in res
    assert "positive_contributions" in res
    assert "negative_contributions" in res
    assert "human_readable_summary" in res
    assert "chart_data" in res
    assert len(res["feature_contributions"]) == 6


def test_human_readable_explanation_content(test_setup):
    cand, jd = test_setup
    res = explainability_service.explain_candidate(cand, jd)
    summary = res["human_readable_summary"]

    assert len(summary) >= 3
    # Check for skill recognition
    summary_text = " ".join(summary)
    assert "Strong match:" in summary_text or "skill" in summary_text.lower()
    assert "Overall match score:" in summary_text


def test_fallback_mode_attribution_label(test_setup):
    cand, jd = test_setup
    # Force fallback mode
    res = explainability_service.explain_candidate(cand, jd, force_fallback=True)

    assert res["is_shap_explanation"] is False
    assert "Fallback" in res["explanation_method"]
    assert "Not Genuine SHAP" in res["explanation_method"]
    assert len(res["feature_contributions"]) == 6
    assert "chart_data" in res
