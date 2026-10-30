"""Tests for ranking service and feature engineering (Person 2 & 3 Modules)."""

import pytest
import numpy as np
from backend.services.ranking_service import ranking_service, FEATURE_NAMES


@pytest.fixture
def sample_jd():
    jd_text = (
        "Senior Python Engineer\n"
        "Requirements:\n"
        "- 4+ years of experience with Python and SQL.\n"
        "- Experience with Docker and FastAPI.\n"
        "- Degree in Computer Science or related field."
    )
    return ranking_service.parse_job_description(jd_text, "test_jd")


@pytest.fixture
def strong_candidate():
    return {
        "candidate_id": "cand_strong",
        "candidate_name": "Strong Candidate",
        "raw_text": "Experienced Python Engineer with 5 years in FastAPI, SQL, and Docker. Master of Science degree.",
        "skills": ["python", "sql", "docker", "fastapi"],
        "experience_years": 5.0,
        "education": "Master's",
        "word_count": 50,
    }


@pytest.fixture
def weak_candidate():
    return {
        "candidate_id": "cand_weak",
        "candidate_name": "Weak Candidate",
        "raw_text": "Recent graduate with basic HTML and CSS knowledge. 0.5 years experience.",
        "skills": ["html", "css"],
        "experience_years": 0.5,
        "education": "High School",
        "word_count": 25,
    }


def test_parse_job_description(sample_jd):
    assert "Senior Python Engineer" in sample_jd["title"]
    assert sample_jd["expected_years"] == 4.0
    assert "python" in sample_jd["required_skills"]
    assert "sql" in sample_jd["required_skills"]
    assert "fastapi" in sample_jd["required_skills"]


def test_compute_features(sample_jd, strong_candidate):
    feat_vec, meta = ranking_service.compute_features(strong_candidate, sample_jd)
    assert len(feat_vec) == len(FEATURE_NAMES)
    # Skills match ratio should be 1.0 (all 4 skills matched)
    assert meta["skills_match_ratio"] == 1.0
    assert meta["missing_skills_count"] == 0
    assert meta["experience_alignment"] >= 1.0


def test_score_candidate_monotonicity(sample_jd, strong_candidate, weak_candidate):
    strong_score, _, _ = ranking_service.score_candidate(strong_candidate, sample_jd)
    weak_score, _, _ = ranking_service.score_candidate(weak_candidate, sample_jd)

    assert 0.0 <= strong_score <= 100.0
    assert 0.0 <= weak_score <= 100.0
    assert strong_score > weak_score, f"Strong candidate ({strong_score}) must outscore weak candidate ({weak_score})"


def test_rank_candidates(sample_jd, strong_candidate, weak_candidate):
    profiles = [weak_candidate, strong_candidate]
    ranked = ranking_service.rank_candidates(profiles, sample_jd)

    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "cand_strong"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate_id"] == "cand_weak"
    assert ranked[1]["rank"] == 2
    assert ranked[0]["match_score"] >= ranked[1]["match_score"]
