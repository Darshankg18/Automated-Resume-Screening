"""Tests for Bias Mitigation and Fairness Checks (Phase 9 Module)."""

import pytest
from backend.services.fairness import fairness_service


def test_mask_demographic_identifiers():
    sample_text = (
        "Mr. John Doe\n"
        "He has 5 years of experience. His primary language is Python. "
        "Date of Birth: 15/08/1990. Marital status: Single."
    )

    masked, redactions = fairness_service.mask_demographic_identifiers(sample_text, candidate_name="John Doe")

    assert "John Doe" not in masked
    assert "[CANDIDATE_NAME]" in masked
    assert "Mr." not in masked
    assert "[TITLE]" in masked
    assert "Date of Birth" not in masked
    assert "[SENSITIVE_FIELD]" in masked
    assert len(redactions) > 0


def test_detect_proxy_features():
    text_with_proxies = (
        "Active member of Society of Women Engineers and Women in Tech. "
        "Graduated class of 1988 with High School diploma."
    )
    result = fairness_service.detect_proxy_features(text_with_proxies)

    assert result["potential_proxies_detected"] is True
    assert result["total_proxy_flags"] >= 2
    assert "Affinity / Diversity Organizations" in result["categories"]
    assert "Age / Graduation Era Proxies" in result["categories"]


def test_evaluate_group_fairness_unavailable_data():
    candidate_scores = [
        {"candidate_id": "c1", "match_score": 85.0},
        {"candidate_id": "c2", "match_score": 65.0},
    ]
    # No demographic data provided
    result = fairness_service.evaluate_group_fairness(candidate_scores, demographic_data=None)

    assert result["evaluation_data_available"] is False
    assert result["status"] == "Demographic Evaluation Data Unavailable"
    assert len(result["warnings"]) > 0
    assert len(result["limitations"]) > 0


def test_evaluate_group_fairness_with_valid_data():
    candidate_scores = [
        {"candidate_id": "c1", "match_score": 85.0},
        {"candidate_id": "c2", "match_score": 80.0},
        {"candidate_id": "c3", "match_score": 75.0},
        {"candidate_id": "c4", "match_score": 60.0},
    ]
    demographic_data = {
        "c1": "Cohort A",
        "c2": "Cohort A",
        "c3": "Cohort B",
        "c4": "Cohort B",
    }
    # Threshold at 70.0: Cohort A has 2/2 = 1.0; Cohort B has 1/2 = 0.5
    result = fairness_service.evaluate_group_fairness(candidate_scores, demographic_data, selection_threshold=70.0)

    assert result["evaluation_data_available"] is True
    assert len(result["group_metrics"]) == 2
    assert result["demographic_parity_difference"] == 0.5
    assert result["disparate_impact_ratio"] == 0.5
    assert result["four_fifths_rule_passed"] is False  # 0.5 < 0.8
    assert any("Disparate impact ratio" in w for w in result["warnings"])
