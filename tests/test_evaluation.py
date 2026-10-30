"""Tests for Model and Ranking Evaluation module (Phase 11 Module)."""

import pytest
from backend.services.evaluation import evaluation_service


def test_missing_evaluation_data_guardrail():
    # Calling compute_metrics with None or empty list
    res = evaluation_service.compute_metrics(candidate_eval_items=None)

    assert res["status"] == "Evaluation data required"
    assert res["data_available"] is False
    assert res["classification_metrics"] is None
    assert res["ranking_metrics"] is None
    assert len(res["notes"]) > 0


def test_compute_classification_and_ranking_metrics():
    # 6 candidate evaluation records with ground truth
    items = [
        {"candidate_id": "c1", "predicted_score": 90.0, "actual_relevant": 1, "actual_relevance_grade": 3},
        {"candidate_id": "c2", "predicted_score": 85.0, "actual_relevant": 1, "actual_relevance_grade": 3},
        {"candidate_id": "c3", "predicted_score": 75.0, "actual_relevant": 1, "actual_relevance_grade": 2},
        {"candidate_id": "c4", "predicted_score": 72.0, "actual_relevant": 0, "actual_relevance_grade": 1},
        {"candidate_id": "c5", "predicted_score": 60.0, "actual_relevant": 0, "actual_relevance_grade": 0},
        {"candidate_id": "c6", "predicted_score": 40.0, "actual_relevant": 0, "actual_relevance_grade": 0},
    ]

    res = evaluation_service.compute_metrics(items, k=3, threshold=70.0, dataset_name="Test Set")

    assert res["status"] == "Available"
    assert res["data_available"] is True
    assert res["total_samples"] == 6

    # Classification Metrics Check
    cls = res["classification_metrics"]
    assert cls["precision"] > 0.0
    assert cls["recall"] > 0.0
    assert cls["f1_score"] > 0.0
    assert cls["roc_auc"] is not None
    assert 0.0 <= cls["roc_auc"] <= 1.0

    # Ranking Metrics Check
    rnk = res["ranking_metrics"]
    assert rnk["precision_at_k"] == 1.0  # Top 3 are all actual_relevant=1
    assert rnk["mean_reciprocal_rank"] == 1.0  # 1st item is relevant -> 1/1 = 1.0
    assert 0.0 <= rnk["ndcg_at_k"] <= 1.0
