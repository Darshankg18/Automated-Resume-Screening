"""Model and Ranking Evaluation Module (Phase 11 Module)

Calculates rigorous classification and ranking metrics on labeled test/benchmark datasets.
- Classification Metrics: Precision, Recall, F1-score, ROC-AUC
- Ranking Metrics: NDCG@K, Precision@K, Mean Reciprocal Rank (MRR)
- Explicitly distinguishes classification (binary decision) from ranking (ordering quality).
- Strictly displays "Evaluation data required" if no ground-truth labeled evaluation data is available.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class EvaluationService:
    """Service to evaluate classification and ranking performance."""

    def compute_metrics(
        self,
        candidate_eval_items: Optional[List[Dict[str, Any]]] = None,
        k: int = 5,
        threshold: float = 70.0,
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute classification and ranking metrics if ground truth is supplied.

        Each item in `candidate_eval_items` should contain:
        - "candidate_id": str
        - "predicted_score": float (0 - 100)
        - "actual_relevant": int / bool (1 if truly qualified / hired, 0 otherwise)
        - "actual_relevance_grade": Optional[int] (e.g. 0 to 3 scale for graded NDCG)

        If candidate_eval_items is None or empty, returns transparent
        "Evaluation data required" state.
        """
        if not candidate_eval_items or len(candidate_eval_items) < 2:
            return {
                "status": "Evaluation data required",
                "data_available": False,
                "dataset_name": dataset_name,
                "total_samples": 0,
                "classification_metrics": None,
                "ranking_metrics": None,
                "notes": [
                    "Evaluation metrics cannot be computed without a labeled ground-truth evaluation dataset.",
                    "In production or college demonstration mode, load the provided benchmark test dataset or upload ground-truth candidate hiring outcomes.",
                    "The system avoids fabricating or halluncinating synthetic evaluation metrics."
                ]
            }

        y_true = []
        y_scores = []
        y_pred = []
        grades = []

        for item in candidate_eval_items:
            actual = int(item.get("actual_relevant", 0))
            score = float(item.get("predicted_score", 0.0))
            grade = int(item.get("actual_relevance_grade", actual))

            y_true.append(actual)
            y_scores.append(score / 100.0)  # normalized probability
            y_pred.append(1 if score >= threshold else 0)
            grades.append(grade)

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_scores_np = np.array(y_scores)

        # 1. Classification Metrics
        prec = float(precision_score(y_true_np, y_pred_np, zero_division=0))
        rec = float(recall_score(y_true_np, y_pred_np, zero_division=0))
        f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))

        roc_auc = None
        # ROC-AUC requires at least one positive and one negative sample
        if len(np.unique(y_true_np)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_np, y_scores_np))
            except Exception:
                roc_auc = None

        classification_res = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "threshold": threshold,
        }

        # 2. Ranking Metrics
        # Sort items by predicted score descending
        sorted_indices = np.argsort(-y_scores_np)
        sorted_true = y_true_np[sorted_indices]
        sorted_grades = np.array(grades)[sorted_indices]

        effective_k = min(k, len(y_scores_np))

        # Precision@K
        p_at_k = float(np.mean(sorted_true[:effective_k]))

        # Mean Reciprocal Rank (MRR) - rank of first relevant item
        first_rel_rank = 0
        for rank_idx, val in enumerate(sorted_true, 1):
            if val == 1:
                first_rel_rank = rank_idx
                break
        mrr = float(1.0 / first_rel_rank) if first_rel_rank > 0 else 0.0

        # Normalized Discounted Cumulative Gain (NDCG@K)
        ndcg_k = self._calculate_ndcg_at_k(sorted_grades, effective_k)

        ranking_res = {
            "ndcg_at_k": round(ndcg_k, 4),
            "precision_at_k": round(p_at_k, 4),
            "mean_reciprocal_rank": round(mrr, 4),
            "k": effective_k,
        }

        return {
            "status": "Available",
            "data_available": True,
            "dataset_name": dataset_name or "Custom Evaluation Set",
            "total_samples": len(candidate_eval_items),
            "classification_metrics": classification_res,
            "ranking_metrics": ranking_res,
            "notes": [
                f"Classification evaluated at decision threshold of {threshold} score points.",
                f"Ranking metrics evaluated across Top-{effective_k} candidates.",
                "Precision@K measures the proportion of relevant candidates in the top recommendations.",
                "NDCG@K evaluates whether the most relevant candidates appear earliest in the ranked list."
            ]
        }

    def _calculate_ndcg_at_k(self, graded_relevance: np.ndarray, k: int) -> float:
        """Calculate NDCG@K given ranked relevance grades."""
        if k == 0 or len(graded_relevance) == 0:
            return 0.0

        top_k = graded_relevance[:k]

        # DCG@K: sum((2^rel - 1) / log2(i + 1))
        dcg = 0.0
        for i, rel in enumerate(top_k, 1):
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)

        # Ideal DCG@K
        ideal_relevance = np.sort(graded_relevance)[::-1][:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance, 1):
            idcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)

        if idcg == 0.0:
            return 0.0

        return float(dcg / idcg)


# Singleton instance
evaluation_service = EvaluationService()
