"""Ranking and Feature Engineering Service (Person 2 & 3 Modules / Adapters)

Handles:
- Job description parsing and required skill extraction
- Feature engineering on job-relevant dimensions
- ML scoring model (Random Forest Regressor / Ensemble) for predicting match score (0-100)
- Fully transparent and compatible with SHAP TreeExplainer
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.resume_parser import COMMON_SKILLS, EDUCATION_PATTERNS


FEATURE_NAMES = [
    "skills_match_ratio",
    "missing_skills_count",
    "experience_alignment",
    "education_score",
    "keyword_similarity",
    "tfidf_similarity"
]

FEATURE_DISPLAY_NAMES = {
    "skills_match_ratio": "Required Skills Match Ratio",
    "missing_skills_count": "Missing Required Skills Count",
    "experience_alignment": "Relevant Experience Alignment",
    "education_score": "Education Level Relevance",
    "keyword_similarity": "Keyword & Vocabulary Overlap",
    "tfidf_similarity": "Semantic TF-IDF Similarity"
}

EDUCATION_WEIGHTS = {
    "PhD": 1.0,
    "Master's": 0.9,
    "Bachelor's": 0.75,
    "Associate / Diploma": 0.5,
    "High School": 0.3
}


class RankingService:
    """Service to extract features and compute candidate ranking scores."""

    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self._initialize_or_train_scoring_model()

    def _initialize_or_train_scoring_model(self):
        """Train and calibrate an interpretable Random Forest scoring model.

        The model is trained on a synthetic recruitment manifold representing varying
        degrees of candidate qualification to ensure monotonic, realistic, and stable behavior.
        """
        np.random.seed(42)
        n_samples = 400

        # Synthetic feature generation for calibration
        skills_ratio = np.random.uniform(0.1, 1.0, n_samples)
        missing_count = np.clip(np.random.poisson(lam=(1 - skills_ratio) * 6), 0, 10)
        exp_alignment = np.random.uniform(0.2, 1.0, n_samples)
        edu_score = np.random.choice([0.3, 0.5, 0.75, 0.9, 1.0], size=n_samples)
        keyword_sim = np.clip(skills_ratio * 0.7 + np.random.normal(0, 0.05, n_samples), 0.05, 0.95)
        tfidf_sim = np.clip(skills_ratio * 0.65 + exp_alignment * 0.2 + np.random.normal(0, 0.05, n_samples), 0.05, 0.95)

        X = np.column_stack([
            skills_ratio,
            missing_count,
            exp_alignment,
            edu_score,
            keyword_sim,
            tfidf_sim
        ])

        # Ground-truth heuristic function for candidate fit (scaled to 15 - 98)
        y = (
            15.0  # Base intercept
            + 40.0 * skills_ratio
            - 1.0 * np.minimum(missing_count, 5.0)
            + 25.0 * exp_alignment
            + 10.0 * edu_score
            + 8.0 * keyword_sim
            + 8.0 * tfidf_sim
        )
        y = np.clip(y, 10.0, 98.0)

        rf = RandomForestRegressor(n_estimators=40, max_depth=5, random_state=42)
        rf.fit(X, y)
        self.model = rf

    def parse_job_description(self, jd_text: str, jd_id: str = "jd_default") -> Dict[str, Any]:
        """Parse job description to extract required skills, experience expectations, and keywords."""
        jd_lower = jd_text.lower()

        # Extract required skills from known taxonomy
        required_skills = []
        for skill in COMMON_SKILLS:
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
            if re.search(pattern, jd_lower):
                required_skills.append(skill)

        # Infer required years of experience (e.g. "5+ years", "3-5 years")
        exp_match = re.search(r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience", jd_lower)
        expected_years = float(exp_match.group(1)) if exp_match else 3.0

        # Infer target job title from first 3 lines
        lines = [line.strip() for line in jd_text.split("\n") if line.strip()]
        title = lines[0] if lines else "Software / Tech Role"
        if len(title) > 60:
            title = title[:60] + "..."

        return {
            "jd_id": jd_id,
            "title": title,
            "required_skills": sorted(list(set(required_skills))),
            "expected_years": expected_years,
            "raw_text": jd_text,
            "word_count": len(jd_text.split()),
        }

    def compute_features(
        self,
        candidate_profile: Dict[str, Any],
        jd_profile: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate the 6 job-relevant features comparing candidate against the job description."""
        cand_skills = set(candidate_profile.get("skills", []))
        jd_skills = set(jd_profile.get("required_skills", []))

        # 1. Skills match ratio
        if len(jd_skills) > 0:
            matched_skills = sorted(list(cand_skills.intersection(jd_skills)))
            missing_skills = sorted(list(jd_skills.difference(cand_skills)))
            skills_match_ratio = float(len(matched_skills) / len(jd_skills))
        else:
            # Fallback if no specific skills detected in JD
            matched_skills = sorted(list(cand_skills))
            missing_skills = []
            skills_match_ratio = 0.5

        # 2. Missing skills count
        missing_skills_count = float(len(missing_skills))

        # 3. Experience alignment
        cand_exp = float(candidate_profile.get("experience_years", 1.0))
        expected_exp = float(jd_profile.get("expected_years", 3.0))
        if expected_exp > 0:
            exp_ratio = cand_exp / expected_exp
            # S-curve / capped alignment (1.0 = fully qualified)
            experience_alignment = float(min(1.0, max(0.2, exp_ratio)))
        else:
            experience_alignment = 0.8

        # 4. Education relevance
        cand_edu = candidate_profile.get("education", "Bachelor's")
        education_score = EDUCATION_WEIGHTS.get(cand_edu, 0.7)

        # 5. Keyword similarity (Jaccard on words, normalized by realistic text length divergence)
        cand_words = set(re.findall(r"\b\w{3,}\b", candidate_profile.get("raw_text", "").lower()))
        jd_words = set(re.findall(r"\b\w{3,}\b", jd_profile.get("raw_text", "").lower()))
        inter = cand_words.intersection(jd_words)
        union = cand_words.union(jd_words)
        raw_jaccard = float(len(inter) / len(union)) if union else 0.05
        # In resume screening, Jaccard of 0.25+ is exceptionally high lexical alignment
        keyword_similarity = float(np.clip(raw_jaccard / 0.30, 0.05, 1.0))

        # 6. TF-IDF Cosine Similarity (normalized)
        try:
            corpus = [jd_profile.get("raw_text", ""), candidate_profile.get("raw_text", "")]
            tfidf_mat = self.tfidf_vectorizer.fit_transform(corpus)
            sim_matrix = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])
            raw_tfidf = float(sim_matrix[0][0])
            # Cosine similarity of 0.35+ is strong topical alignment
            tfidf_similarity = float(np.clip(raw_tfidf / 0.40, 0.05, 1.0))
        except Exception:
            tfidf_similarity = keyword_similarity

        feature_vector = np.array([
            skills_match_ratio,
            missing_skills_count,
            experience_alignment,
            education_score,
            keyword_similarity,
            tfidf_similarity
        ], dtype=float)

        meta = {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skills_match_ratio": round(skills_match_ratio, 3),
            "missing_skills_count": int(missing_skills_count),
            "experience_alignment": round(experience_alignment, 3),
            "education_score": round(education_score, 3),
            "keyword_similarity": round(keyword_similarity, 3),
            "tfidf_similarity": round(tfidf_similarity, 3),
            "candidate_exp": cand_exp,
            "expected_exp": expected_exp,
            "education_level": cand_edu,
        }

        return feature_vector, meta

    def score_candidate(
        self,
        candidate_profile: Dict[str, Any],
        jd_profile: Dict[str, Any]
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """Calculate overall match score (0-100) using the trained scoring model."""
        features, meta = self.compute_features(candidate_profile, jd_profile)
        X = features.reshape(1, -1)

        raw_score = float(self.model.predict(X)[0])
        match_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)

        return match_score, features, meta

    def rank_candidates(
        self,
        candidate_profiles: List[Dict[str, Any]],
        jd_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Score and sort all candidates descending by match score."""
        scored_candidates = []

        for cand in candidate_profiles:
            score, feat_vec, meta = self.score_candidate(cand, jd_profile)
            scored_candidates.append({
                "candidate_id": cand["candidate_id"],
                "candidate_name": cand.get("candidate_name", "Candidate"),
                "match_score": score,
                "matched_skills": meta["matched_skills"],
                "missing_skills": meta["missing_skills"],
                "experience_years": cand.get("experience_years", 1.0),
                "education_level": cand.get("education", "Bachelor's"),
                "feature_vector": feat_vec,
                "meta": meta,
                "explanation_available": True,
                "summary": (
                    f"Matched {len(meta['matched_skills'])} skills with "
                    f"{cand.get('experience_years', 1.0)} yrs experience."
                )
            })

        # Rank descending
        scored_candidates.sort(key=lambda x: x["match_score"], reverse=True)

        # Assign ranks
        for idx, item in enumerate(scored_candidates, 1):
            item["rank"] = idx

        return scored_candidates


# Singleton instance
ranking_service = RankingService()
