"""Explainable AI Service Using SHAP & Transparent Fallback (Phase 8 Module)

Calculates feature contributions to explain why a candidate received a particular
resume-job match score.
- Uses SHAP TreeExplainer when available.
- If SHAP is unavailable, implements transparent baseline feature attribution
  and clearly labels it as a fallback rather than claiming genuine SHAP output.
- Generates human-readable explanations and chart structures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.services.ranking_service import (
    FEATURE_NAMES,
    FEATURE_DISPLAY_NAMES,
    ranking_service,
)

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """Service to generate Explainable AI summaries using SHAP and transparent fallbacks."""

    def __init__(self):
        self.shap_explainer = None
        self.shap_available = False
        self._initialize_shap()

    def _initialize_shap(self):
        """Attempt to instantiate SHAP TreeExplainer with the trained model."""
        try:
            import shap
            if ranking_service.model is not None:
                self.shap_explainer = shap.TreeExplainer(ranking_service.model)
                self.shap_available = True
                logger.info("SHAP TreeExplainer successfully initialized.")
        except Exception as e:
            self.shap_available = False
            self.shap_explainer = None
            logger.warning(f"SHAP initialization deferred or unavailable: {e}")

    def explain_candidate(
        self,
        candidate_profile: Dict[str, Any],
        jd_profile: Dict[str, Any],
        force_fallback: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive explanation for a single candidate match score.

        Returns match score, base score, feature contributions, positive/negative drivers,
        natural language human-readable explanation, and visualization payload.
        """
        match_score, feat_vec, meta = ranking_service.score_candidate(candidate_profile, jd_profile)
        X = feat_vec.reshape(1, -1)

        is_shap = False
        method_name = ""
        base_value = 50.0
        shap_values_arr = None

        # 1. Attempt SHAP TreeExplainer
        if self.shap_available and self.shap_explainer is not None and not force_fallback:
            try:
                shap_res = self.shap_explainer(X)
                # TreeExplainer expected value
                if hasattr(self.shap_explainer, "expected_value"):
                    ev = self.shap_explainer.expected_value
                    base_value = float(ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
                else:
                    base_value = float(shap_res.base_values[0])

                shap_values_arr = shap_res.values[0]
                is_shap = True
                method_name = "SHAP TreeExplainer (Genuine Shapley Values)"
            except Exception as e:
                logger.warning(f"SHAP computation encountered error, using transparent fallback: {e}")
                is_shap = False

        # 2. Transparent Fallback Attribution (if SHAP unavailable or failed)
        if not is_shap:
            method_name = "Transparent Feature Attribution (Fallback — Not Genuine SHAP)"
            base_value = 50.0  # Population median baseline
            # Feature baselines
            baselines = np.array([0.5, 3.0, 0.6, 0.75, 0.25, 0.3])
            weights = np.array([30.0, -3.0, 22.0, 8.0, 15.0, 15.0])
            deviations = feat_vec - baselines
            raw_contribs = deviations * weights

            # Calibrate sum of contributions to match (score - base_value)
            target_diff = match_score - base_value
            sum_raw = np.sum(np.abs(raw_contribs))
            if sum_raw > 0:
                shap_values_arr = raw_contribs * (target_diff / np.sum(raw_contribs)) if np.sum(raw_contribs) != 0 else raw_contribs
            else:
                shap_values_arr = np.zeros_like(feat_vec)

        # 3. Assemble Feature Contribution Items
        contributions_list = []
        pos_drivers = []
        neg_drivers = []

        for idx, fname in enumerate(FEATURE_NAMES):
            display = FEATURE_DISPLAY_NAMES.get(fname, fname)
            val = float(feat_vec[idx])
            phi = float(shap_values_arr[idx])
            impact = "positive" if phi > 0.4 else ("negative" if phi < -0.4 else "neutral")

            desc = self._get_feature_description(fname, val, phi, meta)

            contributions_list.append({
                "feature_name": fname,
                "display_name": display,
                "value": round(val, 3),
                "contribution": round(phi, 2),
                "impact": impact,
                "description": desc,
            })

            if phi > 1.0:
                pos_drivers.append(f"+{phi:.1f} pts from {display} ({desc})")
            elif phi < -1.0:
                neg_drivers.append(f"{phi:.1f} pts from {display} ({desc})")

        # 4. Generate Human-Readable Natural Language Summary
        summary_bullets = self._generate_human_readable_summary(
            candidate_profile, jd_profile, match_score, meta, pos_drivers, neg_drivers
        )

        # 5. Chart Payload for Waterfall and Horizontal Bar Visualizations
        chart_data = {
            "features": [c["display_name"] for c in contributions_list],
            "values": [c["value"] for c in contributions_list],
            "contributions": [c["contribution"] for c in contributions_list],
            "base_score": round(base_value, 1),
            "final_score": round(match_score, 1),
            "method": method_name,
        }

        # Candidate evidence snippets from resume text
        evidence = {
            "matched_skills_count": len(meta["matched_skills"]),
            "missing_skills_count": len(meta["missing_skills"]),
            "candidate_experience_years": candidate_profile.get("experience_years", 1.0),
            "job_expected_years": jd_profile.get("expected_years", 3.0),
            "education": candidate_profile.get("education", "Bachelor's"),
            "word_count": candidate_profile.get("word_count", 0),
        }

        return {
            "candidate_id": candidate_profile["candidate_id"],
            "candidate_name": candidate_profile.get("candidate_name", "Candidate"),
            "match_score": match_score,
            "base_score": round(base_value, 1),
            "is_shap_explanation": is_shap,
            "explanation_method": method_name,
            "matched_skills": meta["matched_skills"],
            "missing_skills": meta["missing_skills"],
            "evidence": evidence,
            "feature_contributions": contributions_list,
            "positive_contributions": pos_drivers,
            "negative_contributions": neg_drivers,
            "human_readable_summary": summary_bullets,
            "chart_data": chart_data,
        }

    def _get_feature_description(
        self,
        feature_name: str,
        value: float,
        phi: float,
        meta: Dict[str, Any]
    ) -> str:
        """Create intuitive explanation snippet for each feature."""
        if feature_name == "skills_match_ratio":
            matched_str = ", ".join(meta["matched_skills"][:4]) if meta["matched_skills"] else "None"
            return f"Found {len(meta['matched_skills'])} matched skills: {matched_str}"
        elif feature_name == "missing_skills_count":
            if meta["missing_skills"]:
                missing_str = ", ".join(meta["missing_skills"][:3])
                return f"Missing {len(meta['missing_skills'])} required skills: {missing_str}"
            return "All required skills present"
        elif feature_name == "experience_alignment":
            return f"{meta['candidate_exp']} yrs observed vs {meta['expected_exp']} yrs expected"
        elif feature_name == "education_score":
            return f"Holds {meta['education_level']} degree"
        elif feature_name == "keyword_similarity":
            return f"Vocabulary overlap index: {value:.2f}"
        elif feature_name == "tfidf_similarity":
            return f"TF-IDF cosine similarity index: {value:.2f}"
        return f"Observed value: {value:.2f}"

    def _generate_human_readable_summary(
        self,
        candidate_profile: Dict[str, Any],
        jd_profile: Dict[str, Any],
        score: float,
        meta: Dict[str, Any],
        pos_drivers: List[str],
        neg_drivers: List[str]
    ) -> List[str]:
        """Synthesize natural, professional recruitment bullet points."""
        bullets = []

        # 1. Strong match skills
        if meta["matched_skills"]:
            top_skills = ", ".join(meta["matched_skills"][:5])
            bullets.append(f"Strong match: Core competencies in {top_skills} verified in resume.")
        else:
            bullets.append("Skill mismatch: No explicit required skills identified from job description.")

        # 2. Experience contribution
        cand_exp = candidate_profile.get("experience_years", 1.0)
        expected_exp = jd_profile.get("expected_years", 3.0)
        if cand_exp >= expected_exp:
            bullets.append(f"Positive contribution: Relevant work experience ({cand_exp} years) meets or exceeds requirements ({expected_exp} years).")
        else:
            bullets.append(f"Growth area: Relevant experience ({cand_exp} years) is below target seniority ({expected_exp} years).")

        # 3. Missing requirements
        if meta["missing_skills"]:
            missing_top = ", ".join(meta["missing_skills"][:3])
            bullets.append(f"Missing requirement: Key skill gaps noted in {missing_top}.")
        else:
            bullets.append("Comprehensive coverage: Full alignment on all listed required skills.")

        # 4. Overall score
        bullets.append(f"Overall match score: {score:.1f}/100.")

        return bullets


# Singleton instance
explainability_service = ExplainabilityService()
