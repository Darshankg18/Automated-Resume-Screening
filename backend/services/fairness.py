"""Bias Mitigation and Fairness Service (Phase 9 Module)

Responsible AI module for resume screening:
- Enforces strict exclusion of protected sensitive attributes (gender, race, caste, religion, age).
- Masks demographic identifiers from resume text (names, gendered pronouns, honorifics).
- Detects proxy features that might indirectly leak demographic information.
- Calculates group fairness metrics (Demographic Parity, Selection Rate Disparity, Disparate Impact)
  ONLY when lawfully provided demographic evaluation data is present.
- Strictly refuses to infer protected demographic attributes from names, photos, or proxies.
- Explicitly documents architectural limitations of debiasing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# Regex patterns for identifying demographic proxies and indicators
GENDERED_PRONOUNS_MAP = {
    r"\bhe\b": "they",
    r"\bshe\b": "they",
    r"\bhim\b": "them",
    r"\bher\b": "them",
    r"\bhis\b": "their",
    r"\bhers\b": "theirs",
    r"\bhimself\b": "themselves",
    r"\bherself\b": "themselves",
}

HONORIFICS_REGEX = r"\b(Mr\.?|Mrs\.?|Ms\.?|Miss|Sir|Madam|Shri|Smt)\b"

SENSITIVE_KEYWORDS_REGEX = r"\b(male|female|gender|marital status|unmarried|married|single|religion|caste|race|ethnicity|date of birth|dob|father'?s name|mother'?s name)\b"

# Proxy categories that could inadvertently disclose demographic background
PROXY_PATTERNS = {
    "Affinity / Diversity Organizations": [
        r"\bwomen in (?:tech|stem|engineering|code|computing)\b",
        r"\bsociety of women engineers\b",
        r"\bgrace hopper\b",
        r"\bblack (?:in tech|girls code|engineers)\b",
        r"\bhispani[cx] in (?:tech|stem)\b",
        r"\blgbtq\+?(?:\s+(?:in tech|stem|group|alliance))?\b",
        r"\bqueer in (?:tech|stem)\b",
    ],
    "Age / Graduation Era Proxies": [
        r"\b(?:matriculated|graduated|class of)\s+(?:19[6-9]\d|200\d)\b",
        r"\b(?:high school|secondary education)\s+(?:19[6-9]\d|200\d)\b",
    ],
    "Religious / Cultural Student Groups": [
        r"\b(?:muslim|hindu|christian|jewish|sikh|buddhist)\s+(?:students?|association|fellowship|society)\b",
    ]
}


class FairnessService:
    """Fairness and Bias Mitigation Engine for Recruitment AI."""

    def __init__(self):
        self.disparate_impact_threshold = 0.80  # 80% / Four-Fifths Rule

    def mask_demographic_identifiers(self, text: str, candidate_name: Optional[str] = None) -> Tuple[str, List[str]]:
        """Remove or mask demographic and identity indicators from resume text.

        Returns masked text and a list of applied redactions.
        """
        masked = text
        applied_redactions = []

        # 1. Mask candidate name if provided
        if candidate_name and len(candidate_name.strip()) > 2 and candidate_name.lower() != "candidate":
            name_pattern = re.compile(re.escape(candidate_name), re.IGNORECASE)
            if name_pattern.search(masked):
                masked = name_pattern.sub("[CANDIDATE_NAME]", masked)
                applied_redactions.append("Redacted candidate name")

        # 2. Mask honorifics (Mr., Mrs., Ms., etc.)
        if re.search(HONORIFICS_REGEX, masked, re.IGNORECASE):
            masked = re.sub(HONORIFICS_REGEX, "[TITLE]", masked, flags=re.IGNORECASE)
            applied_redactions.append("Neutralized demographic honorifics")

        # 3. Replace gendered third-person pronouns with neutral pronouns
        replaced_pronouns = False
        for pattern, replacement in GENDERED_PRONOUNS_MAP.items():
            if re.search(pattern, masked, re.IGNORECASE):
                masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
                replaced_pronouns = True
        if replaced_pronouns:
            applied_redactions.append("Normalized gendered third-person pronouns to gender-neutral equivalents")

        # 4. Mask sensitive biographical headings (DOB, Marital Status, Caste, Religion)
        if re.search(SENSITIVE_KEYWORDS_REGEX, masked, re.IGNORECASE):
            masked = re.sub(SENSITIVE_KEYWORDS_REGEX, "[SENSITIVE_FIELD]", masked, flags=re.IGNORECASE)
            applied_redactions.append("Masked sensitive demographic fields (DOB, marital status, etc.)")

        return masked, applied_redactions

    def detect_proxy_features(self, text: str) -> Dict[str, Any]:
        """Audit text for potential proxy features that may indirectly reveal demographic characteristics.

        NOTE: Identified proxies are flagged for human audit. Candidates are NOT rejected
        based on detected proxies.
        """
        findings = {}
        total_flags = 0

        for category, patterns in PROXY_PATTERNS.items():
            matches_found = []
            for pat in patterns:
                matches = re.findall(pat, text, flags=re.IGNORECASE)
                if matches:
                    matches_found.extend(matches)
            if matches_found:
                findings[category] = list(set(matches_found))
                total_flags += len(findings[category])

        return {
            "potential_proxies_detected": total_flags > 0,
            "total_proxy_flags": total_flags,
            "categories": findings,
            "guidance": (
                "Proxies flagged above indicate extracurricular or temporal clues that could "
                "indirectly disclose demographic background. Features used for scoring do NOT "
                "include these proxy mentions."
            ),
        }

    def evaluate_group_fairness(
        self,
        candidate_scores: List[Dict[str, Any]],
        demographic_data: Optional[Dict[str, str]] = None,
        selection_threshold: float = 70.0,
    ) -> Dict[str, Any]:
        """Compute group fairness metrics when lawfully provided demographic evaluation data exists.

        candidate_scores: list of dicts with {"candidate_id": str, "match_score": float}
        demographic_data: mapping of candidate_id -> group_label (e.g., {"cand_1": "Group A", "cand_2": "Group B"})

        If demographic_data is None or empty, returns transparent 'Evaluation data unavailable' state.
        DOES NOT fabricate or simulate demographic data without user consent.
        """
        if not demographic_data or not candidate_scores:
            return {
                "status": "Demographic Evaluation Data Unavailable",
                "evaluation_data_available": False,
                "selected_threshold": selection_threshold,
                "group_metrics": [],
                "demographic_parity_difference": None,
                "disparate_impact_ratio": None,
                "four_fifths_rule_passed": None,
                "warnings": [
                    "Group fairness metrics cannot be calculated because demographic evaluation data was not provided.",
                    "Candidate demographics are never inferred from names, photos, or text proxies to avoid privacy violations and algorithmic misgendering.",
                ],
                "limitations": self.get_fairness_limitations(),
                "note": (
                    "Fairness checks require lawful, consented demographic labels for evaluation purposes. "
                    "In operational screening, only job-relevant qualifications are extracted and evaluated."
                ),
            }

        # Calculate selection rates by group
        group_counts: Dict[str, int] = {}
        group_selected: Dict[str, int] = {}

        for cand in candidate_scores:
            cid = cand.get("candidate_id")
            score = cand.get("match_score", 0.0)
            group = demographic_data.get(cid)

            if not group:
                continue

            group_counts[group] = group_counts.get(group, 0) + 1
            if score >= selection_threshold:
                group_selected[group] = group_selected.get(group, 0) + 1

        if len(group_counts) < 2:
            return {
                "status": "Insufficient Demographic Diversity for Comparison",
                "evaluation_data_available": False,
                "selected_threshold": selection_threshold,
                "group_metrics": [],
                "demographic_parity_difference": None,
                "disparate_impact_ratio": None,
                "four_fifths_rule_passed": None,
                "warnings": ["Evaluation data must contain at least 2 distinct groups to compute group parity metrics."],
                "limitations": self.get_fairness_limitations(),
                "note": "At least two distinct demographic cohorts are required for parity contrast.",
            }

        group_metrics = []
        rates = []
        for grp, total in group_counts.items():
            sel = group_selected.get(grp, 0)
            rate = sel / total if total > 0 else 0.0
            rates.append(rate)
            group_metrics.append({
                "group_name": grp,
                "total_applicants": total,
                "selected_count": sel,
                "selection_rate": round(rate, 4),
            })

        max_rate = max(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0

        parity_diff = round(max_rate - min_rate, 4)
        disparate_impact = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
        four_fifths_passed = disparate_impact >= self.disparate_impact_threshold

        warnings = []
        if not four_fifths_passed:
            warnings.append(
                f"Disparate impact ratio ({disparate_impact:.2f}) falls below the 80% (Four-Fifths) regulatory benchmark. "
                "Review selection criteria and job requirement weights."
            )
        if parity_diff > 0.15:
            warnings.append(
                f"Demographic parity difference ({parity_diff:.2f}) exceeds 15% between highest and lowest group selection rates."
            )

        return {
            "status": "Demographic Evaluation Completed",
            "evaluation_data_available": True,
            "selected_threshold": selection_threshold,
            "group_metrics": group_metrics,
            "demographic_parity_difference": parity_diff,
            "disparate_impact_ratio": disparate_impact,
            "four_fifths_rule_passed": four_fifths_passed,
            "warnings": warnings,
            "limitations": self.get_fairness_limitations(),
            "note": "Calculated using validated evaluation dataset with explicit demographic labels.",
        }

    def get_fairness_limitations(self) -> List[str]:
        """Documentation of systematic AI recruiting limitations."""
        return [
            "Fairness through Unawareness Limitation: Removing demographic identifiers (names, gender) does not eliminate bias because remaining features (e.g. university pedigree, hobby groups, phrasing style) can act as subtle proxies.",
            "Historical Data Bias: Natural language models and skill taxonomies reflect historical industry hiring patterns, which may perpetuate under-representation in technical disciplines.",
            "No Autonomous Hiring: Match scores reflect keyword and qualification alignment against a specific job description, not innate human merit or holistic potential.",
            "Contextual Diversity: True inclusion requires active sourcing, holistic interview evaluations, and multi-faceted assessment beyond resume text."
        ]


# Singleton instance
fairness_service = FairnessService()
