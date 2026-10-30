"""Pydantic schemas for request validation and response models in the Resume Screening API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current server time (ISO format)")
    shap_ready: bool = Field(..., description="Whether SHAP explainability is loaded and operational")
    active_session_candidates: int = Field(default=0, description="Number of candidates currently held in session")
    active_jd_present: bool = Field(default=False, description="Whether a job description is active in session")


class JobDescriptionUploadResponse(BaseModel):
    message: str
    jd_id: str
    title: str
    word_count: int
    required_skills: List[str]
    preview: str


class ResumeUploadItem(BaseModel):
    filename: str
    candidate_id: str
    candidate_name: str
    status: str
    error: Optional[str] = None
    extracted_skills: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = None


class ResumeUploadResponse(BaseModel):
    message: str
    total_uploaded: int
    successful: int
    failed: int
    items: List[ResumeUploadItem]


class CandidateRankItem(BaseModel):
    rank: int
    candidate_id: str
    candidate_name: str
    match_score: float = Field(..., description="Match score out of 100")
    matched_skills: List[str]
    missing_skills: List[str]
    experience_years: float
    education_level: str
    explanation_available: bool = True
    summary: str


class CandidateRankingResponse(BaseModel):
    jd_title: str
    total_candidates: int
    rankings: List[CandidateRankItem]
    timestamp: str


class FeatureContributionItem(BaseModel):
    feature_name: str
    display_name: str
    value: float
    contribution: float
    impact: str  # "positive", "negative", "neutral"
    description: str


class CandidateExplanationResponse(BaseModel):
    candidate_id: str
    candidate_name: str
    match_score: float
    base_score: float
    is_shap_explanation: bool
    explanation_method: str
    matched_skills: List[str]
    missing_skills: List[str]
    evidence: Dict[str, Any]
    feature_contributions: List[FeatureContributionItem]
    positive_contributions: List[str]
    negative_contributions: List[str]
    human_readable_summary: List[str]
    chart_data: Dict[str, Any]


class DemographicParityMetric(BaseModel):
    group_name: str
    total_applicants: int
    selected_count: int
    selection_rate: float


class FairnessReportResponse(BaseModel):
    status: str
    evaluation_data_available: bool
    selected_threshold: float
    group_metrics: List[DemographicParityMetric] = Field(default_factory=list)
    demographic_parity_difference: Optional[float] = None
    disparate_impact_ratio: Optional[float] = None
    four_fifths_rule_passed: Optional[bool] = None
    proxy_detection_summary: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    note: str


class ClassificationMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    threshold: float


class RankingMetrics(BaseModel):
    ndcg_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    k: int


class EvaluationMetricsResponse(BaseModel):
    status: str
    data_available: bool
    classification_metrics: Optional[ClassificationMetrics] = None
    ranking_metrics: Optional[RankingMetrics] = None
    dataset_name: Optional[str] = None
    total_samples: int = 0
    notes: List[str] = Field(default_factory=list)


class EvaluationUploadRequest(BaseModel):
    dataset_name: Optional[str] = "benchmark_eval"
    use_sample_benchmark: bool = True
