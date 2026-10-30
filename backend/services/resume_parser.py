"""Resume Parsing Service (Person 1 Module / Adapter)

Extracts structured information from PDF, DOCX, and TXT resumes.
Includes text extraction, validation, cleaning, and heuristic skill/experience parsing.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple


# Comprehensive skill taxonomy for automated matching and candidate profiling
COMMON_SKILLS = {
    "python", "java", "c++", "c#", "javascript", "typescript", "html", "css",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "fastapi", "flask", "django", "react", "angular", "vue", "node.js",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd",
    "git", "github", "linux", "rest api", "graphql", "microservices",
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "shap",
    "data analysis", "data engineering", "spark", "airflow", "kafka",
    "prompt engineering", "statistics", "data science", "agile", "scrum"
}

EDUCATION_PATTERNS = [
    (r"\b(ph\.?d|doctor of philosophy|doctorate)\b", "PhD"),
    (r"\b(master(?:'?s)?|m\.?s|m\.?tech|m\.?sc|mba)\b", "Master's"),
    (r"\b(bachelor(?:'?s)?|b\.?s|b\.?tech|b\.?e|b\.?sc|ba)\b", "Bachelor's"),
    (r"\b(associate(?:'?s)?|diploma)\b", "Associate / Diploma"),
    (r"\b(high school|secondary)\b", "High School"),
]


class ResumeParserService:
    """Service to parse, sanitize, and extract structured candidate profiles."""

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Extract raw text from PDF, DOCX, or TXT file bytes."""
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise ValueError(f"File '{filename}' is empty.")

        ext = filename.lower().split(".")[-1]

        if ext == "txt":
            try:
                return file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return file_bytes.decode("latin-1", errors="replace")

        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                text_parts = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                extracted = "\n".join(text_parts).strip()
                if not extracted:
                    raise ValueError("PDF contains no selectable or extractable text (it might be scanned).")
                return extracted
            except Exception as e:
                raise ValueError(f"Failed to extract PDF '{filename}': {str(e)}")

        elif ext in ("docx", "doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(file_bytes))
                text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            text_parts.append(row_text)
                extracted = "\n".join(text_parts).strip()
                if not extracted:
                    raise ValueError("DOCX contains no extractable text.")
                return extracted
            except Exception as e:
                raise ValueError(f"Failed to extract DOCX '{filename}': {str(e)}")

        else:
            raise ValueError(f"Unsupported file format '.{ext}'. Supported: PDF, DOCX, TXT.")

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted resume text."""
        # Normalize non-breaking spaces and irregular whitespace
        cleaned = re.sub(r"[\r\t\f\v]", " ", text)
        cleaned = re.sub(r"[ \u00a0]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def extract_candidate_name(self, text: str, filename: str) -> str:
        """Extract or derive candidate name from resume header or fallback to filename."""
        first_lines = [line.strip() for line in text.split("\n")[:5] if line.strip()]
        for line in first_lines:
            # Check for name-like pattern (2-3 words, no symbols, reasonable length)
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$", line) and len(line) < 40:
                # Avoid headers like "Curriculum Vitae" or "Resume"
                if line.lower() not in {"curriculum vitae", "resume", "personal details", "profile summary"}:
                    return line

        # Fallback to sanitized filename
        base = re.sub(r"\.[a-zA-Z0-9]+$", "", filename)
        base = re.sub(r"[-_]+", " ", base)
        base = re.sub(r"(?i)resume|cv|candidate", "", base).strip()
        if base:
            return base.title()
        return "Candidate"

    def extract_skills(self, text: str) -> List[str]:
        """Extract recognized skills present in the resume text."""
        text_lower = text.lower()
        found_skills = set()

        for skill in COMMON_SKILLS:
            # Exact boundary matching
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
            if re.search(pattern, text_lower):
                found_skills.add(skill)

        return sorted(list(found_skills))

    def extract_experience_years(self, text: str) -> float:
        """Estimate total relevant years of professional experience from resume."""
        # 1. Look for explicit mentions like "5+ years of experience" or "7 years exp"
        explicit_matches = re.findall(
            r"(\b\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
            text,
            flags=re.IGNORECASE
        )
        if explicit_matches:
            try:
                return float(explicit_matches[0])
            except ValueError:
                pass

        # 2. Look for year ranges like (2018 - 2023) or (2020 - Present)
        year_matches = re.findall(
            r"\b(20\d{2}|19\d{2})\s*(?:-|to|–)\s*(20\d{2}|present|current)\b",
            text,
            flags=re.IGNORECASE
        )
        total_span = 0.0
        current_year = 2026

        for start_str, end_str in year_matches:
            try:
                start = int(start_str)
                end = current_year if end_str.lower() in ("present", "current") else int(end_str)
                if 1980 <= start <= current_year and start <= end <= current_year:
                    span = end - start
                    total_span = max(total_span, span)
            except ValueError:
                continue

        if total_span > 0:
            return float(min(total_span, 35.0))

        # Default fallback if not detectable
        return 1.0

    def extract_education(self, text: str) -> str:
        """Extract highest degree level detected."""
        text_lower = text.lower()
        for pattern, degree in EDUCATION_PATTERNS:
            if re.search(pattern, text_lower):
                return degree
        return "Bachelor's"  # Baseline standard assumption

    def parse_resume(self, file_bytes: bytes, filename: str, candidate_id: str) -> Dict[str, Any]:
        """Parse raw resume file into structured profile."""
        raw_text = self.extract_text_from_bytes(file_bytes, filename)
        cleaned_text = self.clean_text(raw_text)
        candidate_name = self.extract_candidate_name(cleaned_text, filename)
        skills = self.extract_skills(cleaned_text)
        exp_years = self.extract_experience_years(cleaned_text)
        education = self.extract_education(cleaned_text)

        return {
            "candidate_id": candidate_id,
            "filename": filename,
            "candidate_name": candidate_name,
            "raw_text": cleaned_text,
            "skills": skills,
            "experience_years": exp_years,
            "education": education,
            "word_count": len(cleaned_text.split()),
        }


# Singleton instance
resume_parser = ResumeParserService()
