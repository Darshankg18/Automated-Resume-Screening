"""Tests for resume parsing and text extraction (Person 1 Module)."""

import io
import pytest
from backend.services.resume_parser import resume_parser, ResumeParserService


def test_clean_text():
    raw = "Hello   world!\r\n\r\n\tThis   is a  test.\n\n\n\nNext line."
    cleaned = resume_parser.clean_text(raw)
    assert "Hello world!" in cleaned
    assert "This is a test." in cleaned
    assert "\t" not in cleaned


def test_extract_skills():
    text = "Candidate has 5 years of Python, SQL, Docker, FastAPI, and AWS experience."
    skills = resume_parser.extract_skills(text)
    assert "python" in skills
    assert "sql" in skills
    assert "docker" in skills
    assert "fastapi" in skills
    assert "aws" in skills


def test_extract_experience_years_explicit():
    text = "Senior Software Engineer with 7+ years of experience in backend systems."
    years = resume_parser.extract_experience_years(text)
    assert years == 7.0


def test_extract_experience_years_range():
    text = "Backend Developer at Acme Corp (2019 - 2024)."
    years = resume_parser.extract_experience_years(text)
    assert years >= 5.0


def test_extract_education():
    phd_text = "Doctor of Philosophy in Computer Science"
    assert resume_parser.extract_education(phd_text) == "PhD"

    ms_text = "Master of Science in Software Engineering"
    assert resume_parser.extract_education(ms_text) == "Master's"

    bs_text = "Bachelor of Technology in Information Technology"
    assert resume_parser.extract_education(bs_text) == "Bachelor's"


def test_parse_txt_resume():
    content = (
        "Alice Smith\n"
        "Senior Backend Engineer with 5 years of experience.\n"
        "Proficient in Python, SQL, Docker, and AWS.\n"
        "Holds a Master of Science in Computer Science."
    ).encode("utf-8")

    parsed = resume_parser.parse_resume(content, "alice_resume.txt", "cand_01")
    assert parsed["candidate_id"] == "cand_01"
    assert parsed["candidate_name"] == "Alice Smith"
    assert "python" in parsed["skills"]
    assert "sql" in parsed["skills"]
    assert parsed["experience_years"] == 5.0
    assert parsed["education"] == "Master's"


def test_parse_empty_resume_raises_error():
    with pytest.raises(ValueError, match="is empty"):
        resume_parser.parse_resume(b"", "empty.txt", "cand_02")


def test_parse_unsupported_format_raises_error():
    with pytest.raises(ValueError, match="Unsupported file format"):
        resume_parser.parse_resume(b"sample bytes", "resume.png", "cand_03")


def test_parse_corrupted_pdf_raises_error():
    corrupted_pdf_bytes = b"%PDF-1.4\ncorrupted bytes that are not a valid pdf stream"
    with pytest.raises(ValueError, match="Failed to extract PDF"):
        resume_parser.parse_resume(corrupted_pdf_bytes, "corrupted.pdf", "cand_04")
