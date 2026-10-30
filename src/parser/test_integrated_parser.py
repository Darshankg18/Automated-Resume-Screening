import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.parser.resume_parser import parse_resume


def test_integrated_resume_parser():

    docx_path = (
        project_root
        / "data"
        / "resumes"
        / "real_world_resume.docx"
    )

    resume_data = parse_resume(docx_path)

    # Check top-level structure
    assert "name" in resume_data
    assert "contact" in resume_data
    assert "sections" in resume_data

    # Check candidate name
    assert resume_data["name"] == "JANE DOE"

    # Check contact information
    assert resume_data["contact"]["email"] == "jane.doe@example.com"
    assert resume_data["contact"]["phone"] == "9876543210"

    # Check expected sections
    sections = resume_data["sections"]

    expected_sections = [
        "summary",
        "education",
        "skills",
        "experience",
        "projects",
        "certifications",
        "achievements",
        "interests",
    ]

    for section in expected_sections:
        assert section in sections, (
            f"Missing section: {section}"
        )

    print("Integrated resume parser test passed!")

    print("\nCandidate:")
    print(resume_data["name"])

    print("\nContact:")
    print(resume_data["contact"])

    print("\nSections:")
    for section in sections:
        print("-", section)


if __name__ == "__main__":
    test_integrated_resume_parser()
    print("\nAll integrated parser tests passed!")