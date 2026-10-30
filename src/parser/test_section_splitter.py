from src.parser. section_splitter import detect_section_heading, split_sections


# Test different real-world resume headings
def test_heading_detection():

    test_cases = {
        "EDUCATION": "education",
        "Education:": "education",
        "1. EDUCATION": "education",

        "TECHNICAL SKILLS": "skills",
        "Core Skills": "skills",
        "Technology Stack": "skills",

        "WORK EXPERIENCE": "experience",
        "Professional Experience": "experience",
        "Work History": "experience",

        "PROJECTS": "projects",
        "Academic Projects": "projects",

        "CERTIFICATES": "certifications",
        "Professional Certifications": "certifications",

        "ACHIEVEMENTS": "achievements",
        "AWARDS": "achievements",

        "HOBBIES": "interests",
        "INTERESTS": "interests",

        "CAREER OBJECTIVE": "summary",
        "Professional Summary": "summary",
    }

    for heading, expected_section in test_cases.items():

        result = detect_section_heading(heading)

        assert result == expected_section, (
            f"Failed: {heading} "
            f"Expected={expected_section}, Got={result}"
        )

    print("All heading detection tests passed!")


# Test complete section splitting
def test_section_splitting():

    resume_text = """
    JOHN SMITH

    CAREER OBJECTIVE
    To obtain a software developer position.

    EDUCATION
    Bachelor of Engineering in Computer Science

    TECHNICAL SKILLS
    Java
    Python
    SQL

    WORK EXPERIENCE
    Software Developer Intern
    ABC Technologies

    ACADEMIC PROJECTS
    Student Management System

    CERTIFICATES
    Java Programming Certificate

    HOBBIES
    Reading
    Cricket
    """

    sections = split_sections(resume_text)

    expected_sections = [
        "summary",
        "education",
        "skills",
        "experience",
        "projects",
        "certifications",
        "interests",
    ]

    for section in expected_sections:
        assert section in sections, f"Missing section: {section}"

    print("Complete section splitting test passed!")


# Run the tests
if __name__ == "__main__":

    test_heading_detection()
    test_section_splitting()

    print("All section splitter tests passed!")