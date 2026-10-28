from pathlib import Path

from src.parser.resume_parser import parse_resume


def test_pdf_resume():

    pdf_path = "data/resumes/sample_resume.pdf"

    resume_data = parse_resume(pdf_path)

    assert "name" in resume_data
    assert "contact" in resume_data
    assert "sections" in resume_data

    sections = resume_data["sections"]

    assert "education" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "projects" in sections
    assert "certifications" in sections

    print("PDF resume test passed!")


def test_docx_resume():

    docx_path = "data/resumes/sample_resume.docx"

    resume_data = parse_resume(docx_path)

    assert "name" in resume_data
    assert "contact" in resume_data
    assert "sections" in resume_data

    sections = resume_data["sections"]

    assert "education" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "projects" in sections
    assert "certifications" in sections

    print("DOCX resume test passed!")


def test_invalid_file():

    try:
        parse_resume("data/resumes/does_not_exist.pdf")

    except FileNotFoundError:
        print("Invalid file test passed!")

    else:
        raise AssertionError("Expected FileNotFoundError")


def test_unsupported_file():

    test_file = Path("data/resumes/resume.txt")

    test_file.write_text("This is a test file.")

    try:
        parse_resume(str(test_file))

    except ValueError as error:

        assert "Only PDF and DOCX files are supported." in str(error)

        print("Unsupported file test passed!")

    else:
        raise AssertionError("Expected ValueError")

    finally:
        test_file.unlink(missing_ok=True)


if __name__ == "__main__":

    test_pdf_resume()
    test_docx_resume()
    test_invalid_file()
    test_unsupported_file()

    print("\nAll resume parser tests passed!")