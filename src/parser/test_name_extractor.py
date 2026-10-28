from src.parser.name_extractor import extract_name


def test_name_extraction():

    test_cases = {
        "JOHN SMITH": "JOHN SMITH",
        "Jane Doe": "Jane Doe",
        "Robert James Smith": "Robert James Smith",
        "Name: Michael Johnson": "Michael Johnson",
        "Candidate Name: Sarah Williams": "Sarah Williams",
    }

    for text, expected in test_cases.items():

        result = extract_name(text)

        assert result == expected, (
            f"Name test failed: {text}"
        )

    print("Basic name extraction tests passed!")


def test_name_with_contact_information():

    resume_text = """
    JOHN SMITH

    Email: john.smith@example.com
    Phone: 9876543210

    EDUCATION
    Bachelor of Engineering in Computer Science
    """

    result = extract_name(resume_text)

    assert result == "JOHN SMITH"

    print("Name extraction with contact information passed!")


def test_invalid_name_candidates():

    invalid_cases = [
        "RESUME",
        "CURRICULUM VITAE",
        "EDUCATION",
        "TECHNICAL SKILLS",
        "john.smith@example.com",
        "9876543210",
        "www.example.com",
    ]

    for text in invalid_cases:

        result = extract_name(text)

        assert result is None, (
            f"Invalid name was detected: {text}"
        )

    print("Invalid name tests passed!")


if __name__ == "__main__":

    test_name_extraction()
    test_name_with_contact_information()
    test_invalid_name_candidates()

    print("\nAll name extractor tests passed!")