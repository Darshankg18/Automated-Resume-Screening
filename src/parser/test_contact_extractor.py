from src.parser.contact_extractor import (
    extract_email,
    extract_phone,
    extract_contact_information
)


def test_email_extraction():

    test_cases = {
        "john.smith@example.com": "john.smith@example.com",
        "Contact: jane_doe123@gmail.com": "jane_doe123@gmail.com",
        "Email: student.name@college.edu": "student.name@college.edu",
    }

    for text, expected in test_cases.items():

        result = extract_email(text)

        assert result == expected, (
            f"Email test failed: {text}"
        )

    print("Email extraction tests passed!")


def test_phone_extraction():

    test_cases = [
        ("Phone: 9876543210", "9876543210"),
        ("Phone: +91 98765 43210", "+91 98765 43210"),
        ("Contact: +91-98765-43210", "+91-98765-43210"),
        ("Mobile: 98765-43210", "98765-43210"),
    ]

    for text, expected in test_cases:

        result = extract_phone(text)

        assert result == expected, (
            f"Phone test failed: {text}"
        )

    print("Phone extraction tests passed!")


def test_contact_information():

    resume_text = """
    JOHN SMITH

    Email: john.smith@example.com
    Phone: +91 98765 43210

    EDUCATION
    Bachelor of Engineering in Computer Science
    """

    contact = extract_contact_information(resume_text)

    assert contact["email"] == "john.smith@example.com"
    assert contact["phone"] == "+91 98765 43210"

    print("Complete contact extraction test passed!")


if __name__ == "__main__":

    test_email_extraction()
    test_phone_extraction()
    test_contact_information()

    print("All contact extractor tests passed!")