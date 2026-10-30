import re


EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

PHONE_PATTERNS = [
    r"\+\d{1,3}\s+\d{5}\s+\d{5}",
    r"\+\d{1,3}-\d{5}-\d{5}",
    r"\b\d{5}-\d{5}\b",
    r"\b\d{5}\s\d{5}\b",
    r"\b\d{10}\b",
]


def extract_email(text):
    match = re.search(EMAIL_PATTERN, text)

    if match:
        return match.group(0)

    return None


def extract_phone(text):
    for pattern in PHONE_PATTERNS:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return None


def extract_contact_information(text):
    return {
        "email": extract_email(text),
        "phone": extract_phone(text)
    }