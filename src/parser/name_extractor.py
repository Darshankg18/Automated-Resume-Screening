import re


# Words that should never be treated as a person's name
INVALID_NAME_WORDS = {
    "RESUME",
    "CV",
    "CURRICULUM VITAE",
    "PROFILE",
    "SUMMARY",
    "OBJECTIVE",
    "EDUCATION",
    "EDUCATIONAL QUALIFICATION",
    "EDUCATIONAL QUALIFICATIONS",
    "ACADEMIC QUALIFICATION",
    "ACADEMIC QUALIFICATIONS",
    "SKILLS",
    "TECHNICAL SKILLS",
    "TECHNOLOGY SKILLS",
    "CORE SKILLS",
    "KEY SKILLS",
    "TECHNOLOGIES",
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "WORK HISTORY",
    "EMPLOYMENT HISTORY",
    "PROJECTS",
    "ACADEMIC PROJECTS",
    "PERSONAL PROJECTS",
    "CERTIFICATIONS",
    "CERTIFICATES",
    "ACHIEVEMENTS",
    "AWARDS",
    "HONORS",
    "HONOURS",
    "LANGUAGES",
    "HOBBIES",
    "INTERESTS",
}


def clean_name_candidate(line):
    """
    Clean a possible name extracted from a resume.
    """

    name = line.strip()

    name = re.sub(
        r"^(name|candidate name)\s*:\s*",
        "",
        name,
        flags=re.IGNORECASE
    )

    name = re.sub(r"\s+", " ", name)

    return name.strip()


def looks_like_name(line):
    """
    Check whether a line looks like a person's name.
    """

    name = clean_name_candidate(line)

    if not name:
        return False

    upper_name = name.upper()

    # Reject known resume headings
    if upper_name in INVALID_NAME_WORDS:
        return False

    # Reject lines containing common resume labels
    if any(
        upper_name.startswith(word + ":")
        for word in INVALID_NAME_WORDS
    ):
        return False

    # Reject email addresses
    if "@" in name:
        return False

    # Reject URLs
    if re.search(r"https?://|www\.", name, re.IGNORECASE):
        return False

    # Reject numbers
    if re.search(r"\d", name):
        return False

    words = name.split()

    # A normal full name generally has 2–5 words
    if not 2 <= len(words) <= 5:
        return False

    # Every word must look like a name component
    for word in words:

        if not re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", word):
            return False

    return True


def extract_name(text):
    """
    Extract the most likely candidate name from resume text.
    """

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if looks_like_name(line):
            return clean_name_candidate(line)

    return None