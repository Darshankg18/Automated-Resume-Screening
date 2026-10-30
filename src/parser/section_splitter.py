import re


SECTION_ALIASES = {
    "summary": {
        "SUMMARY",
        "PROFESSIONAL SUMMARY",
        "CAREER SUMMARY",
        "PROFILE",
        "PROFESSIONAL PROFILE",
        "CAREER OBJECTIVE",
        "OBJECTIVE",
    },

    "education": {
        "EDUCATION",
        "EDUCATIONAL QUALIFICATION",
        "EDUCATIONAL QUALIFICATIONS",
        "ACADEMIC QUALIFICATION",
        "ACADEMIC QUALIFICATIONS",
        "EDUCATIONAL BACKGROUND",
        "ACADEMIC BACKGROUND",
    },

    "skills": {
        "SKILLS",
        "TECHNICAL SKILLS",
        "TECHNOLOGY SKILLS",
        "CORE SKILLS",
        "KEY SKILLS",
        "TECHNOLOGIES",
        "TECHNICAL EXPERTISE",
        "TECHNOLOGY STACK",
        "TECHNICAL SKILLS AND TOOLS",
    },

    "experience": {
        "EXPERIENCE",
        "WORK EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "WORK HISTORY",
        "EMPLOYMENT HISTORY",
        "EMPLOYMENT EXPERIENCE",
        "PROFESSIONAL HISTORY",
    },

    "projects": {
        "PROJECTS",
        "ACADEMIC PROJECTS",
        "PERSONAL PROJECTS",
        "PROFESSIONAL PROJECTS",
        "KEY PROJECTS",
    },

    "certifications": {
        "CERTIFICATIONS",
        "CERTIFICATES",
        "PROFESSIONAL CERTIFICATIONS",
    },

    "achievements": {
        "ACHIEVEMENTS",
        "AWARDS",
        "HONORS",
        "HONOURS",
    },

    "languages": {
        "LANGUAGES",
        "LANGUAGE SKILLS",
    },

    "interests": {
        "INTERESTS",
        "HOBBIES",
        "INTERESTS AND HOBBIES",
    },

    "strengths": {
        "STRENGTHS",
        "KEY STRENGTHS",
        "CORE STRENGTHS",
        "PERSONAL STRENGTHS",
    },

    "extracurricular": {
        "EXTRACURRICULAR ACTIVITIES",
        "EXTRACURRICULAR",
        "ACTIVITIES",
    },

    "volunteering": {
        "VOLUNTEER EXPERIENCE",
        "VOLUNTEERING",
        "VOLUNTEER WORK",
    },

    "publications": {
        "PUBLICATIONS",
        "RESEARCH PUBLICATIONS",
        "PAPERS",
    },

    "declaration": {
        "DECLARATION",
    },

    "personal_details": {
        "PERSONAL DETAILS",
        "PERSONAL INFORMATION",
    },

    "references": {
        "REFERENCES",
        "REFERENCE",
    },
}


def normalize_heading(line):
    heading = line.strip()

    heading = re.sub(
        r"^(?:\d+(?:\.\d+)*[\s.)-]+|[-•▪●]\s*)",
        "",
        heading
    )

    heading = heading.rstrip(":")
    heading = re.sub(r"\s+", " ", heading)

    return heading.upper().strip()


def detect_section_heading(line):
    heading = normalize_heading(line)

    for section, aliases in SECTION_ALIASES.items():
        if heading in aliases:
            return section

    return None


def split_sections(text):
    sections = {}
    current_section = None

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        detected_section = detect_section_heading(line)

        if detected_section:
            current_section = detected_section

            if current_section not in sections:
                sections[current_section] = ""

            continue

        if current_section:
            sections[current_section] += line + "\n"

    for section in sections:
        sections[section] = sections[section].strip()

    return sections