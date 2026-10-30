from pathlib import Path

from .pdf_parser import extract_text_from_pdf
from .doc_parser import extract_text_from_docx
from .section_splitter import split_sections
from .contact_extractor import extract_contact_information
from .name_extractor import extract_name


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def parse_resume(file_path):
    """
    Parse a PDF or DOCX resume and return structured resume data.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Resume file not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Only PDF and DOCX files are supported."
        )

    # Extract raw text
    if extension == ".pdf":
        text = extract_text_from_pdf(str(path))
    else:
        text = extract_text_from_docx(str(path))

    if not text or not text.strip():
        raise ValueError(
            "The resume contains no readable text."
        )

    # Extract candidate information
    name = extract_name(text)
    contact = extract_contact_information(text)

    # Split resume into sections
    sections = split_sections(text)

    if not sections:
        raise ValueError(
            "No recognizable resume sections were found."
        )

    return {
        "name": name,
        "contact": contact,
        "sections": sections
    }