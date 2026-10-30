# Import the Document class from python-docx
from docx import Document


# Extract text from a DOCX resume
def extract_text_from_docx(docx_path):

    # Open the DOCX file
    document = Document(docx_path)

    # Create an empty string to store the text
    text = ""

    # Read each paragraph
    for paragraph in document.paragraphs:

        # Add paragraph text
        text += paragraph.text + "\n"

    # Return the complete extracted text
    return text