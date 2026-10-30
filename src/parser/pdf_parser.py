# Import PyMuPDF
import pymupdf


# Extract text from a PDF resume
def extract_text_from_pdf(pdf_path):

    # Open the PDF file
    document = pymupdf.open(pdf_path)

    # Create an empty string to store extracted text
    text = ""

    # Read each page of the PDF
    for page in document:

        # Extract text from the current page
        text += page.get_text()

    # Return the complete extracted text
    return text