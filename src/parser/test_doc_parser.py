# Test the DOCX parser
from src.parser. doc_parser import extract_text_from_docx

# Path to the sample resume
docx_path = r"data\resumes\sample_resume.docx"

# Extract text
text = extract_text_from_docx(docx_path)

# Display the extracted text
print(text)