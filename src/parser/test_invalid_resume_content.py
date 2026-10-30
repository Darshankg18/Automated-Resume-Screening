import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.parser.resume_parser import parse_resume


def test_empty_resume():

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as file:
        file_path = Path(file.name)

    try:
        parse_resume(file_path)
        assert False, "Expected an error for an empty resume"
    except Exception:
        print("Empty resume test passed!")
    finally:
        file_path.unlink(missing_ok=True)


def test_no_sections():

    with tempfile.NamedTemporaryFile(
        suffix=".docx",
        delete=False
    ) as file:
        file_path = Path(file.name)

    try:
        parse_resume(file_path)
        assert False, "Expected an error for invalid resume content"
    except Exception:
        print("Invalid resume content test passed!")
    finally:
        file_path.unlink(missing_ok=True)


if __name__ == "__main__":

    test_empty_resume()
    test_no_sections()

    print("\nAll invalid resume content tests passed!")