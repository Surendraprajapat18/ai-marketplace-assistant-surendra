from pypdf import PdfReader

def extract_pages_text(pdf_path):
    pages = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i, text.strip()))
    return pages
