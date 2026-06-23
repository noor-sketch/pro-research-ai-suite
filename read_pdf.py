import io
import re
from urllib.parse import urlparse

import PyPDF2
import requests
from langchain_core.tools import tool


REQUEST_TIMEOUT_SECONDS = 45
PDF_SIGNATURE = b"%PDF"


def _normalize_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    normalized = url.strip()

    if "arxiv.org" in parsed.netloc:
        if "/abs/" in parsed.path:
            normalized = normalized.replace("/abs/", "/pdf/")
        if "/pdf/" in normalized and not normalized.endswith(".pdf"):
            normalized = f"{normalized}.pdf"

    return normalized


def _extract_arxiv_pdf_url_from_html(html: str) -> str | None:
    match = re.search(r'href="(https?://arxiv\.org/pdf/[^"]+\.pdf)"', html)
    if match:
        return match.group(1)

    match = re.search(r'href="(/pdf/[^"]+\.pdf)"', html)
    if match:
        return f"https://arxiv.org{match.group(1)}"

    return None


def _download_pdf_bytes(url: str) -> bytes:
    normalized_url = _normalize_pdf_url(url)
    response = requests.get(
        normalized_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
        headers={"User-Agent": "AI-Researcher/1.0"},
    )
    response.raise_for_status()

    content = response.content
    content_type = (response.headers.get("Content-Type") or "").lower()

    if content.startswith(PDF_SIGNATURE):
        return content

    if "text/html" in content_type or b"<html" in content[:500].lower():
        fallback_url = _extract_arxiv_pdf_url_from_html(response.text)
        if fallback_url and fallback_url != normalized_url:
            retry = requests.get(
                fallback_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
                headers={"User-Agent": "AI-Researcher/1.0"},
            )
            retry.raise_for_status()
            if retry.content.startswith(PDF_SIGNATURE):
                return retry.content

    raise ValueError(
        "The URL did not return a valid PDF file. Please use a direct PDF link or a supported arXiv paper URL."
    )


@tool
def read_pdf(url: str) -> str:
    """Read and extract text from a PDF file given its URL."""
    try:
        pdf_bytes = _download_pdf_bytes(url)
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
        page_text: list[str] = []

        for page in pdf_reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_text.append(extracted.strip())

        text = "\n\n".join(page_text).strip()
        if not text:
            raise ValueError("The PDF was downloaded, but no readable text could be extracted.")

        return text
    except Exception as exc:
        print(f"Error reading PDF: {exc}")
        raise
