"""PDF extraction for ConstructIQ.

Text is pulled per-page with PyMuPDF (``fitz``); tables are pulled with a
separate ``pdfplumber`` open of the same file. Every page yields one dict —
pages with no text and no tables are kept (with a warning) so page numbering
stays faithful to the source document.
"""
from __future__ import annotations

import logging
import os

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when a PDF is missing or cannot be parsed.

    Always raised with the original exception chained (``raise ... from exc``)
    so callers can inspect ``__cause__`` for the underlying failure.
    """


def _normalize_table(raw_table: list[list[object]]) -> list[list[str]]:
    """Coerce every cell of a pdfplumber table to ``str`` (None -> "")."""
    return [
        [("" if cell is None else str(cell)) for cell in row]
        for row in raw_table
    ]


def _extract_texts(file_path: str) -> list[str]:
    """Per-page text via PyMuPDF. Raises PDFExtractionError (chained)."""
    try:
        doc = fitz.open(file_path)
    except Exception as exc:  # fitz raises several exception types
        raise PDFExtractionError(
            f"PyMuPDF failed to open PDF: {file_path}"
        ) from exc
    try:
        return [page.get_text("text") for page in doc]
    except Exception as exc:
        raise PDFExtractionError(
            f"PyMuPDF failed to read page text from: {file_path}"
        ) from exc
    finally:
        doc.close()


def _extract_tables(file_path: str) -> dict[int, list[list[list[str]]]]:
    """Per-page tables (0-based page index) via a separate pdfplumber open."""
    tables_by_page: dict[int, list[list[list[str]]]] = {}
    try:
        with pdfplumber.open(file_path) as pdf:
            for index, page in enumerate(pdf.pages):
                raw_tables = page.extract_tables() or []
                tables_by_page[index] = [
                    _normalize_table(table) for table in raw_tables if table
                ]
    except Exception as exc:
        raise PDFExtractionError(
            f"pdfplumber failed to extract tables from: {file_path}"
        ) from exc
    return tables_by_page


def extract_pdf(file_path: str) -> list[dict]:
    """Extract one dict per page of the PDF at ``file_path``.

    Returns::

        [{"page_number": int (1-based),
          "text": str,
          "tables": list[list[list[str]]],
          "source_file": str (basename)}, ...]

    Raises:
        PDFExtractionError: if the file is missing or cannot be parsed;
            the original exception is chained as ``__cause__``.
    """
    if not os.path.isfile(file_path):
        original = FileNotFoundError(f"No such file: {file_path}")
        raise PDFExtractionError(f"PDF file not found: {file_path}") from original

    source_file = os.path.basename(file_path)
    page_texts = _extract_texts(file_path)
    tables_by_page = _extract_tables(file_path)

    pages: list[dict] = []
    for index, text in enumerate(page_texts):
        tables = tables_by_page.get(index, [])
        if not text.strip() and not tables:
            logger.warning(
                "Page %d of %s has no extractable text or tables; keeping empty entry",
                index + 1,
                source_file,
            )
        pages.append(
            {
                "page_number": index + 1,
                "text": text,
                "tables": tables,
                "source_file": source_file,
            }
        )

    logger.info("Extracted %d page(s) from %s", len(pages), source_file)
    return pages
