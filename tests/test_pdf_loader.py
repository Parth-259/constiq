"""Hermetic tests for backend.pipeline.pdf_loader.

PDFs are generated inline with reportlab into pytest's tmp_path — no project
data/ files are touched and no network is used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

from backend.pipeline.pdf_loader import PDFExtractionError, extract_pdf


def _make_text_pdf(path: Path) -> None:
    """Two text pages drawn with the low-level canvas API."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawString(72, 720, "Tender for Fe500D TMT steel supply.")
    c.drawString(72, 700, "Quantity required is 120 tonnes.")
    c.showPage()
    c.drawString(72, 720, "Delivery deadline is 2026-08-10.")
    c.showPage()
    c.save()


def _make_blank_second_page_pdf(path: Path) -> None:
    """Page 1 has text; page 2 is completely blank."""
    c = canvas.Canvas(str(path), pagesize=A4)
    c.drawString(72, 720, "Only this first page has content.")
    c.showPage()
    c.showPage()  # blank page 2
    c.save()


def _make_table_pdf(path: Path) -> None:
    """A page containing a gridded table plus a paragraph."""
    styles = getSampleStyleSheet()
    table = Table(
        [
            ["Material", "Qty", "Unit"],
            ["Fe500D TMT", "120", "tonne"],
            ["OPC 53", "800", "bag"],
        ]
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build([Paragraph("Bill of quantities follows.", styles["Normal"]), table])


def test_extract_text_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "tender.pdf"
    _make_text_pdf(pdf_path)

    pages = extract_pdf(str(pdf_path))

    assert len(pages) == 2
    for i, page in enumerate(pages):
        assert set(page.keys()) == {"page_number", "text", "tables", "source_file"}
        assert page["page_number"] == i + 1  # 1-based
        assert isinstance(page["text"], str)
        assert isinstance(page["tables"], list)
        assert page["source_file"] == "tender.pdf"  # basename, not full path
    assert "Fe500D TMT steel" in pages[0]["text"]
    assert "120 tonnes" in pages[0]["text"]
    assert "2026-08-10" in pages[1]["text"]


def test_blank_page_kept_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = tmp_path / "half_blank.pdf"
    _make_blank_second_page_pdf(pdf_path)

    with caplog.at_level(logging.WARNING, logger="backend.pipeline.pdf_loader"):
        pages = extract_pdf(str(pdf_path))

    assert len(pages) == 2  # blank page entry is kept
    assert pages[1]["page_number"] == 2
    assert pages[1]["text"].strip() == ""
    assert pages[1]["tables"] == []
    warning_messages = [
        rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING
    ]
    assert any("no extractable text" in msg for msg in warning_messages)


def test_missing_file_raises_chained(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(PDFExtractionError) as excinfo:
        extract_pdf(str(missing))
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_corrupt_file_raises_chained(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"this is definitely not a pdf")
    with pytest.raises(PDFExtractionError) as excinfo:
        extract_pdf(str(corrupt))
    assert excinfo.value.__cause__ is not None  # original exception chained


def test_tables_return_shape(tmp_path: Path) -> None:
    """Whatever pdfplumber finds, the tables field must honour the contract
    shape: list of tables -> list of rows -> list of str cells."""
    pdf_path = tmp_path / "with_table.pdf"
    _make_table_pdf(pdf_path)

    pages = extract_pdf(str(pdf_path))

    assert len(pages) >= 1
    for page in pages:
        assert isinstance(page["tables"], list)
        for table in page["tables"]:
            assert isinstance(table, list)
            assert len(table) > 0
            for row in table:
                assert isinstance(row, list)
                for cell in row:
                    assert isinstance(cell, str)
    assert "Bill of quantities" in pages[0]["text"]
