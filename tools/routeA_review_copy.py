#!/usr/bin/env python3
"""CCE review copy: continuous line numbers and double spacing.

Does not replace output/Manuscript.docx (zh-academic audit PASS).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output" / "Manuscript.docx"
OUT = ROOT / "output" / "Manuscript review copy.docx"


def add_line_numbers(document: Document) -> None:
    for section in document.sections:
        sect_pr = section._sectPr
        existing = sect_pr.find(qn("w:lnNumType"))
        if existing is not None:
            sect_pr.remove(existing)
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        sect_pr.append(ln)


def double_space(document: Document) -> None:
    skip = {"Title", "Table Text", "Caption", "Reference"}
    for paragraph in document.paragraphs:
        if paragraph.style and paragraph.style.name in skip:
            continue
        pf = paragraph.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
        pf.line_spacing = 2.0
        if pf.space_after is None or pf.space_after < Pt(0):
            pf.space_after = Pt(0)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    shutil.copy2(SRC, OUT)
    document = Document(str(OUT))
    add_line_numbers(document)
    double_space(document)
    document.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
