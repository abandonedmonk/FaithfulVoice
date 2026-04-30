import re
from pathlib import Path
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Text, NarrativeText, Table, ListItem, Image
import warnings

from src.cleaner import SECTION_PATTERNS, should_skip, build_toc_map

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTION_RE = re.compile(r"^item\s*\d+[a-z]?\.", re.I)


@dataclass
class ParsedElement:
    text: str
    element_type: str
    section: str = "Cover Page"
    anchor_id: str = "cover"
    metadata: dict = field(default_factory=dict)


def _detect_section(text: str) -> tuple[str, str] | None:
    clean = text.strip()
    for pattern, anchor_id, section_label in SECTION_PATTERNS:
        if pattern.search(clean):
            return anchor_id, section_label
    return None


def parse_filing(cleaned_path: str | Path) -> list[ParsedElement]:
    cleaned_path = Path(cleaned_path)
    elements = partition_html(
        filename=str(cleaned_path),
        skip_headers_and_footers=True,
        include_metadata=True,
    )

    with open(cleaned_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    toc = build_toc_map(soup)

    parsed: list[ParsedElement] = []
    current_anchor = "cover"
    current_section = "Cover Page"

    for el in elements:
        el_type = type(el).__name__
        text = str(el).strip()

        section_hit = _detect_section(text)
        if section_hit:
            current_anchor, current_section = section_hit
            if el_type == "Text" and len(text) < 80:
                continue

        if should_skip(text):
            continue

        if el_type == "Image":
            continue

        parsed.append(ParsedElement(
            text=text,
            element_type=el_type,
            section=current_section,
            anchor_id=current_anchor,
        ))

    return parsed
