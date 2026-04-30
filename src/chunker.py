import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

from src.parser import ParsedElement, parse_filing
from src.metadata import parse_filename, build_source_url

COVER_ANCHOR = "cover"

SKIP_ANCHORS = {
    "cover",
    "item4",
    "item6",
    "item9",
    "item9b",
    "item9c",
    "item10",
    "item11",
    "item14",
    "item16",
}

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass
class Chunk:
    text: str
    parent_text: str = ""
    table_markdown: str = ""
    company: str = ""
    ticker: str = ""
    cik: str = ""
    year: int = 0
    quarter: int | None = None
    filing_type: str = ""
    accession_number: str = ""
    section: str = ""
    anchor_id: str = ""
    element_type: str = ""
    chunk_index: int = 0
    parent_chunk_id: str = ""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_url: str = ""
    htm_filename: str = ""


def _extract_table_markdown(text: str) -> str:
    try:
        dfs = pd.read_html(text, flavor="lxml") if "<table" in text or "|" in text else []
        if dfs:
            return dfs[0].to_markdown(index=False)
    except Exception:
        pass
    return ""


def _group_by_section(elements: list[ParsedElement]) -> list[tuple[str, str, list[ParsedElement]]]:
    groups = []
    current_anchor = None
    current_section = None
    current_elements = []

    for el in elements:
        if el.anchor_id != current_anchor:
            if current_elements and current_anchor not in SKIP_ANCHORS:
                groups.append((current_anchor, current_section, current_elements))
            current_anchor = el.anchor_id
            current_section = el.section
            current_elements = []
        current_elements.append(el)

    if current_elements and current_anchor not in SKIP_ANCHORS:
        groups.append((current_anchor, current_section, current_elements))

    return groups


def chunk_filing(cleaned_path: str | Path, raw_path: str | Path | None = None) -> list[Chunk]:
    cleaned_path = Path(cleaned_path)
    raw_path = Path(raw_path) if raw_path else cleaned_path

    file_meta = parse_filename(raw_path)
    if not file_meta:
        file_meta = parse_filename(cleaned_path)
    if not file_meta:
        file_meta = {"ticker": "", "cik": "", "year": 0, "quarter": None,
                     "filing_type": "", "htm_filename": cleaned_path.name}

    elements = parse_filing(cleaned_path)
    groups = _group_by_section(elements)

    chunks = []
    chunk_idx = 0

    for anchor_id, section, section_elements in groups:
        narrative_texts = []
        table_texts = []
        list_items = []

        for el in section_elements:
            if el.element_type == "Table":
                table_texts.append(el)
            elif el.element_type == "ListItem":
                list_items.append(el)
            elif el.element_type in ("NarrativeText", "Text"):
                narrative_texts.append(el)

        combined_text = "\n\n".join(el.text for el in narrative_texts + list_items)
        if combined_text.strip():
            parent_chunks = parent_splitter.split_text(combined_text)
            for parent_text in parent_chunks:
                child_chunks = child_splitter.split_text(parent_text)
                parent_id = str(uuid.uuid4())

                for child_text in child_chunks:
                    chunks.append(Chunk(
                        text=child_text,
                        parent_text=parent_text,
                        company=_ticker_to_company(file_meta.get("ticker", "")),
                        ticker=file_meta.get("ticker", ""),
                        cik=file_meta.get("cik", ""),
                        year=file_meta.get("year", 0),
                        quarter=file_meta.get("quarter"),
                        filing_type=file_meta.get("filing_type", ""),
                        section=section,
                        anchor_id=anchor_id,
                        element_type="NarrativeText",
                        chunk_index=chunk_idx,
                        parent_chunk_id=parent_id,
                        htm_filename=file_meta.get("htm_filename", ""),
                    ))
                    chunk_idx += 1

        for tbl_el in table_texts:
            tbl_md = _extract_table_markdown(tbl_el.text)
            tbl_summary = tbl_el.text[:512] if not tbl_md else tbl_md

            parent_id = str(uuid.uuid4())
            chunks.append(Chunk(
                text=tbl_summary,
                parent_text=tbl_el.text[:2000],
                table_markdown=tbl_md,
                company=_ticker_to_company(file_meta.get("ticker", "")),
                ticker=file_meta.get("ticker", ""),
                cik=file_meta.get("cik", ""),
                year=file_meta.get("year", 0),
                quarter=file_meta.get("quarter"),
                filing_type=file_meta.get("filing_type", ""),
                section=section,
                anchor_id=anchor_id,
                element_type="Table",
                chunk_index=chunk_idx,
                parent_chunk_id=parent_id,
                htm_filename=file_meta.get("htm_filename", ""),
            ))
            chunk_idx += 1

    return chunks


def _ticker_to_company(ticker: str) -> str:
    names = {
        "NVDA": "NVIDIA Corporation",
        "AMD": "Advanced Micro Devices, Inc.",
        "INTC": "Intel Corporation",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "META": "Meta Platforms, Inc.",
        "AMZN": "Amazon.com, Inc.",
        "TSLA": "Tesla, Inc.",
        "JPM": "JPMorgan Chase & Co.",
    }
    return names.get(ticker, "")
