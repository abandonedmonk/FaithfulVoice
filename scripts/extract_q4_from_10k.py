import json
import sys
from pathlib import Path
from dataclasses import asdict

from src.cleaner import clean_edgar_html
from src.parser import parse_filing
from src.chunker import _group_by_section, parent_splitter, child_splitter, _extract_table_markdown, _ticker_to_company, Chunk
from src.metadata import parse_filename

Q4_KEEP_ANCHORS = {"item7", "item7a", "item8", "item9a"}

ANCHOR_TO_10Q = {
    "item7": "item2",
    "item7a": "item3",
    "item8": "item1",
    "item9a": "item9a",
}

SECTION_REMAP = {
    "item7": "MD&A",
    "item7a": "Market Risk",
    "item8": "Financial Statements",
    "item9a": "Controls and Procedures",
}

Q4_SKIP_ANCHORS = {"cover", "item6", "item9", "item9b", "item9c", "item10", "item11", "item14", "item16"}


def extract_q4_from_10k(ten_k_path: str | Path, output_path: str | Path) -> int:
    ten_k_path = Path(ten_k_path)
    output_path = Path(output_path)

    cleaned_path = clean_edgar_html(ten_k_path)
    elements = parse_filing(cleaned_path)

    from src.parser import ParsedElement
    q4_elements = []
    for el in elements:
        if el.anchor_id not in Q4_KEEP_ANCHORS:
            continue
        new_anchor = ANCHOR_TO_10Q[el.anchor_id]
        new_section = SECTION_REMAP[el.anchor_id]
        q4_elements.append(ParsedElement(
            text=el.text,
            element_type=el.element_type,
            section=new_section,
            anchor_id=new_anchor,
            metadata=el.metadata,
        ))

    if not q4_elements:
        return 0

    groups = _group_by_section(q4_elements, skip_anchors=Q4_SKIP_ANCHORS)

    file_meta = parse_filename(ten_k_path)
    if not file_meta:
        file_meta = {"ticker": "", "cik": "", "year": 0, "quarter": None,
                     "filing_type": "", "htm_filename": ten_k_path.name}
    file_meta["quarter"] = 4
    file_meta["filing_type"] = "10-Q"
    file_meta["htm_filename"] = f"{file_meta.get('ticker', '')}_Q4_{file_meta.get('year', 0)}_10Q.htm"

    import uuid
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
                        quarter=4,
                        filing_type="10-Q",
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
                quarter=4,
                filing_type="10-Q",
                section=section,
                anchor_id=anchor_id,
                element_type="Table",
                chunk_index=chunk_idx,
                parent_chunk_id=parent_id,
                htm_filename=file_meta.get("htm_filename", ""),
            ))
            chunk_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    return len(chunks)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "JPM"
    year = sys.argv[2] if len(sys.argv) > 2 else "2024"

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    ten_k = raw_dir / f"{ticker}_{year}_10K.htm"
    out_file = Path(__file__).parent.parent / "data" / "processed" / f"{ticker}_Q4_{year}_10Q_chunks.jsonl"

    if not ten_k.exists():
        print(f"ERROR: {ten_k.name} not found")
        sys.exit(1)

    print(f"Extracting Q4 chunks from {ten_k.name}...")
    count = extract_q4_from_10k(ten_k, out_file)
    print(f"  Produced {count} chunks -> {out_file.name} ({out_file.stat().st_size:,} bytes)")
