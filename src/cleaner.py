import re
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SECTION_PATTERNS = [
    (re.compile(r"^item\s*1\b", re.I), "item1", "Business Overview"),
    (re.compile(r"^item\s*1a\b", re.I), "item1a", "Risk Factors"),
    (re.compile(r"^item\s*1b\b", re.I), "item1b", "Unresolved Staff Comments"),
    (re.compile(r"^item\s*1c\b", re.I), "item1c", "Cybersecurity"),
    (re.compile(r"^item\s*2\b", re.I), "item2", "Properties"),
    (re.compile(r"^item\s*3\b", re.I), "item3", "Legal Proceedings"),
    (re.compile(r"^item\s*4\b", re.I), "item4", "Mine Safety"),
    (re.compile(r"^item\s*5\b", re.I), "item5", "Market for Common Equity"),
    (re.compile(r"^item\s*6\b", re.I), "item6", "Reserved"),
    (re.compile(r"^item\s*7\b", re.I), "item7", "MD&A"),
    (re.compile(r"^item\s*7a\b", re.I), "item7a", "Market Risk"),
    (re.compile(r"^item\s*8\b", re.I), "item8", "Financial Statements"),
    (re.compile(r"^item\s*9\b", re.I), "item9", "Accountant Disagreements"),
    (re.compile(r"^item\s*9a\b", re.I), "item9a", "Controls and Procedures"),
    (re.compile(r"^item\s*9b\b", re.I), "item9b", "Other Information"),
    (re.compile(r"^item\s*9c\b", re.I), "item9c", "Foreign Jurisdiction Disclosure"),
    (re.compile(r"^item\s*10\b", re.I), "item10", "Directors and Officers"),
    (re.compile(r"^item\s*11\b", re.I), "item11", "Executive Compensation"),
    (re.compile(r"^item\s*12\b", re.I), "item12", "Security Ownership"),
    (re.compile(r"^item\s*13\b", re.I), "item13", "Related Transactions"),
    (re.compile(r"^item\s*14\b", re.I), "item14", "Accountant Fees"),
    (re.compile(r"^item\s*15\b", re.I), "item15", "Exhibits and Schedules"),
    (re.compile(r"^item\s*16\b", re.I), "item16", "10-K Summary"),
]

BOILERPLATE_PATTERNS = [
    re.compile(r"pursuant to the requirements of the securities exchange act", re.I),
    re.compile(r"incorporated herein by reference", re.I),
    re.compile(r"see exhibit index", re.I),
    re.compile(r"^table of contents$", re.I),
    re.compile(r"^item\s*\d+[a-z]?\.$", re.I),
]


def clean_edgar_html(filepath: str | Path) -> str:
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    for tag in soup.find_all(re.compile(r"^ix:")):
        tag.unwrap()

    for tag in soup.find_all(True, style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()

    ix_header = soup.find("ix:header")
    if ix_header:
        ix_header.decompose()

    cleaned_path = str(filepath).replace(".htm", "_clean.htm")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    return cleaned_path


def build_toc_map(soup: BeautifulSoup) -> dict[str, str]:
    toc = {}
    toc_links = soup.find_all("a", string=re.compile(r"item\s*\d+[a-z]?", re.I))
    for a_tag in toc_links:
        href = a_tag.get("href", "")
        text = a_tag.get_text(strip=True)
        if href.startswith("#"):
            target_id = href[1:]
            for pattern, anchor_id, section_label in SECTION_PATTERNS:
                if pattern.search(text):
                    toc[target_id] = {
                        "anchor_id": anchor_id,
                        "section": section_label,
                        "toc_text": text,
                    }
                    break
    return toc


def extract_sections(soup: BeautifulSoup) -> list[dict]:
    toc = build_toc_map(soup)

    body = soup.find("body")
    if not body:
        return []

    sections = []
    current_section = {"anchor_id": "cover", "section": "Cover Page", "start_div": None}

    for div in body.find_all("div", recursive=True):
        div_id = div.get("id", "")
        if div_id in toc:
            if current_section["start_div"] is not None:
                sections.append(current_section)
            current_section = {
                "anchor_id": toc[div_id]["anchor_id"],
                "section": toc[div_id]["section"],
                "start_div": div,
            }

    if current_section["start_div"] is not None:
        sections.append(current_section)

    header_spans = soup.find_all(
        "span",
        style=re.compile(r"color:#76b900|color:#0000ff", re.I),
        string=re.compile(r"^item\s*\d+[a-z]?", re.I),
    )
    span_sections = []
    for s in header_spans:
        text = s.get_text(strip=True)
        parent_div = s.find_parent("div")
        for pattern, anchor_id, section_label in SECTION_PATTERNS:
            if pattern.search(text):
                span_sections.append({
                    "anchor_id": anchor_id,
                    "section": section_label,
                    "header_text": text,
                    "parent_div": parent_div,
                })
                break

    return sections if sections else span_sections


def should_skip(text: str) -> bool:
    text = text.strip()
    if len(text) < 50:
        return True
    if re.match(r"^\$?[\d,\.]+[bmk]?$", text):
        return True
    return any(p.search(text) for p in BOILERPLATE_PATTERNS)
