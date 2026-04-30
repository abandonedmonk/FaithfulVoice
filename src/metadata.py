import re
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TICKER_MAP = {
    "NVDA": "0001045810",
    "AMD": "0000002488",
    "INTC": "0000050863",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "JPM": "0000019617",
}

CIK_TO_TICKER = {v: k for k, v in TICKER_MAP.items()}

FILING_RE = re.compile(
    r"^([A-Z]+)_(?:"
    r"(\d{4})_(10K)"
    r"|Q([34])_(\d{4})_(10Q)"
    r")\.htm$"
)


def parse_filename(filepath: str | Path) -> dict | None:
    filepath = Path(filepath)
    name = filepath.name
    m = FILING_RE.match(name)
    if not m:
        return None

    if m.group(2):
        ticker = m.group(1)
        year = int(m.group(2))
        filing_type = m.group(3)
        quarter = None
    else:
        ticker = m.group(1)
        quarter = int(m.group(4))
        year = int(m.group(5))
        filing_type = m.group(6)

    cik = TICKER_MAP.get(ticker, "")
    return {
        "ticker": ticker,
        "cik": cik,
        "year": year,
        "quarter": quarter,
        "filing_type": filing_type,
        "htm_filename": name,
    }


def extract_accession_number(soup: BeautifulSoup) -> str:
    ix_resources = soup.find("ix:resources")
    if ix_resources:
        for tag in soup.find_all(re.compile(r"^ix:")):
            pass
    for div in soup.find_all("div", recursive=True):
        text = div.get_text(strip=True)
        if re.match(r"^\d{10}-\d{2}-\d{6}$", text):
            return text
    return ""


def build_source_url(cik: str, accession_number: str, htm_filename: str, anchor_id: str = "") -> str:
    acc_clean = accession_number.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/"
    url = f"{base}{htm_filename}"
    if anchor_id and anchor_id != "cover":
        url += f"#{anchor_id}"
    return url
