import requests
import json
import time
import argparse
import sys
from pathlib import Path

API_HEADERS = {
    "User-Agent": "FaithfulVoice/1.0 research@faithfulvoice.ai",
    "Accept-Encoding": "gzip, deflate",
}

DOWNLOAD_HEADERS = {
    "User-Agent": "FaithfulVoice/1.0 research@faithfulvoice.ai",
    "Accept-Encoding": "gzip, deflate",
}

COMPANIES = {
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

FILING_SPECS = [
    {"form": "10-K", "label": "2024_10K", "selector": "report_year", "year": 2024},
    {"form": "10-K", "label": "2023_10K", "selector": "report_year", "year": 2023},
    {"form": "10-Q", "label": "Q4_2024_10Q", "selector": "report_quarter", "year": 2024, "quarter": 4},
    {"form": "10-Q", "label": "Q3_2024_10Q", "selector": "report_quarter", "year": 2024, "quarter": 3},
]


def fetch_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=API_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _date_to_quarter(date_str: str) -> tuple[int, int]:
    year = int(date_str[:4])
    month = int(date_str[5:7])
    return year, (month - 1) // 3 + 1


def find_filing(submissions: dict, spec: dict) -> dict | None:
    recent = submissions["filings"]["recent"]
    form = spec["form"]
    selector = spec["selector"]

    for i, f in enumerate(recent["form"]):
        if f != form:
            continue
        report_date = recent["reportDate"][i]
        if not report_date:
            continue
        ry, rq = _date_to_quarter(report_date)

        if selector == "report_year":
            if ry == spec["year"]:
                return {
                    "accessionNumber": recent["accessionNumber"][i],
                    "filingDate": recent["filingDate"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                    "reportDate": report_date,
                }
        elif selector == "report_quarter":
            if ry == spec["year"] and rq == spec["quarter"]:
                return {
                    "accessionNumber": recent["accessionNumber"][i],
                    "filingDate": recent["filingDate"][i],
                    "primaryDocument": recent["primaryDocument"][i],
                    "reportDate": report_date,
                }
    return None


def download_filing(cik: str, accession_number: str, filename: str, outpath: Path) -> bool:
    acc_clean = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{filename}"
    try:
        r = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=60)
        r.raise_for_status()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_bytes(r.content)
        print(f"  Downloaded {len(r.content):,} bytes -> {outpath.name}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download SEC EDGAR filings for FaithfulVoice")
    parser.add_argument("--tickers", nargs="+", default=list(COMPANIES.keys()),
                        help="Company tickers to download (default: all 10)")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Output directory for downloaded filings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [t.upper() for t in args.tickers if t.upper() in COMPANIES]
    if not tickers:
        print("No valid tickers found. Available:", list(COMPANIES.keys()))
        sys.exit(1)

    total = len(tickers) * len(FILING_SPECS)
    downloaded = 0
    failed = 0
    skipped = 0

    for ticker in tickers:
        cik = COMPANIES[ticker]
        print(f"\n{'='*60}")
        print(f"  {ticker} (CIK: {cik})")
        print(f"{'='*60}")

        try:
            submissions = fetch_submissions(cik)
        except Exception as e:
            print(f"  FAILED to fetch submissions: {e}")
            failed += len(FILING_SPECS)
            continue

        time.sleep(0.15)

        for spec in FILING_SPECS:
            form = spec["form"]
            label = spec["label"]

            filing = find_filing(submissions, spec)
            if not filing:
                print(f"  {label}: NOT FOUND")
                skipped += 1
                continue

            outpath = output_dir / f"{ticker}_{label}.htm"

            if outpath.exists() and outpath.stat().st_size > 1000:
                print(f"  {label}: already exists ({outpath.stat().st_size:,} bytes), skipping")
                skipped += 1
                continue

            if args.dry_run:
                print(f"  {label}: would download {filing['primaryDocument']} "
                      f"(filed {filing['filingDate']})")
                continue

            ok = download_filing(cik, filing["accessionNumber"],
                                 filing["primaryDocument"], outpath)
            if ok:
                downloaded += 1
            else:
                failed += 1

            time.sleep(0.15)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {downloaded} downloaded, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
