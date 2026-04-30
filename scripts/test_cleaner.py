import sys
sys.path.insert(0, r"D:\Code\PROJECTS\FaithfulVoice")
from src.cleaner import clean_edgar_html, build_toc_map, extract_sections
from bs4 import BeautifulSoup
import warnings, os
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

raw_path = r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm"
cleaned_path = clean_edgar_html(raw_path)
print(f"Cleaned: {cleaned_path}")

raw_size = os.path.getsize(raw_path)
clean_size = os.path.getsize(cleaned_path)
pct = clean_size / raw_size
print(f"Size: {raw_size:,} -> {clean_size:,} ({pct:.1%} of original)")

with open(cleaned_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "lxml")

toc = build_toc_map(soup)
print(f"\nTOC MAP ({len(toc)} entries):")
for div_id, info in list(toc.items())[:10]:
    short_id = div_id[:25]
    aid = info["anchor_id"]
    sec = info["section"]
    print(f"  div#{short_id}... -> {aid}: {sec}")

sections = extract_sections(soup)
print(f"\nSECTIONS ({len(sections)} found):")
for s in sections:
    aid = s["anchor_id"]
    sec = s["section"]
    print(f"  {aid}: {sec}")
