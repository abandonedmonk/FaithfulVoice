from bs4 import BeautifulSoup
import re, warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

with open(r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

# Build the TOC mapping: <a> text -> href -> target <div id>
print("=== TOC LINK -> TARGET DIV MAPPING ===")
toc_links = soup.find_all("a", string=re.compile(r"item\s*\d+[a-z]?", re.I))
for a in toc_links:
    href = a.get("href", "")
    text = a.get_text(strip=True)
    if href.startswith("#"):
        target_id = href[1:]
        target_div = soup.find("div", attrs={"id": target_id})
        if target_div:
            # Get text preview of the target div
            preview = target_div.get_text(strip=True)[:120].replace("\n", " ")
            print(f"  '{text}' -> div#{target_id} => '{preview}'")
        else:
            print(f"  '{text}' -> div#{target_id} => NOT FOUND")

# Now check: what are the section header spans with color #76b900?
print("\n=== SECTION HEADER SPANS (color:#76b900) ===")
green_spans = soup.find_all("span", style=re.compile(r"color:#76b900", re.I))
for s in green_spans:
    text = s.get_text(strip=True)
    if text:
        parent = s.parent
        parent_id = parent.get("id", "") if parent else ""
        print(f"  text='{text}'  parent_id='{parent_id}'")

# Check what the div IDs that TOC links point to contain
# Specifically, does the div with id=i13eac..._13 contain "Item 1. Business"?
print("\n=== WHAT'S INSIDE TARGET DIVS? ===")
for a in toc_links[:5]:
    href = a.get("href", "")
    text = a.get_text(strip=True)
    if href.startswith("#"):
        target_id = href[1:]
        target_div = soup.find("div", attrs={"id": target_id})
        if target_div:
            # Get first few child elements
            children = list(target_div.children)
            for c in children[:5]:
                if hasattr(c, "name") and c.name:
                    ctext = c.get_text(strip=True)[:100]
                    print(f"  div#{target_id} ({text}) child <{c.name}> => '{ctext}'")
