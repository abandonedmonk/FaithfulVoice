from bs4 import BeautifulSoup
import re, warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

with open(r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

# Search for Item markers in text content
print("=== SEARCHING FOR 'Item' SECTION HEADERS ===")

# Look for text nodes containing "Item 1", "Item 1A", etc.
item_pattern = re.compile(r"item\s*\d+[a-z]?", re.I)

# Check all elements with direct text matching "Item X"
for tag in soup.find_all(string=item_pattern):
    parent = tag.parent
    parent_name = parent.name if parent else "none"
    text = tag.strip()[:200]
    print(f"  <{parent_name}> => '{text}'")

print("\n=== SPECIFIC: 'PART I', 'PART II', 'PART III', 'PART IV' ===")
part_pattern = re.compile(r"part\s+(i|ii|iii|iv)\b", re.I)
for tag in soup.find_all(string=part_pattern):
    parent = tag.parent
    parent_name = parent.name if parent else "none"
    text = tag.strip()[:200]
    print(f"  <{parent_name}> => '{text}'")

print("\n=== FIRST 3000 CHARS OF BODY TEXT ===")
body = soup.find("body")
if body:
    text = body.get_text(separator="\n", strip=True)
    print(text[:3000])

print("\n=== SEARCHING FOR SPAN/DIV WITH FONT-WEIGHT:BOLD OR FONT-SIZE:LARGE ===")
bold_els = soup.find_all(["b", "strong", "span"], style=re.compile(r"font-weight\s*:\s*bold|font-size\s*:\s*(1[4-9]|[2-9]\d)px", re.I))
print(f"Found {len(bold_els)} bold/large elements")
for el in bold_els[:30]:
    text = el.get_text(strip=True)[:120]
    if text and len(text) > 5:
        print(f"  <{el.name}> style='{el.get('style','')[:60]}' => '{text}'")
