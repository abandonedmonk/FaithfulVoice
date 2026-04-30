from bs4 import BeautifulSoup
import re, warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

with open(r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

# Find the <a> tags with "Item X" text - are these TOC or section anchors?
print("=== <a> TAGS WITH 'Item X' TEXT ===")
item_a_tags = soup.find_all("a", string=re.compile(r"item\s*\d+[a-z]?", re.I))
for a in item_a_tags[:10]:
    href = a.get("href", "")
    name = a.get("name", "")
    id_attr = a.get("id", "")
    text = a.get_text(strip=True)
    print(f"  href='{href}' name='{name}' id='{id_attr}' text='{text}'")
    # Show parent context
    parent = a.parent
    if parent:
        print(f"    parent=<{parent.name}> class='{parent.get('class','')}'")

# Find the <span> tags with "Item X" text - the actual section headers
print("\n=== <span> TAGS WITH 'Item X' TEXT ===")
item_span_tags = soup.find_all("span", string=re.compile(r"item\s*\d+[a-z]?.*[a-z]", re.I))
for s in item_span_tags[:20]:
    style = s.get("style", "")
    cls = s.get("class", "")
    text = s.get_text(strip=True)[:100]
    print(f"  class='{cls}' style='{style[:60]}' text='{text}'")
    # Show parent and siblings
    parent = s.parent
    if parent:
        parent_style = parent.get("style", "")[:80]
        print(f"    parent=<{parent.name}> style='{parent_style}'")

# Look at the structure around "Item 1A. Risk Factors" specifically
print("\n=== HTML AROUND 'Item 1A. Risk Factors' ===")
target = soup.find("span", string=re.compile(r"Item\s*1A\.\s*Risk", re.I))
if target:
    # Show the target + a few siblings
    for i, sib in enumerate(target.next_siblings):
        if i > 5:
            break
        if hasattr(sib, "name") and sib.name:
            text = sib.get_text(strip=True)[:120]
            print(f"  sibling <{sib.name}> => '{text}'")
        elif str(sib).strip():
            print(f"  text node => '{str(sib).strip()[:80]}'")

    # Show the parent chain
    print("\n  Parent chain:")
    p = target.parent
    depth = 0
    while p and depth < 5:
        print(f"    <{p.name}> class='{p.get('class','')}' id='{p.get('id','')}'")
        p = p.parent
        depth += 1

# Check: does the <a href="#..."> point to anything?
print("\n=== DO <a href='#...'> LINKS POINT TO REAL IDS? ===")
for a in item_a_tags[:5]:
    href = a.get("href", "")
    if href.startswith("#"):
        target_id = href[1:]
        target_el = soup.find(attrs={"id": target_id})
        target_name = soup.find(attrs={"name": target_id})
        print(f"  href='{href}' => id exists: {target_el is not None}, name exists: {target_name is not None}")

# Check ix:resources / ix:header location (hidden XBRL context block)
print("\n=== ix:resources BLOCK ===")
ix_resources = soup.find("ix:resources")
if ix_resources:
    text_preview = ix_resources.get_text(strip=True)[:200]
    print(f"  Found! Text preview: '{text_preview}'")
    # Where is it in the document?
    parent = ix_resources.parent
    print(f"  parent=<{parent.name if parent else 'none'}>")

ix_header = soup.find("ix:header")
if ix_header:
    print(f"  ix:header found, text length: {len(ix_header.get_text())}")
