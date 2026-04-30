from bs4 import BeautifulSoup
import re

with open(r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

# 1. Check for <a name=...> anchors
anchors_name = soup.find_all("a", attrs={"name": True})
print(f"=== ANCHORS (a name=) : {len(anchors_name)} ===")
for a in anchors_name[:30]:
    nm = a.get("name", "")
    txt = a.get_text(strip=True)[:80]
    print(f"  name='{nm}'  text='{txt}'")

# 2. Check for <a id=...> anchors
anchors_id = soup.find_all("a", attrs={"id": True})
print(f"\n=== ANCHORS (a id=) : {len(anchors_id)} ===")
for a in anchors_id[:30]:
    aid = a.get("id", "")
    txt = a.get_text(strip=True)[:80]
    print(f"  id='{aid}'  text='{txt}'")

# 3. Check for div ids that look like section markers
divs_with_id = soup.find_all("div", attrs={"id": True})
print(f"\n=== DIVS with id= : {len(divs_with_id)} ===")
for d in divs_with_id[:40]:
    id_val = d.get("id", "")
    if any(k in id_val.lower() for k in ["item", "risk", "mda", "business", "financi", "propert", "legal", "market"]):
        print(f"  id='{id_val}'")

# 4. Count iXBRL tags
ix_tags = soup.find_all(re.compile(r"^ix:"))
print(f"\n=== iXBRL TAGS : {len(ix_tags)} total ===")
tag_types = {}
for t in ix_tags:
    name = t.name
    tag_types[name] = tag_types.get(name, 0) + 1
for k, v in sorted(tag_types.items(), key=lambda x: -x[1])[:15]:
    print(f"  {k}: {v}")

# 5. Hidden divs (display:none)
hidden = soup.find_all("div", style=re.compile(r"display:\s*none", re.I))
print(f"\n=== HIDDEN DIVS (display:none) : {len(hidden)} ===")

# 6. Sample iXBRL tag
sample_ix = soup.find(re.compile(r"^ix:nonfraction"))
if sample_ix:
    print(f"\n=== SAMPLE ix:nonfraction tag ===")
    print(str(sample_ix)[:500])

# 7. Count tables
tables = soup.find_all("table")
print(f"\n=== TABLE TAGS : {len(tables)} ===")

# 8. Check h1-h6 headings
for level in range(1, 7):
    headings = soup.find_all(f"h{level}")
    if headings:
        print(f"\n=== H{level} TAGS : {len(headings)} ===")
        for h in headings[:10]:
            txt = h.get_text(strip=True)[:100]
            print(f"  '{txt}'")
