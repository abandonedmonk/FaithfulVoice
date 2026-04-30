import sys, os
sys.path.insert(0, r"D:\Code\PROJECTS\FaithfulVoice")

from src.cleaner import clean_edgar_html
from src.parser import parse_filing

raw_path = r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm"
cleaned_path = clean_edgar_html(raw_path)
print(f"Cleaned: {cleaned_path}")

parsed = parse_filing(cleaned_path)
print(f"\nTotal parsed elements: {len(parsed)}")

type_counts = {}
for p in parsed:
    type_counts[p.element_type] = type_counts.get(p.element_type, 0) + 1
print("\nElement type counts:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

section_counts = {}
for p in parsed:
    key = f"{p.anchor_id}: {p.section}"
    section_counts[key] = section_counts.get(key, 0) + 1
print("\nElements per section:")
for s, c in section_counts.items():
    print(f"  {s}: {c}")

print("\nFirst 20 parsed elements:")
for i, p in enumerate(parsed[:20]):
    safe = p.text[:80].encode("ascii", errors="replace").decode("ascii")
    print(f"  [{i}] <{p.element_type}> [{p.anchor_id}] {safe}")

print("\nSample from Item 1A (Risk Factors):")
risk_elements = [p for p in parsed if p.anchor_id == "item1a"]
for p in risk_elements[:5]:
    safe = p.text[:100].encode("ascii", errors="replace").decode("ascii")
    print(f"  <{p.element_type}> {safe}")
