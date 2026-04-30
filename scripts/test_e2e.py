import sys, json
sys.path.insert(0, r"D:\Code\PROJECTS\FaithfulVoice")

from src.cleaner import clean_edgar_html
from src.chunker import chunk_filing
from dataclasses import asdict

raw_path = r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K.htm"
cleaned_path = clean_edgar_html(raw_path)
print(f"1. Cleaned: {cleaned_path}")

chunks = chunk_filing(cleaned_path, raw_path)
print(f"2. Total chunks: {len(chunks)}")

type_counts = {}
for c in chunks:
    type_counts[c.element_type] = type_counts.get(c.element_type, 0) + 1
print(f"3. Chunk types: {type_counts}")

section_counts = {}
for c in chunks:
    key = f"{c.anchor_id}: {c.section}"
    section_counts[key] = section_counts.get(key, 0) + 1
print(f"\n4. Chunks per section:")
for s, cnt in section_counts.items():
    print(f"   {s}: {cnt}")

print(f"\n5. Sample NarrativeText chunk (item1a):")
for c in chunks:
    if c.anchor_id == "item1a" and c.element_type == "NarrativeText":
        safe = c.text[:150].encode("ascii", errors="replace").decode("ascii")
        print(f"   anchor={c.anchor_id} ticker={c.ticker} year={c.year}")
        print(f"   text: {safe}")
        break

print(f"\n6. Sample Table chunk:")
for c in chunks:
    if c.element_type == "Table":
        safe = c.text[:150].encode("ascii", errors="replace").decode("ascii")
        md_preview = c.table_markdown[:150].encode("ascii", errors="replace").decode("ascii") if c.table_markdown else "(no markdown)"
        print(f"   anchor={c.anchor_id} section={c.section}")
        print(f"   text: {safe}")
        print(f"   markdown: {md_preview}")
        break

print(f"\n7. Metadata completeness check:")
sample = chunks[0]
fields = ["company", "ticker", "cik", "year", "filing_type", "section", "anchor_id", "element_type", "chunk_id", "parent_chunk_id", "htm_filename"]
for f in fields:
    val = getattr(sample, f, "MISSING")
    print(f"   {f}: {val}")

print(f"\n8. Writing to data/processed/chunks.jsonl...")
import os
os.makedirs(r"D:\Code\PROJECTS\FaithfulVoice\data\processed", exist_ok=True)
out_path = r"D:\Code\PROJECTS\FaithfulVoice\data\processed\NVDA_2024_10K_chunks.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
print(f"   Written {len(chunks)} chunks to {out_path}")
print(f"   File size: {os.path.getsize(out_path):,} bytes")
