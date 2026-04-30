import sys, os
sys.path.insert(0, r"D:\Code\PROJECTS\FaithfulVoice")
os.environ["PYTHONIOENCODING"] = "utf-8"

from unstructured.partition.html import partition_html

cleaned_path = r"D:\Code\PROJECTS\FaithfulVoice\data\raw\NVDA_2024_10K_clean.htm"
print("Running partition_html...")
elements = partition_html(
    filename=cleaned_path,
    skip_headers_and_footers=True,
    include_metadata=True,
)
print(f"Total elements: {len(elements)}")

type_counts = {}
for el in elements:
    t = type(el).__name__
    type_counts[t] = type_counts.get(t, 0) + 1

print("\nElement type counts:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

print("\nFirst 15 elements:")
for i, el in enumerate(elements[:15]):
    t = type(el).__name__
    text = str(el)[:120].replace("\n", " ")
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(f"  [{i}] {t}: '{safe}'")

print("\nSample Title elements:")
titles = [el for el in elements if type(el).__name__ == "Title"]
for t in titles[:20]:
    text = str(t)[:100].replace("\n", " ")
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(f"  '{safe}'")

print("\nSample Table elements:")
tables = [el for el in elements if type(el).__name__ == "Table"]
print(f"  Total tables: {len(tables)}")
for tbl in tables[:3]:
    text = str(tbl)[:200].replace("\n", " ")
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(f"  '{safe}'")
