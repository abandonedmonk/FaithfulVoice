import json
import warnings
import sys
from pathlib import Path
from dataclasses import asdict

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chunker import chunk_filing

cleaned_dir = Path(__file__).parent.parent / "data" / "cleaned"
raw_dir = Path(__file__).parent.parent / "data" / "raw"
processed_dir = Path(__file__).parent.parent / "data" / "processed"
processed_dir.mkdir(exist_ok=True)

total_ok = 0
total_zero = 0
total_skip = 0

for f in sorted(cleaned_dir.glob("*_clean.htm")):
    raw_name = f.name.replace("_clean.htm", ".htm")
    raw_path = raw_dir / raw_name
    out_name = f.name.replace("_clean.htm", "_chunks.jsonl")
    out_file = processed_dir / out_name

    if out_file.exists() and out_file.stat().st_size > 100:
        total_skip += 1
        print(f"SKIP {f.name} (already exists)")
        continue

    print(f"Chunking {f.name}...", flush=True)
    try:
        chunks = chunk_filing(f, raw_path if raw_path.exists() else None)
        with open(out_file, "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        if len(chunks) == 0:
            total_zero += 1
            print(f"  ZERO chunks -> {out_name}")
        else:
            total_ok += 1
            print(f"  OK: {len(chunks)} chunks -> {out_name}")
    except Exception as e:
        total_zero += 1
        print(f"  ERROR: {e}")

print(f"\nDone: {total_ok} OK, {total_zero} zero/error, {total_skip} skipped")
