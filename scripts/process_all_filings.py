import json
import sys
import argparse
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cleaner import clean_edgar_html
from src.chunker import chunk_filing


def process_filing(raw_path: Path, processed_dir: Path) -> int | None:
    stem = raw_path.stem
    out_file = processed_dir / f"{stem}_chunks.jsonl"

    if out_file.exists() and out_file.stat().st_size > 100:
        print(f"  SKIP: {out_file.name} already exists")
        return None

    print(f"  Cleaning {raw_path.name}...")
    cleaned_path = clean_edgar_html(raw_path)

    print(f"  Chunking...")
    chunks = chunk_filing(cleaned_path, raw_path)

    if not chunks:
        print(f"  WARNING: 0 chunks produced from {raw_path.name}")
        return 0

    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"  -> {len(chunks)} chunks -> {out_file.name}")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Batch process raw SEC filings into chunk JSONL")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw HTM filings")
    parser.add_argument("--processed-dir", default="data/processed", help="Output directory for JSONL")
    parser.add_argument("--pattern", default="*.htm", help="Glob pattern for raw files")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    raw_dir = project_root / args.raw_dir
    processed_dir = project_root / args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    htm_files = sorted(raw_dir.glob(args.pattern))
    if not htm_files:
        print(f"No files matching '{args.pattern}' in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(htm_files)} filing(s) in {raw_dir}\n")

    if args.force:
        for f in htm_files:
            out_file = processed_dir / f"{f.stem}_chunks.jsonl"
            if out_file.exists():
                out_file.unlink()

    total_chunks = 0
    processed = 0
    skipped = 0

    for i, htm_file in enumerate(htm_files, 1):
        print(f"[{i}/{len(htm_files)}] {htm_file.name}")
        result = process_filing(htm_file, processed_dir)
        if result is None:
            skipped += 1
        else:
            total_chunks += result
            processed += 1

    print(f"\n{'='*50}")
    print(f"  Processed: {processed} filings")
    print(f"  Skipped:   {skipped} (already exist)")
    print(f"  Total chunks: {total_chunks}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
