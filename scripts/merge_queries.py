import json
from pathlib import Path


def merge_jsonl_files(input_folder: str | Path, output_file: str | Path) -> None:
    input_path = Path(input_folder)
    output_path = Path(output_file)

    jsonl_files = sorted(input_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {input_folder}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to merge")

    total_records = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file.name}")
            try:
                with open(jsonl_file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        line = line.strip()
                        if line:
                            outfile.write(line + "\n")
                            total_records += 1
            except Exception as e:
                print(f"Error reading {jsonl_file.name}: {e}")

    print(f"\nMerge complete!")
    print(f"Total records merged: {total_records}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    queries_folder = project_root / "data" / "queries"
    output_file = project_root / "data" / "queries_full_dataset" / "merged_queries.jsonl"

    merge_jsonl_files(queries_folder, output_file)
