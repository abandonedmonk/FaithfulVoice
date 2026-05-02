import json
import os
from pathlib import Path

def merge_jsonl_files(input_folder, output_file):
    """
    Merge all JSONL files from the input folder into a single JSONL file.
    
    Args:
        input_folder (str): Path to the folder containing JSONL files
        output_file (str): Path to the output merged JSONL file
    """
    input_path = Path(input_folder)
    output_path = Path(output_file)
    
    # Get all JSONL files from the input folder
    jsonl_files = sorted(input_path.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_folder}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to merge")
    
    total_records = 0
    
    # Merge all JSONL files
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file.name}")
            
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            outfile.write(line + '\n')
                            total_records += 1
            except Exception as e:
                print(f"Error reading {jsonl_file.name}: {e}")
    
    print(f"\nMerge complete!")
    print(f"Total records merged: {total_records}")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    queries_folder = script_dir.parent / "queries"
    output_file = script_dir / "merged_queries.jsonl"
    
    # Merge the files
    merge_jsonl_files(queries_folder, output_file)
