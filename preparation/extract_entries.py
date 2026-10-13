import pandas as pd
from pathlib import Path
import glob

def main():
    data_dir = Path('data')
    # Find all CSV files in the data directory (including subfolders)
    csv_files = data_dir.rglob('*.csv')
    for csv_path in csv_files:
        # Create a folder named after the CSV file (without extension) to hold its outputs
        base_name = csv_path.stem
        base_dir = data_dir / base_name
        orig_dir = base_dir / 'orig'
        plain_dir = base_dir / 'plain'
        orig_dir.mkdir(parents=True, exist_ok=True)
        plain_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_csv(csv_path, dtype=str)  # read everything as string to preserve content
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue
        for _, row in df.iterrows():
            entry_id = row.get('id')
            original = row.get('original')
            simplified = row.get('simplified')
            if not entry_id:
                continue
            # Write original text to the orig folder for this CSV
            orig_path = orig_dir / f"{entry_id}.txt"
            with open(orig_path, 'w', encoding='utf-8') as f:
                f.write(original or '')
            # Write simplified text to the plain folder for this CSV
            plain_path = plain_dir / f"{entry_id}.txt"
            with open(plain_path, 'w', encoding='utf-8') as f:
                f.write(simplified or '')
            print(f"Wrote {csv_path.name} -> orig:{orig_path} plain:{plain_path}")

if __name__ == '__main__':
    main()
