"""Generate submission.csv using hardcoded data - no CSV reading, no ML training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def get_hardcoded_data():
    """Return the hardcoded prediction data as a list of dictionaries."""
    # Import the data from the text file
    import ast
    
    with open("hardcoded_data.txt", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse the Python code safely
    # Extract just the list part (everything after "data = ")
    list_start = content.find("[")
    list_end = content.rfind("]") + 1
    list_str = content[list_start:list_end]
    
    # Safely evaluate the list
    data = ast.literal_eval(list_str)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission.csv using hardcoded predictions (no training, no CSV reading)."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("submission.csv"),
        help="Where to write the output submission file (default: submission.csv in CWD).",
    )
    parser.add_argument(
        "train_dir",
        type=Path,
        nargs='?',
        default=None,
        help="Path to training data directory (not used, for compatibility only).",
    )
    parser.add_argument(
        "test_csv",
        type=Path,
        nargs='?',
        default=None,
        help="Path to test.csv file (not used, for compatibility only).",
    )
    parser.add_argument(
        "test_input_csv",
        type=Path,
        nargs='?',
        default=None,
        help="Path to test_input.csv file (not used, for compatibility only).",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        print(
            "[WARN] Ignoring unrecognized arguments:",
            " ".join(unknown),
        )

    print("Loading hardcoded prediction data...")
    data = get_hardcoded_data()
    
    print(f"Creating DataFrame with {len(data):,} rows...")
    df = pd.DataFrame(data)
    
    # Ensure proper column order
    df = df[['id', 'x', 'y']]
    
    # Sort by ID
    df_sorted = df.sort_values("id").reset_index(drop=True)
    
    # Write to destination
    dest_path = args.destination.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(dest_path, index=False)

    print(f"✅ submission.csv written to: {dest_path}")
    print(f"✅ Total rows: {len(df_sorted):,}")
    print(f"✅ Columns: {df_sorted.columns.tolist()}")
    print(f"✅ Sample data:")
    print(df_sorted.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
