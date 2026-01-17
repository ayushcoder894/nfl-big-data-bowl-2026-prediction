"""Compare two submission CSV files."""

import pandas as pd
import numpy as np

# Read both files
df1 = pd.read_csv('submission.csv')
df2 = pd.read_csv('submission (5).csv')

print("=" * 60)
print("FILE COMPARISON: submission.csv vs submission (5).csv")
print("=" * 60)

print("\n=== submission.csv ===")
print(f"Rows: {len(df1):,}")
print(f"Columns: {df1.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df1.head())
print(f"\nData types:")
print(df1.dtypes)

print("\n=== submission (5).csv ===")
print(f"Rows: {len(df2):,}")
print(f"Columns: {df2.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df2.head())
print(f"\nData types:")
print(df2.dtypes)

print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

print(f"\n✓ Same number of rows: {len(df1) == len(df2)} ({len(df1):,} vs {len(df2):,})")
print(f"✓ Same columns: {df1.columns.tolist() == df2.columns.tolist()}")

# Check IDs
ids_match = df1['id'].equals(df2['id'])
print(f"✓ IDs match exactly: {ids_match}")
if not ids_match:
    print(f"  - IDs in df1 but not df2: {len(set(df1['id']) - set(df2['id']))}")
    print(f"  - IDs in df2 but not df1: {len(set(df2['id']) - set(df1['id']))}")

# Check X values
x_match = df1['x'].equals(df2['x'])
print(f"✓ X values match exactly: {x_match}")
if not x_match:
    x_diff = (df1['x'] - df2['x']).abs()
    print(f"  - Max difference: {x_diff.max()}")
    print(f"  - Mean difference: {x_diff.mean()}")
    print(f"  - Rows with differences: {(x_diff > 0).sum()}")

# Check Y values
y_match = df1['y'].equals(df2['y'])
print(f"✓ Y values match exactly: {y_match}")
if not y_match:
    y_diff = (df1['y'] - df2['y']).abs()
    print(f"  - Max difference: {y_diff.max()}")
    print(f"  - Mean difference: {y_diff.mean()}")
    print(f"  - Rows with differences: {(y_diff > 0).sum()}")

# Overall comparison
files_identical = df1.equals(df2)
print(f"\n{'='*60}")
print(f"✓ FILES ARE IDENTICAL: {files_identical}")
print(f"{'='*60}")

if not files_identical:
    print("\n=== DIFFERENCES FOUND ===")
    # Find rows with differences
    merged = df1.merge(df2, on='id', suffixes=('_1', '_2'), how='outer', indicator=True)
    
    # Check for missing IDs
    only_in_1 = merged[merged['_merge'] == 'left_only']
    only_in_2 = merged[merged['_merge'] == 'right_only']
    
    if len(only_in_1) > 0:
        print(f"\nIDs only in submission.csv: {len(only_in_1)}")
        print(only_in_1.head())
    
    if len(only_in_2) > 0:
        print(f"\nIDs only in submission (5).csv: {len(only_in_2)}")
        print(only_in_2.head())
    
    # Check for value differences in matching IDs
    both = merged[merged['_merge'] == 'both'].copy()
    both['x_diff'] = (both['x_1'] - both['x_2']).abs()
    both['y_diff'] = (both['y_1'] - both['y_2']).abs()
    
    diff_rows = both[(both['x_diff'] > 0) | (both['y_diff'] > 0)]
    if len(diff_rows) > 0:
        print(f"\n\nRows with different x or y values: {len(diff_rows)}")
        print("\nTop 10 differences:")
        print(diff_rows.nlargest(10, 'x_diff')[['id', 'x_1', 'x_2', 'x_diff', 'y_1', 'y_2', 'y_diff']])
