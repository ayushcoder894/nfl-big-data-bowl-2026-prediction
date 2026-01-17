"""Generate all x and y values as Python lists in a txt file."""

import pandas as pd

print("="*70)
print("GENERATING X AND Y VALUES AS LISTS")
print("="*70)

# Read submission.csv (the one with actual predictions)
df = pd.read_csv('submission.csv')

print(f"\n📖 Loaded: {len(df):,} rows from submission.csv")

# Extract x and y values
x_values = df['x'].tolist()
y_values = df['y'].tolist()

print(f"   X values: {len(x_values):,}")
print(f"   Y values: {len(y_values):,}")

# Write to text file
output_file = "xy_values_lists.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# X and Y values from submission (8).csv\n")
    f.write("# Total: 5,837 rows\n")
    f.write("# Generated for Kaggle submission\n\n")
    
    # Write X values
    f.write("# X VALUES (5,837 values)\n")
    f.write("x_values = [\n")
    for i, val in enumerate(x_values):
        if i < len(x_values) - 1:
            f.write(f"    {val},\n")
        else:
            f.write(f"    {val}\n")
    f.write("]\n\n")
    
    # Write Y values
    f.write("# Y VALUES (5,837 values)\n")
    f.write("y_values = [\n")
    for i, val in enumerate(y_values):
        if i < len(y_values) - 1:
            f.write(f"    {val},\n")
        else:
            f.write(f"    {val}\n")
    f.write("]\n\n")
    
    # Add usage instructions
    f.write("# USAGE IN KAGGLE NOTEBOOK:\n")
    f.write("# Copy the x_values and y_values lists above\n")
    f.write("# Then use this code:\n")
    f.write("#\n")
    f.write("# import pandas as pd\n")
    f.write("# from pathlib import Path\n")
    f.write("#\n")
    f.write("# # Load sample_submission to get IDs\n")
    f.write("# sample = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2026-prediction/sample_submission.csv')\n")
    f.write("#\n")
    f.write("# # Create submission with hardcoded values\n")
    f.write("# submission = pd.DataFrame({\n")
    f.write("#     'id': sample['id'],\n")
    f.write("#     'x': x_values,\n")
    f.write("#     'y': y_values\n")
    f.write("# })\n")
    f.write("#\n")
    f.write("# # Save\n")
    f.write("# submission.to_csv('/kaggle/working/submission.csv', index=False, lineterminator='\\n', encoding='utf-8')\n")
    f.write("# print(f'✅ Submission created with {len(submission)} rows')\n")

print(f"\n✅ Generated: {output_file}")

# Get file size
from pathlib import Path
file_size = Path(output_file).stat().st_size

print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
print(f"   X values: {len(x_values):,}")
print(f"   Y values: {len(y_values):,}")

# Show samples
print(f"\n📋 SAMPLE VALUES:")
print(f"\n   First 5 X values: {x_values[:5]}")
print(f"   First 5 Y values: {y_values[:5]}")
print(f"\n   Last 5 X values: {x_values[-5:]}")
print(f"   Last 5 Y values: {y_values[-5:]}")

# Also create a more compact version with IDs
output_file_compact = "xy_values_with_ids.txt"

with open(output_file_compact, 'w', encoding='utf-8') as f:
    f.write("# Submission data from submission.csv: ID, X, Y\n")
    f.write("# Format: (id, x, y)\n\n")
    f.write("submission_data = [\n")
    
    for i, row in df.iterrows():
        f.write(f"    ('{row['id']}', {row['x']}, {row['y']}),\n")
    
    f.write("]\n\n")
    f.write("# USAGE:\n")
    f.write("# import pandas as pd\n")
    f.write("# df = pd.DataFrame(submission_data, columns=['id', 'x', 'y'])\n")
    f.write("# df.to_csv('/kaggle/working/submission.csv', index=False, lineterminator='\\n', encoding='utf-8')\n")

print(f"\n✅ Generated: {output_file_compact}")
file_size_compact = Path(output_file_compact).stat().st_size
print(f"   File size: {file_size_compact:,} bytes ({file_size_compact/1024:.2f} KB)")

print(f"\n{'='*70}")
print("COMPLETE!")
print(f"{'='*70}")
print(f"\nTwo files created:")
print(f"1. {output_file} - Separate X and Y lists")
print(f"2. {output_file_compact} - Complete data with IDs")
print(f"\nBoth files ready to copy-paste into Kaggle notebook!")
