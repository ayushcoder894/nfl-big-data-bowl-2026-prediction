"""
Kaggle Notebook Code - Read from Uploaded Dataset

INSTRUCTIONS:
1. Upload your submission.csv as a Kaggle Dataset
2. Add the dataset to this notebook
3. Update the path below to match your dataset
4. Run this cell
"""

import pandas as pd
from pathlib import Path

# UPDATE THIS PATH to match your uploaded dataset
# Example: /kaggle/input/my-nfl-predictions/submission.csv
input_path = Path("/kaggle/input/YOUR-DATASET-NAME/submission.csv")

# Read the uploaded submission
df = pd.read_csv(input_path)

# Save to Kaggle working directory (required for submission)
output_path = Path("/kaggle/working/submission.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False, lineterminator="\n", encoding="utf-8")

print(f"✅ submission.csv created with {len(df):,} rows")
print(f"   Source: {input_path}")
print(f"   Output: {output_path}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Ready for submission!")
