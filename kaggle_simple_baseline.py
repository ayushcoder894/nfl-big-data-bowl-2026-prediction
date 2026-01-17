"""
ALTERNATIVE APPROACH: Generate submission.csv directly in Kaggle
This version creates the CSV without hardcoding data - uses computation instead
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("GENERATING SUBMISSION.CSV FROM SCRATCH")
print("="*70)

# Strategy: Instead of hardcoding all values, we'll read from the sample
# and generate predictions using a simple baseline approach

# Read sample submission to get the required IDs
sample_path = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/sample_submission.csv")
test_path = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/test.csv")
test_input_path = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/test_input.csv")

print("\n📖 Reading competition data...")

try:
    sample = pd.read_csv(sample_path)
    print(f"   ✅ sample_submission.csv: {len(sample):,} rows")
except Exception as e:
    print(f"   ❌ Failed to read sample_submission.csv: {e}")
    # Fallback: create IDs manually
    print("   Creating IDs from test.csv...")
    test = pd.read_csv(test_path)
    sample = test.copy()
    sample['id'] = (sample['game_id'].astype(str) + '_' + 
                    sample['play_id'].astype(str) + '_' + 
                    sample['nfl_id'].astype(str) + '_' + 
                    sample['frame_id'].astype(str))
    sample = sample[['id']].drop_duplicates()
    print(f"   Created {len(sample):,} IDs from test.csv")

try:
    test_input = pd.read_csv(test_input_path)
    print(f"   ✅ test_input.csv: {len(test_input):,} rows")
except Exception as e:
    print(f"   ❌ Failed to read test_input.csv: {e}")
    test_input = None

print("\n🔧 Generating predictions...")

# Create submission DataFrame
submission = sample[['id']].copy()

# APPROACH 1: Use last known position from test_input
if test_input is not None:
    # Create ID in test_input
    test_input['id'] = (test_input['game_id'].astype(str) + '_' + 
                        test_input['play_id'].astype(str) + '_' + 
                        test_input['nfl_id'].astype(str) + '_' + 
                        test_input['frame_id'].astype(str))
    
    # Get the last known position for each prediction
    # Merge to get x, y from test_input
    test_input_subset = test_input[['id', 'x', 'y']].copy()
    
    # Merge
    submission = submission.merge(test_input_subset, on='id', how='left')
    
    # Fill any missing values with field center
    submission['x'].fillna(60.0, inplace=True)
    submission['y'].fillna(26.65, inplace=True)
    
    print(f"   ✅ Used last known positions from test_input")
else:
    # APPROACH 2: Use field center as baseline
    submission['x'] = 60.0  # Middle of field
    submission['y'] = 26.65  # Middle of field width
    print(f"   ✅ Used field center as baseline prediction")

# Ensure proper order (same as sample_submission)
submission = submission[['id', 'x', 'y']]

print(f"\n📊 Submission created:")
print(f"   Rows: {len(submission):,}")
print(f"   Columns: {submission.columns.tolist()}")
print(f"   Missing values: {submission.isna().sum().sum()}")

print(f"\n💾 Saving to /kaggle/working/submission.csv...")

output_path = Path("/kaggle/working/submission.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

submission.to_csv(output_path, index=False, lineterminator='\n', encoding='utf-8')

print(f"   ✅ Saved successfully!")
print(f"   File: {output_path}")
print(f"   Size: {output_path.stat().st_size:,} bytes")

print(f"\n📋 First 5 rows:")
print(submission.head())

print(f"\n📋 Last 5 rows:")
print(submission.tail())

print("\n" + "="*70)
print("✅ SUBMISSION READY FOR KAGGLE!")
print("="*70)
