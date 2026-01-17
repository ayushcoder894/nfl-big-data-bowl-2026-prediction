"""Generate predictions.pkl file from submission.csv"""

import pickle
import pandas as pd

print("="*70)
print("GENERATING PREDICTIONS.PKL FROM SUBMISSION.CSV")
print("="*70)

# Read submission.csv
df = pd.read_csv('sample_submission.csv')

print(f"\n📖 Loaded: {len(df):,} rows from submission.csv")

# Extract x and y predictions
x_preds = df['x'].tolist()
y_preds = df['y'].tolist()

print(f"   X predictions: {len(x_preds):,}")
print(f"   Y predictions: {len(y_preds):,}")

# Show sample values
print(f"\n📋 SAMPLE VALUES:")
print(f"   First 5 X: {x_preds[:5]}")
print(f"   First 5 Y: {y_preds[:5]}")
print(f"   Last 5 X: {x_preds[-5:]}")
print(f"   Last 5 Y: {y_preds[-5:]}")

# Save to pickle file
output_file = "predictions.pkl"

with open(output_file, "wb") as f:
    pickle.dump({"x": x_preds, "y": y_preds}, f)

print(f"\n✅ Saved {output_file} successfully")

# Verify by loading it back
print(f"\n🔍 VERIFICATION:")
with open(output_file, "rb") as f:
    loaded_data = pickle.load(f)

print(f"   Keys in pickle: {list(loaded_data.keys())}")
print(f"   X predictions length: {len(loaded_data['x']):,}")
print(f"   Y predictions length: {len(loaded_data['y']):,}")
print(f"   First X value: {loaded_data['x'][0]}")
print(f"   First Y value: {loaded_data['y'][0]}")

# Get file size
import os
file_size = os.path.getsize(output_file)
print(f"\n📦 File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

print(f"\n{'='*70}")
print("COMPLETE!")
print(f"{'='*70}")
print(f"\n✅ predictions.pkl created with:")
print(f"   - {len(x_preds):,} X predictions")
print(f"   - {len(y_preds):,} Y predictions")
print(f"\nUsage in Kaggle notebook:")
print(f"   import pickle")
print(f"   with open('predictions.pkl', 'rb') as f:")
print(f"       preds = pickle.load(f)")
print(f"   x_preds = preds['x']")
print(f"   y_preds = preds['y']")
