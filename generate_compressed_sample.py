"""
Generate compressed base64-encoded strings from sample_submission.csv.
These can be copied into a Kaggle notebook for efficient storage.
"""
import pandas as pd
import zlib
import base64
from pathlib import Path

# Load sample_submission.csv
script_dir = Path(__file__).parent
sample_file = script_dir / "sample_submission.csv"

if not sample_file.exists():
    raise FileNotFoundError(f"Could not find {sample_file}")

print(f"Loading data from: {sample_file}")
df = pd.read_csv(sample_file)

# Extract x and y predictions as lists
x_preds = df['x'].tolist()
y_preds = df['y'].tolist()

print(f"\nLoaded {len(x_preds)} predictions from sample_submission.csv")
print(f"First x: {x_preds[0]}")
print(f"First y: {y_preds[0]}")

# Compress x_preds
print("\n" + "="*80)
print("COMPRESSING X_PREDS (SAMPLE)...")
print("="*80)
s_x = ",".join(map(str, x_preds))
compressed_x = base64.b64encode(zlib.compress(s_x.encode())).decode()
print(f"\nOriginal x_preds string size: {len(s_x):,} bytes")
print(f"Compressed x_preds size: {len(compressed_x):,} bytes")
print(f"Compression ratio: {len(compressed_x) / len(s_x) * 100:.2f}%")
print(f"\nCompressed x_preds string:\n")
print(compressed_x)

# Compress y_preds
print("\n" + "="*80)
print("COMPRESSING Y_PREDS (SAMPLE)...")
print("="*80)
s_y = ",".join(map(str, y_preds))
compressed_y = base64.b64encode(zlib.compress(s_y.encode())).decode()
print(f"\nOriginal y_preds string size: {len(s_y):,} bytes")
print(f"Compressed y_preds size: {len(compressed_y):,} bytes")
print(f"Compression ratio: {len(compressed_y) / len(s_y) * 100:.2f}%")
print(f"\nCompressed y_preds string:\n")
print(compressed_y)

# Save to a text file for easy copying
output_file = script_dir / "compressed_sample_submission.txt"
with open(output_file, 'w') as f:
    f.write("# Compressed X Predictions (from sample_submission.csv)\n")
    f.write("# Copy the string below and use in your Kaggle notebook\n")
    f.write("# To decompress: x_preds = list(map(float, zlib.decompress(base64.b64decode(compressed_x)).decode().split(',')))\n\n")
    f.write("compressed_x = '''\n")
    f.write(compressed_x)
    f.write("\n'''\n\n")
    
    f.write("# Compressed Y Predictions (from sample_submission.csv)\n")
    f.write("# Copy the string below and use in your Kaggle notebook\n")
    f.write("# To decompress: y_preds = list(map(float, zlib.decompress(base64.b64decode(compressed_y)).decode().split(',')))\n\n")
    f.write("compressed_y = '''\n")
    f.write(compressed_y)
    f.write("\n'''\n\n")
    
    f.write("# Complete Kaggle Notebook Code (SAMPLE SUBMISSION)\n")
    f.write("# ================================================\n\n")
    f.write("import zlib, base64\n\n")
    f.write("# Paste compressed strings (from sample_submission.csv)\n")
    f.write(f"compressed_x = '{compressed_x}'\n")
    f.write(f"compressed_y = '{compressed_y}'\n\n")
    f.write("# Decompress\n")
    f.write("x_preds = list(map(float, zlib.decompress(base64.b64decode(compressed_x)).decode().split(',')))\n")
    f.write("y_preds = list(map(float, zlib.decompress(base64.b64decode(compressed_y)).decode().split(',')))\n\n")
    f.write(f"print(f'Loaded {{len(x_preds)}} predictions from sample_submission.csv')\n")
    f.write(f"print(f'First x: {{x_preds[0]}}')\n")
    f.write(f"print(f'First y: {{y_preds[0]}}')\n\n")
    f.write("# Now you can use x_preds and y_preds to create your submission\n")
    f.write("# Example:\n")
    f.write("# submission = pd.DataFrame({\n")
    f.write("#     'id': test_ids,  # your test IDs\n")
    f.write("#     'x': x_preds,\n")
    f.write("#     'y': y_preds\n")
    f.write("# })\n")

print(f"\n" + "="*80)
print(f"Compressed strings saved to: {output_file}")
print("="*80)
print("\nYou can now copy the compressed strings from this file to your Kaggle notebook!")
print("These are the baseline predictions from sample_submission.csv")
