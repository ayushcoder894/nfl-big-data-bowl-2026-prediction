"""
Generate chunked string predictions for Kaggle notebook.
Splits predictions into small string chunks that are safe for notebook parsers.
"""
import pandas as pd
from pathlib import Path

# Configuration
CHUNK_SIZE = 100  # Number of values per chunk (safe for notebook parsers)

# Load predictions
script_dir = Path(__file__).parent

possible_files = [
    script_dir / "submission.csv",
    script_dir / "submission (10).csv",
    script_dir / "predictions.csv",
]

df = None
for filepath in possible_files:
    if filepath.exists():
        print(f"Loading predictions from: {filepath}")
        df = pd.read_csv(filepath)
        break

if df is None:
    raise FileNotFoundError("Could not find submission.csv or predictions.csv")

# Extract x and y predictions as lists
x_preds = df['x'].tolist()
y_preds = df['y'].tolist()

print(f"\nLoaded {len(x_preds)} predictions")
print(f"First x: {x_preds[0]}")
print(f"First y: {y_preds[0]}")
print(f"Chunk size: {CHUNK_SIZE} values per chunk")

# Function to create chunks
def create_chunks(values, chunk_size):
    """Split values into chunks and create string representation."""
    chunks = []
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i + chunk_size]
        chunk_str = ",".join(map(str, chunk))
        chunks.append(chunk_str)
    return chunks

# Create chunks for x and y
print("\nGenerating chunks...")
x_chunks = create_chunks(x_preds, CHUNK_SIZE)
y_chunks = create_chunks(y_preds, CHUNK_SIZE)

print(f"X predictions split into {len(x_chunks)} chunks")
print(f"Y predictions split into {len(y_chunks)} chunks")

# Save to output file
output_file = script_dir / "chunked_predictions.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Chunked Predictions for Kaggle Notebook\n")
    f.write("# =========================================\n")
    f.write(f"# Total predictions: {len(x_preds)}\n")
    f.write(f"# Chunk size: {CHUNK_SIZE} values per chunk\n")
    f.write(f"# Number of chunks: {len(x_chunks)}\n\n")
    
    f.write("# Copy the code below into your Kaggle notebook:\n")
    f.write("# " + "="*60 + "\n\n")
    
    # Write X chunks
    f.write("# X Predictions (chunked)\n")
    f.write("x_chunks = [\n")
    for i, chunk in enumerate(x_chunks):
        # Use repr() to properly escape the string
        f.write(f'    "{chunk}"')
        if i < len(x_chunks) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n\n")
    
    # Write Y chunks
    f.write("# Y Predictions (chunked)\n")
    f.write("y_chunks = [\n")
    for i, chunk in enumerate(y_chunks):
        f.write(f'    "{chunk}"')
        if i < len(y_chunks) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n\n")
    
    # Write parsing code
    f.write("# Parse chunks into lists\n")
    f.write('x_preds = [float(val) for chunk in x_chunks for val in chunk.split(",")]\n')
    f.write('y_preds = [float(val) for chunk in y_chunks for val in chunk.split(",")]\n\n')
    
    f.write("# Verify\n")
    f.write("print(f'Loaded {len(x_preds)} x predictions')\n")
    f.write("print(f'Loaded {len(y_preds)} y predictions')\n")
    f.write("print(f'First x: {x_preds[0]}')\n")
    f.write("print(f'First y: {y_preds[0]}')\n")

print(f"\n{'='*80}")
print(f"Chunked predictions saved to: {output_file}")
print("="*80)
print("\nYou can now copy the entire code from this file to your Kaggle notebook!")
print(f"The predictions are split into {len(x_chunks)} manageable chunks.")
