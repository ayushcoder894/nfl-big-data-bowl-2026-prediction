"""
Generate chunked predictions for sample_submission.csv
Splits predictions into small string chunks for reliable Kaggle notebook parsing
"""

import pandas as pd
from pathlib import Path

# Configuration
CHUNK_SIZE = 100  # Number of values per chunk
INPUT_FILE = "sample_submission.csv"

def create_chunks(values, chunk_size):
    """Split values into chunks of specified size."""
    chunks = []
    for i in range(0, len(values), chunk_size):
        chunk = values[i:i + chunk_size]
        chunk_str = ",".join(map(str, chunk))
        chunks.append(chunk_str)
    return chunks

def main():
    # Get the script directory
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_FILE
    output_path = script_dir / "chunked_sample_submission.txt"
    
    print(f"Loading predictions from: {input_path}\n")
    
    # Load the submission file
    df = pd.read_csv(input_path)
    
    # Extract predictions
    x_preds = df['x'].tolist()
    y_preds = df['y'].tolist()
    
    print(f"Loaded {len(x_preds)} predictions")
    print(f"First x: {x_preds[0]}, First y: {y_preds[0]}")
    print(f"Chunk size: {CHUNK_SIZE} values per chunk\n")
    
    # Create chunks
    print("Generating chunks...")
    x_chunks = create_chunks(x_preds, CHUNK_SIZE)
    y_chunks = create_chunks(y_preds, CHUNK_SIZE)
    
    print(f"X predictions split into {len(x_chunks)} chunks")
    print(f"Y predictions split into {len(y_chunks)} chunks\n")
    
    # Generate output file with Python code
    with open(output_path, 'w') as f:
        f.write("# Chunked Sample Submission Predictions (Baseline - All Zeros)\n")
        f.write("# " + "=" * 76 + "\n")
        f.write(f"# Total predictions: {len(x_preds)}\n")
        f.write(f"# Chunk size: {CHUNK_SIZE} values per chunk\n")
        f.write(f"# Number of chunks: {len(x_chunks)}\n\n")
        
        f.write("# Copy the code below into your Kaggle notebook:\n")
        f.write("# " + "=" * 60 + "\n\n")
        
        # X predictions
        f.write("# X Predictions (chunked)\n")
        f.write("x_chunks = [\n")
        for i, chunk in enumerate(x_chunks):
            f.write(f'    "{chunk}"')
            if i < len(x_chunks) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n\n")
        
        # Y predictions
        f.write("# Y Predictions (chunked)\n")
        f.write("y_chunks = [\n")
        for i, chunk in enumerate(y_chunks):
            f.write(f'    "{chunk}"')
            if i < len(y_chunks) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n\n")
        
        # Parsing code
        f.write("# Parse chunks back into lists\n")
        f.write("x_preds = [float(val) for chunk in x_chunks for val in chunk.split(\",\")]\n")
        f.write("y_preds = [float(val) for chunk in y_chunks for val in chunk.split(\",\")]\n\n")
        
        f.write(f"# Verify: x_preds should have {len(x_preds)} values\n")
        f.write(f"# Verify: y_preds should have {len(y_preds)} values\n")
        f.write("print(f'Loaded {len(x_preds)} x predictions')\n")
        f.write("print(f'Loaded {len(y_preds)} y predictions')\n")
        f.write("print(f'First x: {x_preds[0]}, First y: {y_preds[0]}')\n")
    
    print("=" * 80)
    print(f"Chunked sample predictions saved to: {output_path}")
    print("=" * 80)
    print("\nYou can now copy the entire code from this file to your Kaggle notebook!")
    print(f"The predictions are split into {len(x_chunks)} manageable chunks.")

if __name__ == "__main__":
    main()
