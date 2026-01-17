"""Check data format and types in submission files."""

import pandas as pd
import numpy as np

def check_file(filepath):
    print(f"\n{'='*70}")
    print(f"CHECKING: {filepath}")
    print(f"{'='*70}")
    
    df = pd.read_csv(filepath)
    
    # Basic info
    print(f"\n📊 BASIC INFO:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Data types
    print(f"\n📝 DATA TYPES:")
    for col in df.columns:
        print(f"   {col}: {df[col].dtype}")
    
    # ID column details
    print(f"\n🆔 ID COLUMN ANALYSIS:")
    print(f"   Type: {df['id'].dtype}")
    print(f"   Unique IDs: {df['id'].nunique():,}")
    print(f"   Any duplicates: {df['id'].duplicated().any()}")
    print(f"   Any nulls: {df['id'].isna().any()}")
    print(f"   Sample IDs:")
    for i, id_val in enumerate(df['id'].head(3)):
        print(f"      [{i}] '{id_val}' (type: {type(id_val).__name__})")
    
    # Check for decimal points in IDs
    has_decimal = df['id'].astype(str).str.contains(r'\.').any()
    print(f"   Contains decimal points: {has_decimal}")
    
    # ID format validation
    import re
    pattern = r'^(\d+)_(\d+)_(\d+)_(\d+)$'
    valid_format = df['id'].astype(str).str.match(pattern)
    print(f"   Valid format (game_play_nfl_frame): {valid_format.all()}")
    if not valid_format.all():
        print(f"   Invalid IDs count: {(~valid_format).sum()}")
        print(f"   First invalid: {df.loc[~valid_format, 'id'].head(1).values}")
    
    # X column details
    print(f"\n📍 X COLUMN ANALYSIS:")
    print(f"   Type: {df['x'].dtype}")
    print(f"   Min: {df['x'].min():.10f}")
    print(f"   Max: {df['x'].max():.10f}")
    print(f"   Mean: {df['x'].mean():.10f}")
    print(f"   Std: {df['x'].std():.10f}")
    print(f"   Any nulls: {df['x'].isna().any()}")
    print(f"   Any inf: {np.isinf(df['x']).any()}")
    print(f"   In valid range [0, 120]: {df['x'].between(0, 120).all()}")
    print(f"   Decimal places (sample):")
    for i, val in enumerate(df['x'].head(3)):
        print(f"      [{i}] {val:.17f}")
    
    # Y column details
    print(f"\n📍 Y COLUMN ANALYSIS:")
    print(f"   Type: {df['y'].dtype}")
    print(f"   Min: {df['y'].min():.10f}")
    print(f"   Max: {df['y'].max():.10f}")
    print(f"   Mean: {df['y'].mean():.10f}")
    print(f"   Std: {df['y'].std():.10f}")
    print(f"   Any nulls: {df['y'].isna().any()}")
    print(f"   Any inf: {np.isinf(df['y']).any()}")
    print(f"   In valid range [0, 53.3]: {df['y'].between(0, 53.3).all()}")
    print(f"   Decimal places (sample):")
    for i, val in enumerate(df['y'].head(3)):
        print(f"      [{i}] {val:.17f}")
    
    # Missing values
    print(f"\n❓ MISSING VALUES:")
    for col in df.columns:
        missing = df[col].isna().sum()
        print(f"   {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
    # Sample rows
    print(f"\n📋 FIRST 5 ROWS:")
    print(df.head().to_string(index=True))
    
    print(f"\n📋 LAST 5 ROWS:")
    print(df.tail().to_string(index=True))
    
    return df

# Check both files
print("\n" + "="*70)
print("SUBMISSION FILES DATA FORMAT AND TYPE ANALYSIS")
print("="*70)

df1 = check_file('submission.csv')
df2 = check_file('submission (5).csv')

# Cross-comparison
print(f"\n{'='*70}")
print("CROSS-COMPARISON")
print(f"{'='*70}")

print(f"\n🔄 STRUCTURE COMPARISON:")
print(f"   Same shape: {df1.shape == df2.shape}")
print(f"   Same dtypes: {(df1.dtypes == df2.dtypes).all()}")
print(f"   Same column names: {df1.columns.tolist() == df2.columns.tolist()}")

print(f"\n🔄 DATA COMPARISON:")
print(f"   IDs identical: {df1['id'].equals(df2['id'])}")
print(f"   X values identical: {df1['x'].equals(df2['x'])}")
print(f"   Y values identical: {df1['y'].equals(df2['y'])}")

if not df1['x'].equals(df2['x']):
    x_diff = (df1['x'] - df2['x']).abs()
    print(f"\n   X differences:")
    print(f"      Max diff: {x_diff.max():.17e}")
    print(f"      Mean diff: {x_diff.mean():.17e}")
    print(f"      Median diff: {x_diff.median():.17e}")
    print(f"      Rows with diff: {(x_diff > 0).sum()} / {len(df1)}")

if not df1['y'].equals(df2['y']):
    y_diff = (df1['y'] - df2['y']).abs()
    print(f"\n   Y differences:")
    print(f"      Max diff: {y_diff.max():.17e}")
    print(f"      Mean diff: {y_diff.mean():.17e}")
    print(f"      Median diff: {y_diff.median():.17e}")
    print(f"      Rows with diff: {(y_diff > 0).sum()} / {len(df1)}")

print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*70}")
