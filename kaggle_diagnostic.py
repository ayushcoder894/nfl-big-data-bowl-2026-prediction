"""
KAGGLE NOTEBOOK DIAGNOSTIC - Run this first to understand what's wrong
"""

import pandas as pd
from pathlib import Path
import sys

print("="*70)
print("KAGGLE ENVIRONMENT DIAGNOSTIC")
print("="*70)

# 1. Check Python version
print(f"\n1. Python version: {sys.version}")

# 2. Check pandas version
print(f"\n2. Pandas version: {pd.__version__}")

# 3. Check current working directory
print(f"\n3. Current directory: {Path.cwd()}")

# 4. Check if /kaggle/working exists
kaggle_working = Path("/kaggle/working")
print(f"\n4. /kaggle/working exists: {kaggle_working.exists()}")
if not kaggle_working.exists():
    print("   Creating /kaggle/working...")
    try:
        kaggle_working.mkdir(parents=True, exist_ok=True)
        print("   ✅ Created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create: {e}")

# 5. List files in /kaggle/input
input_path = Path("/kaggle/input")
print(f"\n5. /kaggle/input exists: {input_path.exists()}")
if input_path.exists():
    print("   Contents of /kaggle/input:")
    try:
        for item in input_path.iterdir():
            print(f"      - {item.name}")
            if item.is_dir():
                for subitem in item.iterdir():
                    print(f"         - {subitem.name}")
    except Exception as e:
        print(f"   ❌ Cannot list: {e}")

# 6. Check if we can create a simple CSV
print(f"\n6. Testing CSV creation...")
try:
    test_df = pd.DataFrame({
        'id': ['test_1', 'test_2'],
        'x': [1.0, 2.0],
        'y': [3.0, 4.0]
    })
    test_path = Path("/kaggle/working/test.csv")
    test_df.to_csv(test_path, index=False, lineterminator='\n', encoding='utf-8')
    
    # Read it back
    read_back = pd.read_csv(test_path)
    print(f"   ✅ CSV creation successful")
    print(f"   File: {test_path}")
    print(f"   File exists: {test_path.exists()}")
    print(f"   File size: {test_path.stat().st_size} bytes")
    print(f"   Read back successfully: {len(read_back)} rows")
except Exception as e:
    print(f"   ❌ CSV creation failed: {e}")
    import traceback
    traceback.print_exc()

# 7. Check memory
print(f"\n7. Testing large dictionary creation...")
try:
    large_dict = [{'id': f'test_{i}', 'x': float(i), 'y': float(i*2)} for i in range(5837)]
    print(f"   ✅ Created dictionary with {len(large_dict)} items")
    
    df_from_dict = pd.DataFrame(large_dict)
    print(f"   ✅ Converted to DataFrame: {len(df_from_dict)} rows")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\n⚠️  Copy the output above and share it so I can help debug!")
