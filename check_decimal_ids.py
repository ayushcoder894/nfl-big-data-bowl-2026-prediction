"""Check for decimal points in submission IDs."""

from pathlib import Path

import pandas as pd


def main() -> None:
    print("=" * 70)
    print("DECIMAL POINT CHECKER FOR IDs")
    print("=" * 70)
    
    submission_path = Path("submission.csv")
    
    if not submission_path.exists():
        print(f"\n❌ ERROR: {submission_path} not found!")
        return
    
    print(f"\n📄 Reading submission.csv...")
    sub = pd.read_csv(submission_path)
    
    print(f"\n{'='*70}")
    print("CHECKING FOR DECIMAL POINTS IN IDs")
    print("=" * 70)
    
    print(f"\n1. ID Data Type: {sub['id'].dtype}")
    
    print(f"\n2. First 10 IDs (raw):")
    for idx, id_val in enumerate(sub['id'].head(10), 1):
        print(f"   {idx}. {repr(id_val)} (type: {type(id_val).__name__})")
    
    print(f"\n3. Checking for decimal points...")
    ids_with_decimals = []
    
    for idx, id_val in enumerate(sub['id']):
        id_str = str(id_val)
        if '.' in id_str:
            ids_with_decimals.append((idx, id_val, id_str))
    
    if ids_with_decimals:
        print(f"\n   ❌ FOUND {len(ids_with_decimals)} IDs WITH DECIMAL POINTS!")
        print(f"\n   First 20 problematic IDs:")
        for i, (idx, id_val, id_str) in enumerate(ids_with_decimals[:20], 1):
            print(f"      {i}. Row {idx}: {repr(id_val)} -> '{id_str}'")
        
        print(f"\n   💡 FIX REQUIRED:")
        print(f"      IDs must be strings without decimal points")
        print(f"      Example: '2024120805_74_54586_1' not '2024120805_74_54586_1.0'")
        
    else:
        print(f"   ✅ No decimal points found in any IDs!")
    
    print(f"\n4. Checking ID format after string conversion...")
    
    # Check if converting to string removes decimals properly
    sample_ids = sub['id'].head(10)
    print(f"\n   Converting first 10 IDs to clean strings:")
    for idx, id_val in enumerate(sample_ids, 1):
        original = repr(id_val)
        as_string = str(id_val)
        # Remove .0 if present
        clean = as_string.replace('.0', '') if '.0' in as_string else as_string
        print(f"      {idx}. {original} -> '{clean}'")
    
    print(f"\n5. Comparing with template...")
    template_path = Path("sample_submission.csv")
    
    if template_path.exists():
        template = pd.read_csv(template_path)
        print(f"\n   Template ID type: {template['id'].dtype}")
        print(f"\n   First 5 template IDs:")
        for idx, id_val in enumerate(template['id'].head(), 1):
            print(f"      {idx}. {repr(id_val)} (type: {type(id_val).__name__})")
        
        template_has_decimals = any('.' in str(x) for x in template['id'].head(100))
        print(f"\n   Template IDs have decimals: {template_has_decimals}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    
    if ids_with_decimals:
        print(f"\n❌ CRITICAL ISSUE FOUND!")
        print(f"   {len(ids_with_decimals)} out of {len(sub)} IDs contain decimal points")
        print(f"\n   This will cause Kaggle submission errors!")
        print(f"\n   🔧 TO FIX:")
        print(f"      1. Read CSV with dtype={{'id': 'str'}} to preserve ID format")
        print(f"      2. Or use .astype(str).str.replace('.0', '', regex=False)")
        print(f"      3. Ensure IDs are clean strings before writing CSV")
    else:
        print(f"\n✅ ALL IDs ARE CLEAN!")
        print(f"   No decimal points found in any ID")
    
    print(f"\n{'='*70}\n")
    
    # Show fix example if needed
    if ids_with_decimals:
        print("\n" + "=" * 70)
        print("EXAMPLE FIX CODE")
        print("=" * 70)
        print("""
# When reading:
df = pd.read_csv('file.csv', dtype={'id': 'str'})

# Or when cleaning:
df['id'] = df['id'].astype(str).str.replace('.0', '', regex=False)

# Verify:
assert not any('.' in str(x) for x in df['id']), "IDs still have decimals!"

# Save:
df.to_csv('submission.csv', index=False)
        """)
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
