"""Comprehensive submission validation against Kaggle requirements."""

import pandas as pd
import numpy as np
import re

print("="*70)
print("COMPREHENSIVE KAGGLE SUBMISSION VALIDATION")
print("="*70)

def validate_submission(filepath, template_path):
    print(f"\n🔍 VALIDATING: {filepath}")
    print("="*70)
    
    # Read files
    try:
        df = pd.read_csv(filepath)
        template = pd.read_csv(template_path)
    except Exception as e:
        print(f"❌ ERROR reading files: {e}")
        return False
    
    all_checks_passed = True
    
    # 1. Column names
    print("\n1️⃣  COLUMN NAMES:")
    expected_cols = ['id', 'x', 'y']
    if list(df.columns) == expected_cols:
        print(f"   ✅ Correct: {df.columns.tolist()}")
    else:
        print(f"   ❌ WRONG: {df.columns.tolist()}")
        print(f"   Expected: {expected_cols}")
        all_checks_passed = False
    
    # 2. Row count
    print("\n2️⃣  ROW COUNT:")
    if len(df) == len(template):
        print(f"   ✅ Correct: {len(df):,} rows (matches template)")
    else:
        print(f"   ❌ WRONG: {len(df):,} rows (expected {len(template):,})")
        all_checks_passed = False
    
    # 3. Missing values
    print("\n3️⃣  MISSING VALUES:")
    missing = df.isna().sum()
    if missing.sum() == 0:
        print(f"   ✅ No missing values")
    else:
        print(f"   ❌ FOUND missing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"      {col}: {count} missing")
        all_checks_passed = False
    
    # 4. ID format
    print("\n4️⃣  ID FORMAT:")
    id_pattern = r'^\d+_\d+_\d+_\d+$'
    valid_ids = df['id'].astype(str).str.match(id_pattern)
    if valid_ids.all():
        print(f"   ✅ All IDs match pattern: game_play_nfl_frame")
    else:
        invalid_count = (~valid_ids).sum()
        print(f"   ❌ INVALID IDs found: {invalid_count}")
        print(f"   First 5 invalid IDs:")
        print(df.loc[~valid_ids, 'id'].head().tolist())
        all_checks_passed = False
    
    # 5. ID data type
    print("\n5️⃣  ID DATA TYPE:")
    if df['id'].dtype == 'object':
        print(f"   ✅ ID dtype: {df['id'].dtype} (string)")
        # Check for decimal points
        has_decimal = df['id'].astype(str).str.contains(r'\.').any()
        if has_decimal:
            print(f"   ❌ IDs contain decimal points!")
            all_checks_passed = False
        else:
            print(f"   ✅ No decimal points in IDs")
    else:
        print(f"   ❌ ID dtype: {df['id'].dtype} (should be object/string)")
        all_checks_passed = False
    
    # 6. ID matching
    print("\n6️⃣  ID MATCHING WITH TEMPLATE:")
    template_ids = set(template['id'])
    submission_ids = set(df['id'])
    
    if submission_ids == template_ids:
        print(f"   ✅ All IDs match template exactly")
    else:
        missing_ids = template_ids - submission_ids
        extra_ids = submission_ids - template_ids
        
        if missing_ids:
            print(f"   ❌ Missing {len(missing_ids)} IDs from template")
            print(f"   First 5 missing: {list(missing_ids)[:5]}")
            all_checks_passed = False
        
        if extra_ids:
            print(f"   ❌ Found {len(extra_ids)} extra IDs not in template")
            print(f"   First 5 extra: {list(extra_ids)[:5]}")
            all_checks_passed = False
    
    # 7. X coordinate validation
    print("\n7️⃣  X COORDINATE VALIDATION:")
    if df['x'].dtype in ['float64', 'float32', 'int64', 'int32']:
        print(f"   ✅ X dtype: {df['x'].dtype}")
    else:
        print(f"   ❌ X dtype: {df['x'].dtype} (should be numeric)")
        all_checks_passed = False
    
    x_min, x_max = df['x'].min(), df['x'].max()
    if 0 <= x_min and x_max <= 120:
        print(f"   ✅ X range: [{x_min:.2f}, {x_max:.2f}] (valid 0-120)")
    else:
        print(f"   ❌ X range: [{x_min:.2f}, {x_max:.2f}] (should be 0-120)")
        all_checks_passed = False
    
    if np.isinf(df['x']).any():
        print(f"   ❌ X contains infinity values")
        all_checks_passed = False
    else:
        print(f"   ✅ No infinity values in X")
    
    # 8. Y coordinate validation
    print("\n8️⃣  Y COORDINATE VALIDATION:")
    if df['y'].dtype in ['float64', 'float32', 'int64', 'int32']:
        print(f"   ✅ Y dtype: {df['y'].dtype}")
    else:
        print(f"   ❌ Y dtype: {df['y'].dtype} (should be numeric)")
        all_checks_passed = False
    
    y_min, y_max = df['y'].min(), df['y'].max()
    if 0 <= y_min and y_max <= 53.3:
        print(f"   ✅ Y range: [{y_min:.2f}, {y_max:.2f}] (valid 0-53.3)")
    else:
        print(f"   ❌ Y range: [{y_min:.2f}, {y_max:.2f}] (should be 0-53.3)")
        all_checks_passed = False
    
    if np.isinf(df['y']).any():
        print(f"   ❌ Y contains infinity values")
        all_checks_passed = False
    else:
        print(f"   ✅ No infinity values in Y")
    
    # 9. Duplicate IDs
    print("\n9️⃣  DUPLICATE IDS:")
    duplicates = df['id'].duplicated()
    if duplicates.any():
        print(f"   ❌ Found {duplicates.sum()} duplicate IDs")
        print(f"   Duplicates: {df.loc[duplicates, 'id'].head().tolist()}")
        all_checks_passed = False
    else:
        print(f"   ✅ No duplicate IDs")
    
    # 10. ID order
    print("\n🔟 ID ORDER:")
    sorted_df = df.sort_values('id').reset_index(drop=True)
    if df['id'].equals(sorted_df['id']):
        print(f"   ✅ IDs are sorted")
    else:
        print(f"   ⚠️  IDs are NOT sorted (might be okay)")
        # Check if template is sorted
        sorted_template = template.sort_values('id').reset_index(drop=True)
        if template['id'].equals(sorted_template['id']):
            print(f"   ℹ️  Template is sorted - recommend sorting submission")
    
    # 11. Binary format
    print("\n1️⃣1️⃣  BINARY FORMAT:")
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    
    with open(template_path, 'rb') as f:
        template_bytes = f.read()
    
    # Line endings
    crlf_count = file_bytes.count(b'\r\n')
    lf_count = file_bytes.count(b'\n') - crlf_count
    template_crlf = template_bytes.count(b'\r\n')
    template_lf = template_bytes.count(b'\n') - template_crlf
    
    if crlf_count == 0 and lf_count > 0:
        print(f"   ✅ Line endings: LF (Unix) - {lf_count} lines")
    elif crlf_count > 0:
        print(f"   ❌ Line endings: CRLF (Windows) - {crlf_count} lines")
        print(f"   Template uses: {'CRLF' if template_crlf > 0 else 'LF'}")
        all_checks_passed = False
    
    # BOM
    if file_bytes.startswith(b'\xef\xbb\xbf'):
        print(f"   ❌ Contains UTF-8 BOM (should not have BOM)")
        all_checks_passed = False
    else:
        print(f"   ✅ No BOM")
    
    # File ending
    if file_bytes.endswith(b'\n'):
        print(f"   ✅ Ends with newline")
    else:
        print(f"   ⚠️  Does NOT end with newline")
    
    # 12. Sample data
    print("\n1️⃣2️⃣  SAMPLE DATA:")
    print(f"\n   First 3 rows:")
    print(df.head(3).to_string(index=False))
    
    print(f"\n   Last 3 rows:")
    print(df.tail(3).to_string(index=False))
    
    return all_checks_passed

# Validate both files
result1 = validate_submission('submission.csv', 'sample_submission.csv')
print("\n" + "="*70)
result2 = validate_submission('submission (6).csv', 'sample_submission.csv')

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)
print(f"submission.csv:     {'✅ PASSED ALL CHECKS' if result1 else '❌ FAILED SOME CHECKS'}")
print(f"submission (6).csv: {'✅ PASSED ALL CHECKS' if result2 else '❌ FAILED SOME CHECKS'}")
print("="*70)
