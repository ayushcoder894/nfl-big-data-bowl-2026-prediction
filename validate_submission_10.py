"""Validate submission (10).csv format"""

import pandas as pd

print("="*70)
print("VALIDATING: submission (10).csv")
print("="*70)

# Load files
sub = pd.read_csv('submission (10).csv')
template = pd.read_csv('sample_submission.csv')

print(f"\n✅ LOADED FILES:")
print(f"   submission (10).csv: {len(sub):,} rows")
print(f"   sample_submission.csv: {len(template):,} rows")

# Check basics
print(f"\n📋 BASIC CHECKS:")
print(f"   Columns: {sub.columns.tolist()}")
print(f"   ✅ Row count matches: {len(sub) == len(template)}")
print(f"   ✅ No missing values: {not sub.isna().any().any()}")
print(f"   ✅ Column names match: {list(sub.columns) == ['id', 'x', 'y']}")

# Check IDs
print(f"\n🆔 ID CHECKS:")
template_ids = set(template['id'])
sub_ids = set(sub['id'])
print(f"   ✅ All IDs match template: {sub_ids == template_ids}")
print(f"   ✅ No duplicate IDs: {not sub['id'].duplicated().any()}")

# Check ID order
print(f"\n📊 ID ORDER:")
template_id_list = template['id'].tolist()
sub_id_list = sub['id'].tolist()
if template_id_list == sub_id_list:
    print(f"   ✅ IDs in EXACT same order as template")
else:
    print(f"   ⚠️  IDs are NOT in same order as template")
    # Find first difference
    for i, (t_id, s_id) in enumerate(zip(template_id_list, sub_id_list)):
        if t_id != s_id:
            print(f"      First diff at row {i}:")
            print(f"      Template: {t_id}")
            print(f"      Submission (10): {s_id}")
            break

# Check coordinates
print(f"\n📍 COORDINATE CHECKS:")
print(f"   X range: [{sub['x'].min():.2f}, {sub['x'].max():.2f}] (valid: 0-120)")
x_in_range = sub['x'].between(0, 120).all()
print(f"   {'✅' if x_in_range else '❌'} X in range: {x_in_range}")
print(f"   Y range: [{sub['y'].min():.2f}, {sub['y'].max():.2f}] (valid: 0-53.3)")
y_in_range = sub['y'].between(0, 53.3).all()
print(f"   {'✅' if y_in_range else '❌'} Y in range: {y_in_range}")

# Check data types
print(f"\n🔤 DATA TYPES:")
print(f"   id: {sub['id'].dtype} (should be object)")
print(f"   x: {sub['x'].dtype} (should be float64)")
print(f"   y: {sub['y'].dtype} (should be float64)")

# Binary format check
print(f"\n💾 BINARY FORMAT:")
with open('submission (10).csv', 'rb') as f:
    content = f.read()
    
crlf = content.count(b'\r\n')
lf = content.count(b'\n') - crlf

line_ending_status = 'LF (Unix) ✅' if lf > 0 and crlf == 0 else 'CRLF (Windows) ❌'
print(f"   Line endings: {line_ending_status}")

bom_present = content.startswith(b'\xef\xbb\xbf')
bom_status = 'Present ❌' if bom_present else 'Not present ✅'
print(f"   BOM: {bom_status}")

ends_newline = content.endswith(b'\n')
newline_status = 'Yes ✅' if ends_newline else 'No ❌'
print(f"   Ends with newline: {newline_status}")

# File size
import os
file_size = os.path.getsize('submission (10).csv')
print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

# Sample data
print(f"\n📋 FIRST 5 ROWS:")
print(sub.head().to_string(index=False))

print(f"\n📋 LAST 5 ROWS:")
print(sub.tail().to_string(index=False))

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

all_good = (
    len(sub) == len(template) and
    not sub.isna().any().any() and
    list(sub.columns) == ['id', 'x', 'y'] and
    sub_ids == template_ids and
    not sub['id'].duplicated().any() and
    x_in_range and
    y_in_range and
    crlf == 0 and lf > 0
)

if all_good and template_id_list == sub_id_list:
    print("✅ PERFECT! All checks passed!")
    print("✅ submission (10).csv is ready for Kaggle!")
elif all_good:
    print("✅ All format checks passed!")
    print("⚠️  BUT: IDs are not in template order")
    print("   This might cause Kaggle to reject it")
else:
    print("⚠️  Some checks failed - see details above")
