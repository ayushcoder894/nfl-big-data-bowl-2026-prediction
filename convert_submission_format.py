"""Convert submission.csv to match sample_submission.csv format exactly."""

import pandas as pd

print("="*70)
print("CONVERTING submission.csv TO MATCH sample_submission.csv FORMAT")
print("="*70)

# Read the current submission.csv
print("\n📖 Reading submission.csv...")
df = pd.read_csv('submission.csv')
print(f"   Rows: {len(df):,}")
print(f"   Columns: {df.columns.tolist()}")

# Backup the original
print("\n💾 Creating backup: submission_backup.csv...")
df.to_csv('submission_backup.csv', index=False)
print("   ✅ Backup created")

# Save with Unix line endings (LF) to match sample_submission.csv
print("\n🔧 Converting to Unix (LF) line endings...")
print("   - Encoding: UTF-8 (no BOM)")
print("   - Line terminator: \\n (Unix/Mac)")
print("   - No index column")

# Write with Unix line endings
df.to_csv('submission.csv', index=False, lineterminator='\n', encoding='utf-8')

print("   ✅ Conversion complete")

# Verify the conversion
print("\n🔍 VERIFYING CONVERSION...")

with open('sample_submission.csv', 'rb') as f:
    sample_bytes = f.read()
    
with open('submission.csv', 'rb') as f:
    new_bytes = f.read()

# Check line endings
sample_crlf = sample_bytes.count(b'\r\n')
sample_lf = sample_bytes.count(b'\n') - sample_crlf
new_crlf = new_bytes.count(b'\r\n')
new_lf = new_bytes.count(b'\n') - new_crlf

print(f"\n   sample_submission.csv:")
print(f"      CRLF: {sample_crlf}, LF: {sample_lf}")
print(f"      Line ending type: {'CRLF (Windows)' if sample_crlf > 0 else 'LF (Unix)'}")

print(f"\n   submission.csv (NEW):")
print(f"      CRLF: {new_crlf}, LF: {new_lf}")
print(f"      Line ending type: {'CRLF (Windows)' if new_crlf > 0 else 'LF (Unix)'}")

# Check encoding
print(f"\n   Encoding check:")
try:
    sample_bytes.decode('utf-8')
    print(f"      sample_submission.csv: UTF-8 ✅")
except:
    print(f"      sample_submission.csv: Not UTF-8 ❌")

try:
    new_bytes.decode('utf-8')
    print(f"      submission.csv: UTF-8 ✅")
except:
    print(f"      submission.csv: Not UTF-8 ❌")

# Check ASCII
try:
    sample_bytes.decode('ascii')
    print(f"      sample_submission.csv: ASCII compatible ✅")
except:
    print(f"      sample_submission.csv: Contains non-ASCII ⚠️")

try:
    new_bytes.decode('ascii')
    print(f"      submission.csv: ASCII compatible ✅")
except:
    print(f"      submission.csv: Contains non-ASCII ⚠️")

# Check BOM
sample_has_bom = sample_bytes.startswith(b'\xef\xbb\xbf')
new_has_bom = new_bytes.startswith(b'\xef\xbb\xbf')
print(f"\n   BOM (Byte Order Mark):")
print(f"      sample_submission.csv: {'Present' if sample_has_bom else 'Not present'}")
print(f"      submission.csv: {'Present' if new_has_bom else 'Not present'}")

# Check file ending
sample_ends_newline = sample_bytes.endswith(b'\n')
new_ends_newline = new_bytes.endswith(b'\n')
print(f"\n   File ending:")
print(f"      sample_submission.csv: {'Ends with newline ✅' if sample_ends_newline else 'No newline ❌'}")
print(f"      submission.csv: {'Ends with newline ✅' if new_ends_newline else 'No newline ❌'}")

# Check headers
sample_lines = sample_bytes.splitlines()
new_lines = new_bytes.splitlines()
print(f"\n   Headers:")
print(f"      sample_submission.csv: {sample_lines[0].decode('utf-8')}")
print(f"      submission.csv: {new_lines[0].decode('utf-8')}")
print(f"      Match: {'✅' if sample_lines[0] == new_lines[0] else '❌'}")

# Overall verification
print(f"\n{'='*70}")
print("VERIFICATION SUMMARY")
print(f"{'='*70}")

checks = {
    "Line endings match (LF)": (sample_lf > 0 and new_lf > 0 and sample_crlf == 0 and new_crlf == 0),
    "UTF-8 encoding": True,
    "No BOM": (not sample_has_bom and not new_has_bom),
    "Ends with newline": (sample_ends_newline and new_ends_newline),
    "Headers match": (sample_lines[0] == new_lines[0]),
}

all_passed = True
for check, passed in checks.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"   {status}: {check}")
    if not passed:
        all_passed = False

print(f"\n{'='*70}")
if all_passed:
    print("🎉 SUCCESS! submission.csv now matches sample_submission.csv format!")
    print("✅ Ready for Kaggle submission!")
else:
    print("⚠️ Some checks failed. Manual review recommended.")
print(f"{'='*70}")

print(f"\n📊 File sizes:")
print(f"   sample_submission.csv: {len(sample_bytes):,} bytes")
print(f"   submission.csv (NEW): {len(new_bytes):,} bytes")
print(f"   Difference: {len(new_bytes) - len(sample_bytes):,} bytes (due to prediction values)")

print(f"\n💡 Note: The original file is backed up as 'submission_backup.csv'")
