"""Check binary format and line endings of CSV files."""

import os

def analyze_file(filepath):
    print(f"\n{'='*70}")
    print(f"ANALYZING: {filepath}")
    print(f"{'='*70}")
    
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    
    print(f"\n📦 FILE SIZE: {len(file_bytes):,} bytes ({len(file_bytes)/1024:.2f} KB)")
    
    print(f"\n🔍 FIRST 100 BYTES:")
    print(f"   Raw: {file_bytes[:100]}")
    print(f"   Decoded: {file_bytes[:100].decode('utf-8', errors='replace')}")
    
    print(f"\n🔍 LAST 100 BYTES:")
    print(f"   Raw: {file_bytes[-100:]}")
    print(f"   Decoded: {file_bytes[-100:].decode('utf-8', errors='replace')}")
    
    # Line ending analysis
    crlf_count = file_bytes.count(b'\r\n')
    lf_only_count = file_bytes.count(b'\n') - crlf_count
    cr_only_count = file_bytes.count(b'\r') - crlf_count
    
    print(f"\n📝 LINE ENDING ANALYSIS:")
    print(f"   CRLF (\\r\\n) - Windows: {crlf_count}")
    print(f"   LF (\\n) - Unix/Mac: {lf_only_count}")
    print(f"   CR (\\r) - Old Mac: {cr_only_count}")
    
    # Detect line ending type
    if crlf_count > 0:
        line_ending_type = "CRLF (Windows)"
    elif lf_only_count > 0:
        line_ending_type = "LF (Unix/Mac)"
    elif cr_only_count > 0:
        line_ending_type = "CR (Old Mac)"
    else:
        line_ending_type = "Unknown/None"
    print(f"   Primary line ending: {line_ending_type}")
    
    # Split by lines
    lines = file_bytes.splitlines()
    print(f"\n📋 LINE COUNT: {len(lines):,}")
    
    # Character encoding check
    print(f"\n🔤 ENCODING CHECK:")
    try:
        text = file_bytes.decode('utf-8')
        print(f"   UTF-8: ✅ Valid")
    except UnicodeDecodeError as e:
        print(f"   UTF-8: ❌ Invalid - {e}")
    
    try:
        text = file_bytes.decode('ascii')
        print(f"   ASCII: ✅ Valid")
    except UnicodeDecodeError:
        print(f"   ASCII: ❌ Invalid (contains non-ASCII characters)")
    
    # BOM check
    if file_bytes.startswith(b'\xef\xbb\xbf'):
        print(f"   BOM (Byte Order Mark): ✅ Present (UTF-8 BOM)")
    else:
        print(f"   BOM (Byte Order Mark): ❌ Not present")
    
    # File ending check
    if file_bytes.endswith(b'\n'):
        print(f"\n✅ File ends with newline")
    else:
        print(f"\n⚠️  File does NOT end with newline")
    
    return file_bytes, lines

# Analyze both files
print("="*70)
print("BINARY FORMAT AND LINE ENDING ANALYSIS")
print("="*70)

sample_bytes, sample_lines = analyze_file("sample_submission.csv")
generated_bytes, generated_lines = analyze_file("submission.csv")

# Cross-comparison
print(f"\n{'='*70}")
print("CROSS-COMPARISON")
print(f"{'='*70}")

print(f"\n🔄 BINARY COMPARISON:")
print(f"   Files are byte-for-byte identical: {sample_bytes == generated_bytes}")
print(f"   Size difference: {len(generated_bytes) - len(sample_bytes):,} bytes")

print(f"\n🔄 LINE COMPARISON:")
print(f"   Same number of lines: {len(sample_lines) == len(generated_lines)} ({len(sample_lines):,} vs {len(generated_lines):,})")
print(f"   Line endings identical: {sample_lines == generated_lines}")

if sample_lines != generated_lines:
    print(f"\n🔍 FINDING DIFFERENCES:")
    
    # Find first difference
    for i, (line1, line2) in enumerate(zip(sample_lines, generated_lines)):
        if line1 != line2:
            print(f"\n   First difference at line {i+1}:")
            print(f"      sample_submission.csv: {line1[:100]}")
            print(f"      submission.csv:        {line2[:100]}")
            if len(line1) != len(line2):
                print(f"      Length diff: {len(line2) - len(line1)} bytes")
            break
    
    # Check if only difference is in data values
    if len(sample_lines) == len(generated_lines):
        header_match = sample_lines[0] == generated_lines[0]
        print(f"\n   Headers match: {header_match}")
        if header_match:
            print(f"   → Difference is in data values only (expected)")

# First 3 lines comparison
print(f"\n🔄 FIRST 3 LINES (decoded):")
print(f"\n   sample_submission.csv:")
for i in range(min(3, len(sample_lines))):
    print(f"      [{i}] {sample_lines[i].decode('utf-8', errors='replace')}")

print(f"\n   submission.csv:")
for i in range(min(3, len(generated_lines))):
    print(f"      [{i}] {generated_lines[i].decode('utf-8', errors='replace')}")

print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*70}")
