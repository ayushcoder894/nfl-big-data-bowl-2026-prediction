"""Deep dive into ID matching - check if IDs are actually what Kaggle expects."""

import pandas as pd

print("="*70)
print("DEEP ID ANALYSIS")
print("="*70)

# Load files
submission = pd.read_csv('submission.csv')
template = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv')

print(f"\n📊 FILE SIZES:")
print(f"   submission.csv:         {len(submission):,} rows")
print(f"   sample_submission.csv:  {len(template):,} rows")
print(f"   test.csv:               {len(test):,} rows")

# Check if test.csv IDs match template
print(f"\n🔍 CHECKING TEST.CSV IDs:")
test['id'] = test['game_id'].astype(str) + '_' + test['play_id'].astype(str) + '_' + test['nfl_id'].astype(str) + '_' + test['frame_id'].astype(str)
test_ids = set(test['id'])
template_ids = set(template['id'])
submission_ids = set(submission['id'])

print(f"\n   Unique IDs in test.csv:              {len(test_ids):,}")
print(f"   Unique IDs in sample_submission.csv: {len(template_ids):,}")
print(f"   Unique IDs in submission.csv:        {len(submission_ids):,}")

print(f"\n   test.csv IDs == template IDs: {test_ids == template_ids}")
print(f"   submission IDs == template IDs: {submission_ids == template_ids}")

if test_ids != template_ids:
    print(f"\n   ⚠️  WARNING: test.csv IDs don't match template!")
    missing = template_ids - test_ids
    extra = test_ids - template_ids
    if missing:
        print(f"   Missing from test.csv: {len(missing)}")
        print(f"   First 5: {list(missing)[:5]}")
    if extra:
        print(f"   Extra in test.csv: {len(extra)}")
        print(f"   First 5: {list(extra)[:5]}")

# Compare first few IDs
print(f"\n📋 FIRST 10 IDs COMPARISON:")
print(f"\n   sample_submission.csv:")
for i, id_val in enumerate(template['id'].head(10)):
    print(f"      [{i}] {id_val}")

print(f"\n   submission.csv:")
for i, id_val in enumerate(submission['id'].head(10)):
    print(f"      [{i}] {id_val}")

print(f"\n   test.csv:")
for i, id_val in enumerate(test['id'].head(10)):
    print(f"      [{i}] {id_val}")

# Check if submission IDs are in the right order
print(f"\n🔍 ID ORDER CHECK:")
template_id_list = template['id'].tolist()
submission_id_list = submission['id'].tolist()

if template_id_list == submission_id_list:
    print(f"   ✅ IDs are in EXACT same order as template")
else:
    print(f"   ⚠️  IDs are NOT in same order as template")
    print(f"   Finding first difference...")
    for i, (t_id, s_id) in enumerate(zip(template_id_list, submission_id_list)):
        if t_id != s_id:
            print(f"   First difference at position {i}:")
            print(f"      Template: {t_id}")
            print(f"      Submission: {s_id}")
            break

# Check for any weird characters in IDs
print(f"\n🔍 CHARACTER ANALYSIS:")
submission_id_str = ''.join(submission['id'].astype(str))
template_id_str = ''.join(template['id'].astype(str))

import string
valid_chars = set(string.digits + '_')
submission_chars = set(submission_id_str)
template_chars = set(template_id_str)

print(f"   Characters in submission IDs: {sorted(submission_chars)}")
print(f"   Characters in template IDs: {sorted(template_chars)}")

if submission_chars == template_chars:
    print(f"   ✅ Same character set")
else:
    print(f"   ⚠️  Different character sets!")
    extra = submission_chars - template_chars
    missing = template_chars - submission_chars
    if extra:
        print(f"   Extra characters in submission: {extra}")
    if missing:
        print(f"   Missing characters in submission: {missing}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
