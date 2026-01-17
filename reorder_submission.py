"""Reorder submission.csv to match sample_submission.csv ID order exactly."""

import pandas as pd

print("="*70)
print("REORDERING SUBMISSION TO MATCH TEMPLATE ORDER")
print("="*70)

# Load files
print("\n📖 Loading files...")
submission = pd.read_csv('submission.csv')
template = pd.read_csv('sample_submission.csv')

print(f"   submission.csv: {len(submission):,} rows")
print(f"   template: {len(template):,} rows")

# Backup
print("\n💾 Creating backup: submission_before_reorder.csv...")
submission.to_csv('submission_before_reorder.csv', index=False, lineterminator='\n', encoding='utf-8')
print("   ✅ Backup created")

# Reorder submission to match template ID order
print("\n🔄 Reordering...")
template_id_order = template[['id']].copy()
template_id_order['order'] = range(len(template_id_order))

# Merge to get the correct order
reordered = template_id_order.merge(submission, on='id', how='left')
reordered = reordered.sort_values('order').drop('order', axis=1)

# Check for any missing values (shouldn't happen if IDs match)
missing_count = reordered[['x', 'y']].isna().sum().sum()
if missing_count > 0:
    print(f"   ⚠️  WARNING: {missing_count} missing values after merge!")
    print(f"   This means some template IDs are not in your submission")
else:
    print(f"   ✅ No missing values - all template IDs found")

# Verify order
print("\n✅ Verification:")
print(f"   Row count: {len(reordered):,}")
print(f"   IDs in template order: {reordered['id'].tolist() == template['id'].tolist()}")

# Save
print("\n💾 Saving reordered submission.csv...")
reordered.to_csv('submission.csv', index=False, lineterminator='\n', encoding='utf-8')
print("   ✅ Saved!")

# Show comparison
print("\n📋 FIRST 10 IDs - BEFORE vs AFTER:")
print("\n   BEFORE (wrong order):")
before = pd.read_csv('submission_before_reorder.csv')
for i in range(min(10, len(before))):
    print(f"      [{i}] {before['id'].iloc[i]}")

print("\n   AFTER (correct order):")
for i in range(min(10, len(reordered))):
    print(f"      [{i}] {reordered['id'].iloc[i]}")

print("\n   TEMPLATE (target order):")
for i in range(min(10, len(template))):
    print(f"      [{i}] {template['id'].iloc[i]}")

print(f"\n{'='*70}")
print("✅ REORDERING COMPLETE!")
print(f"{'='*70}")
print("\nYour submission.csv now has IDs in the EXACT same order as the template!")
print("Try submitting to Kaggle again! 🚀")
