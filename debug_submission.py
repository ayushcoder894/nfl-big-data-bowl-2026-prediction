"""Debug submission.csv format to identify Kaggle scoring errors."""

from pathlib import Path

import pandas as pd


def main() -> None:
    print("=" * 70)
    print("SUBMISSION FORMAT VALIDATOR")
    print("=" * 70)
    
    submission_path = Path("submission.csv")
    sample_path = Path("sample_submission.csv")
    
    if not submission_path.exists():
        print(f"\n❌ ERROR: {submission_path} not found!")
        return
    
    if not sample_path.exists():
        print(f"\n❌ ERROR: {sample_path} not found!")
        return
    
    print(f"\n📄 Reading files...")
    sub = pd.read_csv(submission_path)
    template = pd.read_csv(sample_path)
    
    print(f"\n{'='*70}")
    print("BASIC CHECKS")
    print("=" * 70)
    
    print(f"\n1. Column Names:")
    print(f"   Submission columns: {sub.columns.tolist()}")
    print(f"   Expected columns:   ['id', 'x', 'y']")
    cols_match = set(sub.columns) == {"id", "x", "y"}
    print(f"   ✅ Match: {cols_match}" if cols_match else f"   ❌ Mismatch!")
    
    print(f"\n2. Row Count:")
    print(f"   Submission rows: {len(sub):,}")
    print(f"   Sample rows:     {len(template):,}")
    rows_match = len(sub) == len(template)
    print(f"   ✅ Match: {rows_match}" if rows_match else f"   ❌ Mismatch!")
    
    print(f"\n3. Missing Values:")
    has_na = sub.isna().any().any()
    na_count = sub.isna().sum().sum()
    print(f"   Total NaN/None values: {na_count}")
    print(f"   ✅ No missing values" if not has_na else f"   ❌ Contains missing values!")
    
    if has_na:
        print(f"\n   Missing value breakdown:")
        for col in sub.columns:
            missing = sub[col].isna().sum()
            if missing > 0:
                print(f"     - {col}: {missing} missing")
    
    print(f"\n4. Data Types:")
    print(f"   id:  {sub['id'].dtype}")
    print(f"   x:   {sub['x'].dtype if 'x' in sub.columns else 'N/A'}")
    print(f"   y:   {sub['y'].dtype if 'y' in sub.columns else 'N/A'}")
    
    if 'x' in sub.columns and 'y' in sub.columns:
        x_numeric = pd.api.types.is_numeric_dtype(sub['x'])
        y_numeric = pd.api.types.is_numeric_dtype(sub['y'])
        print(f"   ✅ x and y are numeric" if x_numeric and y_numeric else f"   ❌ x or y not numeric!")
    
    print(f"\n{'='*70}")
    print("DETAILED CHECKS")
    print("=" * 70)
    
    print(f"\n5. ID Column Check:")
    template_ids = set(template['id'])
    submission_ids = set(sub['id'])
    
    missing_ids = template_ids - submission_ids
    extra_ids = submission_ids - template_ids
    
    print(f"   IDs in sample:      {len(template_ids):,}")
    print(f"   IDs in submission:  {len(submission_ids):,}")
    print(f"   Missing IDs:        {len(missing_ids):,}")
    print(f"   Extra IDs:          {len(extra_ids):,}")
    
    if missing_ids:
        print(f"\n   ❌ Missing IDs (first 10):")
        for idx, missing_id in enumerate(list(missing_ids)[:10], 1):
            print(f"      {idx}. {missing_id}")
    
    if extra_ids:
        print(f"\n   ⚠️  Extra IDs (first 10):")
        for idx, extra_id in enumerate(list(extra_ids)[:10], 1):
            print(f"      {idx}. {extra_id}")
    
    if not missing_ids and not extra_ids:
        print(f"   ✅ All IDs match perfectly!")
    
    if 'x' in sub.columns and 'y' in sub.columns:
        print(f"\n6. Coordinate Range Check:")
        print(f"   x range: [{sub['x'].min():.2f}, {sub['x'].max():.2f}]")
        print(f"   y range: [{sub['y'].min():.2f}, {sub['y'].max():.2f}]")
        print(f"   Expected x: [0, 120]")
        print(f"   Expected y: [0, 53.3]")
        
        x_valid = (sub['x'] >= 0).all() and (sub['x'] <= 120).all()
        y_valid = (sub['y'] >= 0).all() and (sub['y'] <= 53.3).all()
        
        if x_valid and y_valid:
            print(f"   ✅ Coordinates within valid field bounds")
        else:
            print(f"   ⚠️  Some coordinates outside field bounds")
            
            out_of_bounds = sub[
                (sub['x'] < 0) | (sub['x'] > 120) | 
                (sub['y'] < 0) | (sub['y'] > 53.3)
            ]
            if len(out_of_bounds) > 0:
                print(f"   Out of bounds rows: {len(out_of_bounds)}")
                print(f"\n   First 5 out-of-bounds entries:")
                print(out_of_bounds.head())
        
        print(f"\n7. Duplicate ID Check:")
        duplicates = sub[sub.duplicated(subset=['id'], keep=False)]
        if len(duplicates) > 0:
            print(f"   ❌ Found {len(duplicates)} duplicate IDs!")
            print(f"\n   First 10 duplicates:")
            print(duplicates.head(10))
        else:
            print(f"   ✅ No duplicate IDs")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    
    issues = []
    if not cols_match:
        issues.append("Column names don't match")
    if not rows_match:
        issues.append("Row count mismatch")
    if has_na:
        issues.append("Contains missing values")
    if missing_ids:
        issues.append(f"{len(missing_ids)} missing IDs")
    if extra_ids:
        issues.append(f"{len(extra_ids)} extra IDs")
    
    if issues:
        print(f"\n❌ FOUND {len(issues)} ISSUE(S):")
        for idx, issue in enumerate(issues, 1):
            print(f"   {idx}. {issue}")
        print(f"\n💡 Fix these issues before submitting to Kaggle!")
    else:
        print(f"\n✅ ALL CHECKS PASSED!")
        print(f"   Your submission.csv appears to be valid.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
