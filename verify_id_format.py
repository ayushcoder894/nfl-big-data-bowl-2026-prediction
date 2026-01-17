"""Verify submission ID format matches expected pattern: {game_id}_{play_id}_{nfl_id}_{frame_id}."""

import re
from pathlib import Path

import pandas as pd


def validate_id_format(id_value: str) -> tuple[bool, str]:
    """
    Check if ID matches pattern {game_id}_{play_id}_{nfl_id}_{frame_id}.
    Returns (is_valid, error_message).
    """
    pattern = r'^(\d+)_(\d+)_(\d+)_(\d+)$'
    match = re.match(pattern, str(id_value))
    
    if not match:
        return False, f"Does not match pattern game_id_play_id_nfl_id_frame_id"
    
    game_id, play_id, nfl_id, frame_id = match.groups()
    
    # Basic sanity checks
    if len(game_id) != 10:
        return False, f"game_id should be 10 digits, got {len(game_id)}"
    
    if int(frame_id) < 1:
        return False, f"frame_id should be >= 1, got {frame_id}"
    
    return True, "Valid"


def main() -> None:
    print("=" * 70)
    print("ID FORMAT VALIDATOR")
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
    print("ID FORMAT ANALYSIS")
    print("=" * 70)
    
    print(f"\n1. Sample IDs from submission (first 5):")
    for idx, id_val in enumerate(sub['id'].head(), 1):
        print(f"   {idx}. {id_val}")
    
    print(f"\n2. Sample IDs from template (first 5):")
    for idx, id_val in enumerate(template['id'].head(), 1):
        print(f"   {idx}. {id_val}")
    
    print(f"\n3. Validating submission ID format...")
    invalid_ids = []
    
    for idx, id_val in enumerate(sub['id']):
        is_valid, error = validate_id_format(id_val)
        if not is_valid:
            invalid_ids.append((idx, id_val, error))
    
    if invalid_ids:
        print(f"\n   ❌ Found {len(invalid_ids)} invalid IDs!")
        print(f"\n   First 10 invalid IDs:")
        for idx, (row_num, id_val, error) in enumerate(invalid_ids[:10], 1):
            print(f"      {idx}. Row {row_num}: '{id_val}' - {error}")
    else:
        print(f"   ✅ All {len(sub)} IDs match expected format!")
    
    print(f"\n4. Validating template ID format...")
    invalid_template_ids = []
    
    for idx, id_val in enumerate(template['id']):
        is_valid, error = validate_id_format(id_val)
        if not is_valid:
            invalid_template_ids.append((idx, id_val, error))
    
    if invalid_template_ids:
        print(f"   ⚠️  Template has {len(invalid_template_ids)} non-standard IDs")
    else:
        print(f"   ✅ Template IDs all match expected format")
    
    print(f"\n5. Extracting ID components (first 3 from submission)...")
    pattern = r'^(\d+)_(\d+)_(\d+)_(\d+)$'
    
    for idx, id_val in enumerate(sub['id'].head(3), 1):
        match = re.match(pattern, str(id_val))
        if match:
            game_id, play_id, nfl_id, frame_id = match.groups()
            print(f"\n   ID {idx}: {id_val}")
            print(f"      game_id:  {game_id}")
            print(f"      play_id:  {play_id}")
            print(f"      nfl_id:   {nfl_id}")
            print(f"      frame_id: {frame_id}")
    
    print(f"\n6. ID Component Statistics:")
    
    # Parse all valid submission IDs
    valid_ids = []
    for id_val in sub['id']:
        match = re.match(pattern, str(id_val))
        if match:
            game_id, play_id, nfl_id, frame_id = match.groups()
            valid_ids.append({
                'game_id': game_id,
                'play_id': int(play_id),
                'nfl_id': int(nfl_id),
                'frame_id': int(frame_id),
            })
    
    if valid_ids:
        ids_df = pd.DataFrame(valid_ids)
        print(f"\n   Unique game_ids:  {ids_df['game_id'].nunique()}")
        print(f"   Unique play_ids:  {ids_df['play_id'].nunique()}")
        print(f"   Unique nfl_ids:   {ids_df['nfl_id'].nunique()}")
        print(f"   Frame ID range:   {ids_df['frame_id'].min()} - {ids_df['frame_id'].max()}")
        
        print(f"\n   game_ids present:")
        for game_id in sorted(ids_df['game_id'].unique()):
            count = (ids_df['game_id'] == game_id).sum()
            print(f"      {game_id}: {count} predictions")
    
    print(f"\n7. Comparing submission vs template IDs...")
    
    template_ids_set = set(template['id'])
    submission_ids_set = set(sub['id'])
    
    missing = template_ids_set - submission_ids_set
    extra = submission_ids_set - template_ids_set
    
    if missing or extra:
        print(f"   ❌ ID sets don't match!")
        print(f"      Missing from submission: {len(missing)}")
        print(f"      Extra in submission:     {len(extra)}")
        
        if missing:
            print(f"\n   First 5 missing IDs:")
            for idx, id_val in enumerate(list(missing)[:5], 1):
                print(f"      {idx}. {id_val}")
        
        if extra:
            print(f"\n   First 5 extra IDs:")
            for idx, id_val in enumerate(list(extra)[:5], 1):
                print(f"      {idx}. {id_val}")
    else:
        print(f"   ✅ Submission and template have identical ID sets!")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    
    issues = []
    if invalid_ids:
        issues.append(f"{len(invalid_ids)} IDs with invalid format")
    if missing:
        issues.append(f"{len(missing)} required IDs missing")
    if extra:
        issues.append(f"{len(extra)} unexpected extra IDs")
    
    if issues:
        print(f"\n❌ FOUND {len(issues)} ISSUE(S):")
        for idx, issue in enumerate(issues, 1):
            print(f"   {idx}. {issue}")
    else:
        print(f"\n✅ ALL ID FORMAT CHECKS PASSED!")
        print(f"   All IDs match pattern: {{game_id}}_{{play_id}}_{{nfl_id}}_{{frame_id}}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
