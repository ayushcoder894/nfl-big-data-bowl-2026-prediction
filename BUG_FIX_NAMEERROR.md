# 🔧 Bug Fix - NameError in Coverage Intelligence

## Date: October 20, 2025

## Error Encountered

```python
NameError: name 'frame' is not defined
```

**Location**: Line 782 in `generate_advanced_submission.py`
**Function**: `compute_frame_level_coverage()`

## Root Cause

When adding the advanced defensive coverage features, I used the variable name `frame` in the latter part of the function, but the parameter was named `frame_df`. This caused a NameError when the function tried to access `frame`.

## Code Before Fix

```python
def compute_frame_level_coverage(frame_df: pd.DataFrame) -> pd.DataFrame:
    # ... early code uses frame_df correctly ...
    
    # BUG: Later code used 'frame' instead of 'frame_df'
    coverage_weights = np.zeros(len(frame), dtype="float32")  # ❌ ERROR
    frame["distance_to_targeted_receiver"] = dist_to_target    # ❌ ERROR
    frame["angle_to_receiver_ball_path"] = angle_to_path       # ❌ ERROR
    # ... more references to 'frame' ...
    
    return frame  # ❌ ERROR
```

## Fix Applied

Changed all instances of `frame` to `frame_df` throughout the function:

```python
def compute_frame_level_coverage(frame_df: pd.DataFrame) -> pd.DataFrame:
    # ... code ...
    
    # FIXED: Consistent variable naming
    coverage_weights = np.zeros(len(frame_df), dtype="float32")  # ✅
    frame_df.loc[:, "distance_to_targeted_receiver"] = dist_to_target  # ✅
    frame_df.loc[:, "angle_to_receiver_ball_path"] = angle_to_path     # ✅
    # ... all references now use frame_df ...
    
    return frame_df  # ✅
```

## Changed Lines

- Line 782: `len(frame)` → `len(frame_df)`
- Line 790: `frame["distance_to_targeted_receiver"]` → `frame_df.loc[:, "distance_to_targeted_receiver"]`
- Line 791: `frame["angle_to_receiver_ball_path"]` → `frame_df.loc[:, "angle_to_receiver_ball_path"]`
- Line 792: `frame["coverage_responsibility"]` → `frame_df.loc[:, "coverage_responsibility"]`
- Line 794-796: Similar changes for else clause
- Line 799: `frame[target_mask]` → `frame_df[target_mask]`
- Lines 809-812: `frame.loc[defense_mask, ...]` → `frame_df.loc[defense_mask, ...]`
- Lines 820-823: Similar changes for angle coverage features
- Lines 825-828: Similar changes for else clause
- Lines 832-833: `frame["ball_land_x"]` → `frame_df["ball_land_x"]`
- Line 836: `frame[["vx", "vy"]]` → `frame_df[["vx", "vy"]]`
- Line 839: `len(frame)` → `len(frame_df)`
- Line 846-847: `len(frame)` → `len(frame_df)`
- Line 849: `frame["interception_angle"]` → `frame_df.loc[:, "interception_angle"]`
- Line 851: `return frame` → `return frame_df`

## Total Changes

**26 instances** of `frame` changed to `frame_df`

## Validation

✅ Training restarted successfully
✅ Data loading Phase 1 completed
✅ Feature engineering Phase 2 in progress
✅ No more NameError

## Status

**FIXED** - Training now running successfully with all advanced features enabled.

## Prevention

This type of error can be prevented by:
1. Using consistent variable naming from the start
2. Running incremental tests after adding new code
3. Using an IDE with variable highlighting
4. Code review before committing

## Impact

- **Downtime**: ~2 minutes (quick fix)
- **Data loss**: None (checkpoints preserved)
- **Accuracy impact**: None (fix was naming only, no logic changed)
