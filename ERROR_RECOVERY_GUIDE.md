# Checkpoint System & Error Recovery Guide

## Quick Diagnosis

### Check Current State
```powershell
python checkpoint_manager.py list
```

### View Recent Log
```powershell
python checkpoint_manager.py log 50
```

### Check if Training is Running
```powershell
Get-Process python | Select-Object Id, ProcessName, StartTime, CPU, WorkingSet
```

## Current Situation Analysis

Based on your checkpoints, here's what happened:

### ✅ First Run (14:07 - 16:30)
- **Status**: Completed successfully
- **RMSE**: 0.3858 yards
- **Output**: submission.csv created
- **Note**: This was the OLD version (before train/val split changes)

### ⚠️ Second Run (18:02 - 19:24+)
- **Status**: Started with NEW version (train/val split + TTA)
- **Progress**: Got to Phase 4 (model training)
- **Last action**: Started training "Targeted Receiver" model
- **Problem**: No completion message - likely still running or crashed

## What To Do

### Option 1: Check If Still Running

```powershell
# Check Python processes
Get-Process python -ErrorAction SilentlyContinue

# If you see processes with high CPU (>50%), training is ongoing
# If you see processes with low CPU (<5%), they're likely stuck
```

**If stuck/crashed**: Kill the processes
```powershell
Stop-Process -Name python -Force
```

### Option 2: Start Fresh (Recommended)

Since you're at Phase 4 and training with separate train/val + TTA is new, it's best to start fresh:

```powershell
# 1. Clean old checkpoints (keep only latest for reference)
python checkpoint_manager.py clean 1

# 2. Start training fresh
python generate_advanced_submission.py
```

The script will automatically:
- ✅ Ask if you want to resume from checkpoint
- ✅ Show you what checkpoints exist
- ✅ Let you choose to start fresh or resume

### Option 3: Review What Was Completed

```powershell
# View the successful first run metrics
python checkpoint_manager.py view 04_training_complete_20251019_163006.json

# View the first run submission details
python checkpoint_manager.py view 06_final_submission_20251019_163027.pkl
```

## Understanding Your Checkpoints

### Phase 6 Checkpoint (16:30:27)
This is from the FIRST successful run (OLD version):
- Used combined train+val data
- No TTA
- RMSE: 0.3858 yards
- Generated submission.csv

### Phase 3 Checkpoint (19:23:47)
This is from the SECOND run (NEW version):
- Proper train/val split
- Ready for TTA
- Got stuck during training Phase 4

## Recommended Action Plan

### Step 1: Kill Any Stuck Processes
```powershell
Stop-Process -Name python -Force
```

### Step 2: Check Your submission.csv
```powershell
# If it exists and looks good, you already have working predictions
Test-Path submission.csv
Get-Content submission.csv -Head 10
```

### Step 3: Decide: Use Old or Retrain New?

**Use OLD submission (safe, working)**:
- Already completed
- RMSE: 0.3858 yards
- No train/val split (may be overfitted)
- No TTA
- ✅ Ready to submit now

**Retrain with NEW version (better but longer)**:
- Proper train/val split
- Includes TTA
- Expected better generalization
- Takes 6-7 hours
- More honest metrics

### Step 4: If Retraining with NEW Version

```powershell
# Clear the incomplete run
python checkpoint_manager.py clean 1

# Start fresh
python generate_advanced_submission.py
```

When prompted:
```
⚠️  Found checkpoint from Phase 3. Resume from there? (y/n):
```
Type: `n` (start fresh for consistency)

## Common Issues & Fixes

### Issue: "Found checkpoint but training still fails"

**Solution**: Clear ALL checkpoints and start fresh
```powershell
# Delete all checkpoints
Remove-Item checkpoints\*.pkl
Remove-Item checkpoints\*.json

# Keep the log for reference
# Then restart training
python generate_advanced_submission.py
```

### Issue: "Training takes too long"

The new version with train/val split takes **6-7 hours**. This is normal:
- Phase 1-3: ~45 minutes
- Phase 4: ~5-6 hours (model training)
- Phase 5-6: ~15 minutes

Monitor with:
```powershell
Get-Content checkpoints\training_log.txt -Wait
```

### Issue: "Out of memory error"

**Solutions**:
1. Close other applications
2. Reduce max_samples in the script:
   ```python
   max_samples = 400_000  # Reduce from 600_000
   ```
3. Use the old submission if RAM is limited

### Issue: "Different results each run"

This is normal due to:
- Random CV splits
- Different data sampling (max_samples limit)
- Random state in models
 
Expected variation: ±0.01-0.02 yards RMSE

## Checkpoint Files Explained

### 01_data_loaded_*.pkl
- Stores: Data shapes, week info
- Size: Small (~1 KB)
- Resume: Can skip data loading

### 02_features_engineered_*.pkl  
- Stores: Feature count, pair info
- Size: Small (~2 KB)
- Resume: Can skip feature engineering

### 03_features_selected_*.pkl
- Stores: Selected feature names
- Size: Small (~10 KB)
- Resume: Can skip feature selection

### 04_model_*_*.pkl
- Stores: Model metadata, RMSE
- Size: Medium (~50 KB)
- Resume: Individual role completion

### 04_training_complete_*.json
- Stores: All training metrics
- Size: Small (~2 KB)
- Resume: All models done, skip training

### 06_final_submission_*.pkl
- Stores: Submission metadata
- Size: Small (~1 KB)
- Resume: Training fully complete

## Monitoring Commands

### Watch Training Progress Live
```powershell
Get-Content checkpoints\training_log.txt -Wait
```

### Check Training Status
```powershell
python checkpoint_manager.py list
```

### View Latest Metrics
```powershell
# Find the latest training complete JSON
python checkpoint_manager.py view 04_training_complete_20251019_163006.json
```

### Check System Resources
```powershell
# Memory usage
Get-Process python | Select-Object WorkingSet, PrivateMemorySize

# CPU usage
Get-Process python | Select-Object CPU

# Disk space
Get-PSDrive C | Select-Object Used, Free
```

## Error Messages & Solutions

### "FileNotFoundError: train folder not found"
- Check data_root path is correct
- Ensure train/ folder exists with CSV files

### "MemoryError" or "Cannot allocate memory"
- Close other applications
- Reduce max_samples to 400,000 or 300,000
- Consider using old submission

### "KeyboardInterrupt" (Ctrl+C pressed)
- Checkpoints saved up to last phase
- Can resume from last checkpoint
- Or start fresh

### "Models not trained" error
- Training crashed during Phase 4
- Start fresh recommended
- Check system resources

## Best Practices

### ✅ DO
- Monitor training with checkpoint_manager.py
- Keep training_log.txt open in editor
- Check system resources before starting
- Save old submission.csv before retraining
- Clean old checkpoints periodically

### ❌ DON'T
- Don't interrupt during model training (Phase 4)
- Don't modify checkpoint files manually
- Don't run multiple training processes simultaneously
- Don't delete training_log.txt (useful for debugging)

## Quick Reference

```powershell
# List checkpoints
python checkpoint_manager.py list

# View log
python checkpoint_manager.py log 50

# View specific checkpoint
python checkpoint_manager.py view <filename>

# Clean old checkpoints
python checkpoint_manager.py clean 1

# Start training
python generate_advanced_submission.py

# Monitor training
Get-Content checkpoints\training_log.txt -Wait

# Kill stuck processes
Stop-Process -Name python -Force

# Check if training is running
Get-Process python
```

## Summary for Your Current Situation

1. ✅ **You have a working submission** from the first run (16:30)
2. ⚠️ **Second run is incomplete** - stopped during Phase 4
3. 📋 **Recommendation**: 
   - If deadline is soon → Use the existing submission.csv
   - If you have time → Kill processes, clean checkpoints, restart fresh

**To restart fresh**:
```powershell
Stop-Process -Name python -Force
python checkpoint_manager.py clean 1
python generate_advanced_submission.py
# When prompted, choose 'n' to start fresh
```

Then monitor with:
```powershell
Get-Content checkpoints\training_log.txt -Wait
```

Training will take ~6-7 hours to complete.
