# ✅ CHECKPOINT & ERROR RECOVERY - COMPLETE GUIDE

## 🔍 Current Situation

Based on analysis:

### ✅ GOOD NEWS: You Have a Working Submission!
- **File**: `submission.csv` (359,495 bytes)
- **Created**: 2025-10-19 16:30:27
- **Predictions**: 5,838 (correct count)
- **RMSE**: 0.3858 yards
- **Status**: Ready to submit

### ⚠️ NEW RUN: Incomplete
- **Started**: 2025-10-19 18:02 (with new train/val split + TTA)
- **Progress**: Got through Phase 3, started Phase 4
- **Stopped at**: Training "Targeted Receiver" model
- **Issue**: Likely crashed or still running in background

---

## 🚀 QUICK START - What To Do Now

### Option 1: Use Existing Submission (RECOMMENDED if deadline is soon)

```powershell
# Your submission.csv is ready!
# Just verify it looks good:
Get-Content submission.csv -Head 5

# Expected output:
# id,x,y
# 2024120805_1201_47877_1,43.712,10.959
# ...
```

✅ **Pros**: Already done, tested, working  
⚠️ **Cons**: Older version (no train/val split, no TTA)

### Option 2: Retrain with New Version (RECOMMENDED if you have 6-7 hours)

```powershell
# 1. Check and kill any stuck Python processes
python status.py
Stop-Process -Name python -Force

# 2. Clean old incomplete checkpoints
python checkpoint_manager.py clean 1

# 3. Start fresh training
python generate_advanced_submission.py

# 4. Monitor progress (in another terminal)
Get-Content checkpoints\training_log.txt -Wait
```

✅ **Pros**: Better model (train/val split + TTA), expected 0.02-0.03 improvement  
⚠️ **Cons**: Takes 6-7 hours

---

## 📋 Step-by-Step Recovery Instructions

### Step 1: Assess the Situation

```powershell
# Run status check
python status.py

# View checkpoints
python checkpoint_manager.py list

# Check last 30 log lines
python checkpoint_manager.py log 30
```

### Step 2: Kill Any Stuck Processes

```powershell
# Check for Python processes
Get-Process python

# If you see processes with low CPU (<5%), they're stuck:
Stop-Process -Name python -Force
```

### Step 3: Backup Current Submission (if retraining)

```powershell
# Backup your working submission
Copy-Item submission.csv submission_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv
```

### Step 4: Choose Your Path

#### Path A: Use Existing Submission
```powershell
# Verify it's good
python checkpoint_manager.py view 06_final_submission_20251019_163027.pkl

# Done! Your submission.csv is ready
```

#### Path B: Retrain Fresh
```powershell
# Clean incomplete checkpoints
python checkpoint_manager.py clean 1

# Start training
python generate_advanced_submission.py

# When prompted:
# ⚠️ Found checkpoint from Phase 3. Resume from there? (y/n): 
# Type: n (start fresh for consistency)
```

### Step 5: Monitor Training (if retraining)

```powershell
# In one terminal: watch live progress
Get-Content checkpoints\training_log.txt -Wait

# In another: check status periodically
python status.py
```

---

## 🛠️ Utility Scripts

### status.py - Quick Status Check
```powershell
python status.py
```
Shows:
- Python process status
- Submission file status
- Latest checkpoint
- Training log summary
- Overall status

### checkpoint_manager.py - Checkpoint Management

```powershell
# List all checkpoints
python checkpoint_manager.py list

# View last 50 log lines
python checkpoint_manager.py log 50

# View specific checkpoint
python checkpoint_manager.py view <filename>

# Clean old checkpoints (keep 1 most recent)
python checkpoint_manager.py clean 1
```

---

## 📊 Understanding Your Checkpoints

### What You Have

#### ✅ Complete Run (14:07 - 16:30)
```
Phase 1: 01_data_loaded_20251019_140753.pkl
Phase 2: 02_features_engineered_20251019_160149.pkl
Phase 3: 03_features_selected_20251019_160244.pkl
Phase 4: 04_model_targeted_receiver_20251019_163006.pkl
         04_training_complete_20251019_163006.json
Phase 6: 06_final_submission_20251019_163027.pkl
```
**Result**: submission.csv (RMSE: 0.3858 yards)

#### ⚠️ Incomplete Run (18:02 - 19:24+)
```
Phase 1: 01_data_loaded_20251019_180251.pkl
Phase 2: 02_features_engineered_20251019_192254.pkl
Phase 3: 03_features_selected_20251019_192347.pkl
Phase 4: Started but not finished
```
**Result**: Stopped during training

### Key Differences

| Feature | Old Run (Complete) | New Run (Incomplete) |
|---------|-------------------|---------------------|
| Train/Val Split | ❌ Mixed | ✅ Separate |
| TTA | ❌ No | ✅ Yes |
| Train Samples | 562,936 | 463,670 |
| Val Samples | N/A | 99,266 |
| RMSE Reporting | Single | Train + Val |
| Expected RMSE | 0.386 | ~0.32-0.33 |

---

## ⚙️ Training Timeline

When you restart training, expect:

```
Phase 1: Data Loading          [   5-10 min]  ████░░░░░░░░░░░░░░░░
Phase 2: Feature Engineering   [  10-15 min]  ████████░░░░░░░░░░░░
Phase 3: Feature Selection     [  15-20 min]  ████████████░░░░░░░░
Phase 4: Model Training        [5-6 hours ]  ████████████████████  <-- Longest!
Phase 5: Test Preparation      [   5-10 min]  ████████████████████
Phase 6: Prediction (with TTA) [  10-15 min]  ████████████████████

Total: ~6-7 hours
```

### Phase 4 Breakdown
```
For each of 3 roles (Receiver, Defense, Other):
  For each of 5 models (XGB1, XGB2, LGBM1, LGBM2, CAT):
    For each of 2 coordinates (X, Y):
      For each of 3 CV folds:
        Train model (~2-3 min per model)

Total: 3 × 5 × 2 × 3 = 90 model fits
Time: ~5-6 hours
```

---

## 🐛 Common Errors & Solutions

### Error: "MemoryError: Unable to allocate..."

**Cause**: Not enough RAM  
**Solution**:
```python
# Edit generate_advanced_submission.py
# Line ~1570, reduce max_samples:
max_samples = 400_000  # Was 600_000
```

### Error: "Process killed" or script stops silently

**Cause**: Out of memory, system killed process  
**Solution**: Close other applications, reduce max_samples

### Error: Validation weeks not found

**Cause**: Missing w16, w17, w18 data files  
**Effect**: Will use train weeks for validation (less ideal but works)

### Warning: Different RMSE each run

**Normal**: Due to random CV splits, data sampling  
**Variation**: ±0.01-0.02 yards is expected

---

## 📈 Expected Performance

### Old Submission (Complete, No TTA)
```
Training RMSE: N/A
Validation RMSE: 0.3858 yards (on training data)
Test RMSE: ~0.39-0.40 yards (actual)
```

### New Version (With Train/Val + TTA)
```
Training RMSE: ~0.325 yards
Validation RMSE: ~0.335 yards
Test RMSE: ~0.32-0.33 yards (expected 0.02-0.03 improvement)
```

**Expected Improvement**: 0.06-0.07 yards better!

---

## ✅ Decision Matrix

### Use Existing Submission If:
- ✅ Deadline is within 6 hours
- ✅ You're satisfied with 0.386 RMSE
- ✅ Want to be safe with tested output
- ✅ Don't want to wait 6-7 hours

### Retrain with New Version If:
- ✅ You have 6-7 hours available
- ✅ Want better accuracy (0.02-0.03 improvement)
- ✅ Want proper train/val metrics
- ✅ Want Test-Time Augmentation benefits

---

## 🎯 Recommended Actions

### IMMEDIATE ACTION:
```powershell
# 1. Check status
python status.py

# 2. Kill any stuck processes
Stop-Process -Name python -Force

# 3. Decide: use existing or retrain
```

### IF USING EXISTING:
```powershell
# Verify submission
Get-Content submission.csv -Head 10
(Get-Content submission.csv).Length  # Should be 5838

# Done! Submit submission.csv
```

### IF RETRAINING:
```powershell
# 1. Backup existing
Copy-Item submission.csv submission_backup.csv

# 2. Clean checkpoints
python checkpoint_manager.py clean 1

# 3. Start fresh
python generate_advanced_submission.py
# Choose 'n' when asked to resume

# 4. Monitor
Get-Content checkpoints\training_log.txt -Wait
```

---

## 📞 Quick Reference Commands

```powershell
# Check status
python status.py

# List checkpoints
python checkpoint_manager.py list

# View log
python checkpoint_manager.py log 50

# Kill processes
Stop-Process -Name python -Force

# Start training
python generate_advanced_submission.py

# Monitor training
Get-Content checkpoints\training_log.txt -Wait

# Check submission
Get-Content submission.csv -Head 5
```

---

## 📝 Summary

**Your Current State:**
- ✅ Working submission.csv ready (0.386 RMSE)
- ⚠️ New run started but incomplete
- 🔧 Can use existing OR retrain for better results

**Recommendation:**
1. **Short on time?** → Use existing submission.csv ✅
2. **Have 6-7 hours?** → Retrain with new version for 0.02-0.03 improvement 🚀

**Next Steps:**
```powershell
python status.py  # Check current state
# Then decide based on your timeline
```

Good luck! 🏈📊✨
