# 🚀 Quick Start Guide - Kaggle Submission

## For Kaggle Platform

### Step 1: Create Notebook
1. Go to https://www.kaggle.com/competitions/nfl-big-data-bowl-2026
2. Click "Code" → "New Notebook"
3. Settings:
   - **Accelerator**: GPU (recommended) or CPU
   - **Internet**: ON (for package imports)
   - **Persistence**: Files Only

### Step 2: Add Data
The competition data should already be attached. Verify:
```python
!ls /kaggle/input/nfl-big-data-bowl-2026/
```
Should show: `train/`, `test_input.csv`, `test.csv`

### Step 3: Upload Script
1. Click "Add Data" → "Upload"
2. Upload `kaggle_advanced_submission.py`
3. Or copy-paste the entire script into a code cell

### Step 4: Run
```python
!python /kaggle/input/your-upload/kaggle_advanced_submission.py
```

Or if pasted directly:
```python
# Paste entire script here and run cell
```

### Step 5: Submit
1. Wait ~6-7 hours for completion
2. Download `submission.csv` from output
3. Submit to competition

## For Local Testing

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost
```

### Data Structure
```
./
├── train/
│   ├── input_2023_w01.csv
│   ├── output_2023_w01.csv
│   ├── ...
│   ├── input_2023_w18.csv
│   └── output_2023_w18.csv
├── test_input.csv
├── test.csv
└── kaggle_advanced_submission.py
```

### Run
```bash
python kaggle_advanced_submission.py
```

### Output
- `submission.csv` - Ready to upload to Kaggle

## Expected Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Data Loading | 5-10 min | 0:10 |
| Feature Engineering | 50-60 min | 1:10 |
| Feature Preparation | 5-10 min | 1:20 |
| Model Training | 4-5 hours | 6:20 |
| Test Prediction | 10-20 min | 6:40 |
| **Total** | **~6-7 hours** | **6:40** |

## Monitoring Progress

### Console Output
```
================================================================================
PHASE 1: DATA LOADING
================================================================================
[TRAINING] Loading Data (18 weeks)
  ✓ w01: 285,714 input, 32,088 output rows
  ...
✓ Total: 4,880,579 input rows, 562,936 output rows

================================================================================
PHASE 2: FEATURE ENGINEERING
================================================================================
  → Velocity vectors...
  → Coverage intelligence (memory-optimized)...
✓ Feature engineering complete: 227 features

================================================================================
PHASE 4: MODEL TRAINING  ⏰ LONGEST PHASE
================================================================================
[TRAINING ENSEMBLE: ALL_PLAYERS]
Training samples: 450,348
Validation samples: 112,588
Features: 227
Training base models with 3-fold CV...
  Fold 1/3...  ⏰ ~90 min
  Fold 2/3...  ⏰ ~90 min
  Fold 3/3...  ⏰ ~90 min
Training meta-learner (Ridge)...
Training residual refiners...
✓ Training RMSE: 0.3145 yards
✓ Validation RMSE: 0.3298 yards
```

## Troubleshooting

### Script Stops During Coverage Intelligence
**Symptom**: "Killed" or memory error during feature engineering

**Solution**:
```python
# In compute_coverage_features(), line ~300:
chunk_size = 200  # Increase from 100
```

### Time Limit Exceeded (9 hours)
**Solution 1** - Reduce estimators:
```python
# In train_ensemble(), line ~580:
n_estimators=400,  # from 700
n_estimators=300,  # from 600
```

**Solution 2** - Skip residual refinement:
```python
# In train_ensemble(), comment out lines ~680-700:
# residual_x = ...
# residual_y = ...
# Use stacked predictions directly
```

**Solution 3** - Use 2-fold CV:
```python
# In train_ensemble(), line ~570:
cv = GroupKFold(n_splits=2)  # from 3
```

### Accuracy Too Low (<0.35 RMSE)
**Check 1** - Feature scaling working:
```python
print(f"X mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")
# Should be ~0.0 and ~1.0
```

**Check 2** - Predictions in valid range:
```python
print(f"X range: [{predictions[:,0].min():.2f}, {predictions[:,0].max():.2f}]")
print(f"Y range: [{predictions[:,1].min():.2f}, {predictions[:,1].max():.2f}]")
# Should be [0-120] and [0-53.3]
```

**Check 3** - Target weights applied:
```python
print(f"Target receiver weight: {weights[train_pairs['input_is_targeted_receiver'].fillna(0).astype(bool)].mean():.2f}")
# Should be 2.0
```

## Performance Optimization

### Speed Priority
```python
# Reduce trees
n_estimators=400  # base models
iterations=300     # catboost

# Reduce CV folds
cv = GroupKFold(n_splits=2)

# Reduce augmentations
n_augments=2
```

### Memory Priority
```python
# Increase chunk size
chunk_size = 200

# Use float16 (risky)
X.astype("float16")

# Skip some feature groups
# Comment out lines 200-250 (coverage intelligence)
```

### Accuracy Priority
```python
# More trees
n_estimators=1000

# More CV folds
cv = GroupKFold(n_splits=5)

# More augmentations
n_augments=5

# Add more feature groups
# (requires custom code)
```

## Validation

### Before Submission
```python
# Check file exists
import os
assert os.path.exists("submission.csv"), "submission.csv not found!"

# Check format
sub = pd.read_csv("submission.csv")
print(f"Rows: {len(sub)}")  # Should be 5,837
print(f"Columns: {list(sub.columns)}")  # Should be ['id', 'x', 'y']
print(sub.head())

# Check ranges
assert (sub['x'] >= 0).all() and (sub['x'] <= 120).all(), "X out of bounds!"
assert (sub['y'] >= 0).all() and (sub['y'] <= 53.3).all(), "Y out of bounds!"

print("✅ Submission valid!")
```

## Common Issues

### Issue 1: ImportError
```
ModuleNotFoundError: No module named 'catboost'
```
**Solution**: Ensure all packages installed
```bash
pip install catboost xgboost lightgbm scikit-learn
```

### Issue 2: FileNotFoundError
```
FileNotFoundError: train/input_2023_w01.csv not found
```
**Solution**: Check data paths
```python
# Add at start of script:
import os
print(os.listdir("."))
print(os.listdir("./train"))
```

### Issue 3: MemoryError
```
MemoryError: Unable to allocate array
```
**Solution**: Increase chunk size or reduce features
```python
chunk_size = 200  # or higher
# Or comment out coverage intelligence section
```

### Issue 4: Wrong Predictions
```
All predictions are 60, 26.65
```
**Solution**: Test data not found, using dummy values
- Verify test_input.csv and test.csv exist
- Check paths in script

## Success Indicators

✅ Script completes without errors  
✅ Training RMSE < 0.35 yards  
✅ Validation RMSE < 0.38 yards  
✅ submission.csv has 5,837 rows  
✅ All predictions in bounds  
✅ Runtime < 9 hours  
✅ Memory < 30GB  

## Next Steps After First Submission

1. **Analyze Results**: Check leaderboard score
2. **Tune Hyperparameters**: Adjust if needed
3. **Ensemble**: Combine with other models
4. **Feature Selection**: Remove low-importance features
5. **Add Features**: Domain-specific enhancements

## Contact

Issues? Check:
1. Console output for error messages
2. This guide's troubleshooting section
3. Script comments (detailed inline docs)

---

**Happy Kaggling! 🏈📊**
