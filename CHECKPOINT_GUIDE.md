# Checkpoint System Guide

## Overview
The training script now includes comprehensive checkpoint functionality with timestamps to track progress throughout the 6-7 hour training run.

## Features

### 1. **Progress Logging with Timestamps**
All major events are logged to `checkpoints/training_log.txt` with timestamps:
```
[2024-01-15 14:30:45] 🚀 STARTING NFL BIG DATA BOWL 2026 - ADVANCED SUBMISSION GENERATION
[2024-01-15 14:30:46] Data root: c:\nfl-big-data-bowl-2026-prediction
[2024-01-15 14:31:20] 📊 Phase 1/6: Loading training data (15 weeks)
```

### 2. **Data Checkpoints**
Key data is saved to pickle files in the `checkpoints/` directory:
- `01_data_loaded_*.pkl` - Data shapes and metadata
- `02_features_engineered_*.pkl` - Feature counts and column names
- `03_features_selected_*.pkl` - Selected feature list
- `04_model_*.pkl` - Model performance for each role
- `06_final_submission_*.pkl` - Final submission metadata

### 3. **Metrics Tracking**
JSON files track performance metrics:
- `04_training_complete_*.json` - Overall RMSE and role-wise performance

## Training Phases

### Phase 1: Data Loading (5-10 minutes)
```
📊 Phase 1/6: Loading training data (15 weeks)
✓ Data loaded: 1,234,567 input rows, 89,012 output rows
```

### Phase 2: Feature Engineering (10-15 minutes)
```
🔧 Phase 2/6: Engineering features
✓ Feature engineering complete: 145 features
✓ Training pairs created: 456,789 samples
```

### Phase 3: Feature Selection (15-20 minutes)
```
🎯 Phase 3/6: Feature selection from 145 candidates
✓ Feature selection complete: 98 features selected
```

### Phase 4: Model Training (5-6 hours)
```
🤖 Phase 4/6: Training role-specific ensemble models
Training 3 role groups × 5 models × 2 coordinates × 3 CV folds = 90 model fits

  → Training: Targeted Receiver (123,456 samples)
  ✓ Targeted Receiver complete: RMSE=0.3250 yards
  
  → Training: Defensive Coverage (234,567 samples)
  ✓ Defensive Coverage complete: RMSE=0.3380 yards
  
  → Training: Other Players (98,766 samples)
  ✓ Other Players complete: RMSE=0.3420 yards

✓ All role models trained - Aggregated RMSE: 0.3350 yards
```

### Phase 5: Test Data Preparation (5-10 minutes)
```
🔮 Phase 5/6: Loading and preparing test data
Test input: 45,678 rows, Test template: 5,837 predictions
```

### Phase 6: Prediction Generation (5-10 minutes)
```
🎯 Phase 6/6: Generating predictions for each role
  → Predicting: Targeted Receiver (1,234 samples)
  ✓ Targeted Receiver predictions complete
  → Predicting: Defensive Coverage (2,345 samples)
  ✓ Defensive Coverage predictions complete
  → Predicting: Other Players (2,258 samples)
  ✓ Other Players predictions complete

🎉 SUCCESS! Submission generated: 5,837 predictions
```

## Checkpoint Files

### Directory Structure
```
checkpoints/
├── training_log.txt                           # Main progress log
├── 01_data_loaded_20240115_143145.pkl        # Data metadata
├── 02_features_engineered_20240115_143820.pkl # Feature info
├── 03_features_selected_20240115_145230.pkl   # Selected features
├── 04_model_targeted_receiver_20240115_163445.pkl
├── 04_model_defensive_coverage_20240115_181230.pkl
├── 04_model_other_players_20240115_194512.pkl
├── 04_training_complete_20240115_194512.json  # Overall metrics
└── 06_final_submission_20240115_195030.pkl    # Final metadata
```

### File Contents

**training_log.txt** - Human-readable progress log:
```
[2024-01-15 14:30:45] 🚀 STARTING NFL BIG DATA BOWL 2026
[2024-01-15 14:31:45] ✓ Data loaded: 1,234,567 input rows
[2024-01-15 14:38:20] ✓ Feature engineering complete: 145 features
...
[2024-01-15 19:50:30] 🎉 SUCCESS! Submission generated: 5,837 predictions
```

**Pickle Files (.pkl)** - Python dictionaries with metadata:
```python
{
    "input_shape": (1234567, 145),
    "output_shape": (89012, 3),
    "train_weeks": ["w01", "w02", ...],
    "val_weeks": ["w16", "w17", "w18"]
}
```

**JSON Files (.json)** - Performance metrics:
```json
{
  "overall_rmse": 0.3350,
  "role_rmses": {
    "Targeted Receiver": 0.3250,
    "Defensive Coverage": 0.3380,
    "Other Players": 0.3420
  },
  "role_counts": {
    "Targeted Receiver": 123456,
    "Defensive Coverage": 234567,
    "Other Players": 98766
  },
  "total_samples": 456789
}
```

## Monitoring Progress

### Real-Time Monitoring
Watch the log file update in real-time:
```powershell
Get-Content checkpoints\training_log.txt -Wait
```

### Check Latest Status
View the last 20 lines:
```powershell
Get-Content checkpoints\training_log.txt -Tail 20
```

### Estimate Time Remaining
Look for phase markers:
- Phase 1 (Data Loading): ~5-10 min
- Phase 2 (Features): ~10-15 min
- Phase 3 (Selection): ~15-20 min
- Phase 4 (Training): ~5-6 hours ⏱️ (longest phase)
- Phase 5 (Test Prep): ~5-10 min
- Phase 6 (Prediction): ~5-10 min

**Total Expected Runtime**: 6-7 hours

## Utility Functions

### log_checkpoint(message, also_print=True)
Logs a timestamped message to both console and file.

### save_checkpoint(name, data, suffix="")
Saves data dictionary to pickle file with timestamp.

### save_metrics(name, metrics)
Saves metrics dictionary to JSON file with timestamp.

## Recovery (Future Enhancement)

While the current implementation doesn't include automatic recovery, checkpoint files can be used to:
1. Analyze where the script stopped
2. Manually resume from saved data
3. Debug issues by examining intermediate states

## Performance Impact

- **Logging overhead**: < 1 second per checkpoint
- **File I/O**: < 5 seconds per save
- **Total overhead**: < 2 minutes over entire run (< 0.5% of runtime)

## Best Practices

1. **Monitor during long runs**: Keep `training_log.txt` open in a text editor
2. **Check timestamps**: Verify progress is being made
3. **Review metrics**: Compare role RMSEs to expected values
4. **Archive checkpoints**: Keep checkpoint files for successful runs as reference
5. **Disk space**: Checkpoint files total ~50-100 MB, clean up old runs periodically

## Example Output Timeline

```
14:30:45 - Start
14:31:45 - Data loaded
14:38:20 - Features engineered  
14:52:30 - Features selected
14:53:15 - Scaling complete
14:53:20 - Start training models...
16:34:45 - Targeted Receiver complete (1h 41m into training)
18:12:30 - Defensive Coverage complete (3h 19m into training)
19:45:12 - Other Players complete (4h 52m into training)
19:50:15 - Test data prepared
19:50:30 - Predictions complete
TOTAL: ~5 hours 20 minutes
```

## Troubleshooting

### No checkpoint files created
- Check that `checkpoints/` directory was created
- Verify write permissions

### Training log not updating
- Script may be in a long computation
- Check if Python process is still running
- Look at Phase 4 - model training takes 5-6 hours

### Unexpected RMSE values
- Compare to expected: ~0.33-0.34 yards overall
- Role-specific should be within 10% of expected
- Check `04_training_complete_*.json` for details

## Summary

The checkpoint system provides:
- ✅ Real-time progress visibility
- ✅ Timestamped logging
- ✅ Intermediate data snapshots
- ✅ Performance metrics tracking
- ✅ Debugging support
- ✅ Minimal performance overhead (<0.5%)

Monitor `checkpoints/training_log.txt` to track your 6-7 hour training run!
