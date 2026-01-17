# NFL Big Data Bowl 2026 - Player Position Prediction

## Best Results
| Metric | Value |
|--------|-------|
| Training RMSE | 0.6705 yards |
| Validation RMSE | 0.6809 yards |

## Files
- `generate_advanced_submission.py` - Main training pipeline (10-model ensemble)
- `model_training_submission.ipynb` - Jupyter notebook version
- `submission.csv` / `submission (10).csv` - Final predictions

## Models Used
XGBoost (x2), LightGBM (x2), CatBoost, RandomForest, ExtraTrees, HistGradientBoosting, MLP, HuberRegressor

## Run
```bash
python generate_advanced_submission.py --fresh
```
