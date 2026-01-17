# NFL Big Data Bowl 2026 - Player Position Prediction

## Challenge Overview

**Competition:** NFL Big Data Bowl 2026  
**Objective:** Predict the future (x, y) positions of NFL players after a pass is thrown, using tracking data from before the throw.

### Problem Statement
Given player tracking data (position, velocity, acceleration, orientation) at the moment a pass is thrown, predict where each player will be located in subsequent frames until the ball lands. This involves understanding player movement patterns, route running, defensive reactions, and ball trajectory physics.

### Input Data
- **Player Tracking:** x, y coordinates, speed, acceleration, orientation, direction
- **Player Context:** Position, role (Targeted Receiver, Defensive Coverage, etc.), physical attributes
- **Play Context:** Ball landing position, play direction, yardline

### Target
- Predict future (x, y) coordinates for each player across multiple frames
- Evaluated using **Root Mean Squared Error (RMSE)** in yards

---

## Results Obtained

| Metric | Value |
|--------|-------|
| **Training RMSE** | 0.6705 yards |
| **Validation RMSE** | 0.6809 yards |
| Train/Val Gap | 0.0104 yards (good generalization) |

---

## Methodology

### 1. Data Preprocessing
- Loaded 18 weeks of NFL tracking data (train: weeks 1-15, val: weeks 16-18)
- Computed **delta targets** (future position - current position) instead of absolute positions
- Created player pairs linking input frames to output predictions

### 2. Feature Engineering (200+ features)
- **Kinematic Features:** Velocity components (vx, vy), acceleration, speed
- **Relative Features:** Distance/angle to ball landing spot, distance to target player
- **Player Context:** Role encoding, position encoding, physical attributes
- **Spatial Features:** Field position, distance to sidelines/endzone
- **Interaction Features:** Player-to-player distances, coverage relationships

### 3. Feature Selection
- Permutation importance with LightGBM
- Retained top 80% of features (~214 features)
- Removed low-variance and highly correlated features

### 4. Model Architecture: 10-Model Stacked Ensemble

| Model | Type | Purpose |
|-------|------|---------|
| XGBoost (x2) | Gradient Boosting | Primary predictors with different hyperparameters |
| LightGBM (x2) | Gradient Boosting | Fast training, handles large data |
| CatBoost | Gradient Boosting | Robust to overfitting |
| RandomForest | Bagging | Captures non-linear interactions |
| ExtraTrees | Bagging | Better variance reduction |
| HistGradientBoosting | Gradient Boosting | Native sklearn, fast |
| MLPRegressor | Neural Network | Captures complex patterns |
| HuberRegressor | Linear (Robust) | Stable baseline, outlier resistant |

### 5. Training Strategy
- **Cross-Validation:** 3-fold GroupKFold (grouped by game_id to prevent leakage)
- **Separate Models:** X and Y coordinates trained independently
- **Sample Weighting:** 3x weight for targeted receivers (most important predictions)
- **Meta-Learning:** Ridge regression stacking of base model predictions
- **Residual Refinement:** Secondary gradient boosting to correct systematic errors

### 6. Inference
- **Test-Time Augmentation (TTA):** 3 augmented versions per sample
- **Physics Constraints:** Max acceleration 4 yd/s², max speed 25 yd/s
- **Field Clipping:** Predictions clipped to valid field boundaries (0-120, 0-53.3)

---

## Repository Structure

```
├── generate_advanced_submission.py   # Main training pipeline (2500 lines)
├── model_training_submission.ipynb   # Jupyter notebook version
├── submission.csv                    # Final predictions
├── submission (10).csv               # Alternative submission
└── README.md                         # This file
```

## How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn xgboost lightgbm catboost

# Run training and generate submission
python generate_advanced_submission.py --fresh
```

**Expected runtime:** ~8 hours on CPU  
**Memory requirement:** ~16-30 GB RAM

---

## Key Insights

1. **Delta prediction outperforms absolute:** Predicting position change rather than absolute position reduces error significantly

2. **Ensemble diversity matters:** Combining different algorithm families (boosting, bagging, neural, linear) improves robustness

3. **Separate X/Y models:** Treating coordinates independently allows specialized optimization

4. **Physics constraints help:** Enforcing realistic movement limits reduces unreasonable predictions

5. **Targeted receivers are hardest:** Route runners have highest prediction variance due to sudden cuts and breaks

---

## Future Improvements

- [ ] Hyperparameter tuning with Bayesian optimization
- [ ] Neural network meta-learner instead of Ridge
- [ ] Attention-based sequence models for temporal patterns
- [ ] More aggressive feature diversity (feature bagging per model)
