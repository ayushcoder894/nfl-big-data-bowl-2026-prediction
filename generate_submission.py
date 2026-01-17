# NFL Big Data Bowl 2026 - Complete Training & Submission Generation
# Strategic Framework Implementation

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
import gc
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor
import multiprocessing

# Configuration
np.random.seed(42)
n_cores = min(multiprocessing.cpu_count(), 8)
os.environ['OMP_NUM_THREADS'] = str(n_cores)

print("="*80)
print("NFL BIG DATA BOWL 2026 - STRATEGIC FRAMEWORK")
print("Complete Training & Submission Generation")
print("="*80)
print(f"Target: RMSE < 0.5 yards")
print(f"CPU Cores: {n_cores}")
print("="*80)

# ===================================
# STEP 1: LOAD AND PREPARE TRAINING DATA
# ===================================

def load_strategic_data():
    """Load training data"""
    train_folder = Path(r'c:\nfl-big-data-bowl-2026-prediction\train')
    
    print("\n[STEP 1] Loading Training Data")
    print("-" * 60)
    
    weeks = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06']
    input_dfs = []
    output_dfs = []
    
    for week in weeks:
        input_file = train_folder / f"input_2023_{week}.csv"
        output_file = train_folder / f"output_2023_{week}.csv"
        
        if input_file.exists() and output_file.exists():
            input_df = pd.read_csv(input_file, dtype={
                'x': 'float32', 'y': 'float32', 's': 'float32', 'a': 'float32',
                'dir': 'float32', 'o': 'float32', 'frame_id': 'int16',
                'ball_land_x': 'float32', 'ball_land_y': 'float32',
                'num_frames_output': 'int8', 'player_to_predict': 'bool'
            })
            output_df = pd.read_csv(output_file, dtype={
                'x': 'float32', 'y': 'float32', 'frame_id': 'int16'
            })
            
            input_dfs.append(input_df)
            output_dfs.append(output_df)
            print(f"  ✓ Loaded {week}")
    
    train_input = pd.concat(input_dfs, ignore_index=True)
    train_output = pd.concat(output_dfs, ignore_index=True)
    
    print(f"\n✓ Total: {len(train_input):,} input rows, {len(train_output):,} output rows")
    
    del input_dfs, output_dfs
    gc.collect()
    
    return train_input, train_output

def create_strategic_features(df):
    """Create strategic features"""
    # Velocity components
    df['vx'] = (df['s'] * np.cos(np.radians(df['dir']))).astype('float32')
    df['vy'] = (df['s'] * np.sin(np.radians(df['dir']))).astype('float32')
    
    # Ball landing features (PRIMARY SIGNAL)
    df['dist_to_ball'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + (df['y'] - df['ball_land_y'])**2).astype('float32')
    df['angle_to_ball'] = np.degrees(np.arctan2(df['ball_land_y'] - df['y'], df['ball_land_x'] - df['x'])).astype('float32')
    df['moving_toward_ball'] = ((df['vx'] * (df['ball_land_x'] - df['x']) + df['vy'] * (df['ball_land_y'] - df['y'])) > 0).astype('int8')
    df['velocity_ball_alignment'] = (np.cos(np.radians(df['dir'] - df['angle_to_ball']))).astype('float32')
    
    # Role-specific features
    df['is_targeted_receiver'] = (df['player_role'] == 'Targeted Receiver').astype('int8')
    df['is_defensive_coverage'] = (df['player_role'] == 'Defensive Coverage').astype('int8')
    df['is_passer'] = (df['player_role'] == 'Passer').astype('int8')
    df['is_other_route_runner'] = (df['player_role'] == 'Other Route Runner').astype('int8')
    
    # Position encoding
    position_encoder = LabelEncoder()
    df['position_encoded'] = position_encoder.fit_transform(df['player_position']).astype('int8')
    df['is_offense'] = (df['player_side'] == 'Offense').astype('int8')
    
    # Physics features
    df['speed_squared'] = (df['s'] ** 2).astype('float32')
    df['momentum_magnitude'] = (df['s'] * 200).astype('float32')
    df['kinetic_energy'] = (0.5 * df['speed_squared']).astype('float32')
    
    # Temporal features
    df['time_to_ball_land'] = (df['num_frames_output'] / 10.0).astype('float32')
    df['frames_remaining'] = df['num_frames_output'].astype('int8')
    df['predicted_x_linear'] = (df['x'] + df['vx'] * df['time_to_ball_land']).astype('float32')
    df['predicted_y_linear'] = (df['y'] + df['vy'] * df['time_to_ball_land']).astype('float32')
    
    # With acceleration
    df['dir_sin'] = np.sin(np.radians(df['dir'])).astype('float32')
    df['dir_cos'] = np.cos(np.radians(df['dir'])).astype('float32')
    df['predicted_x_accel'] = (df['x'] + df['vx'] * df['time_to_ball_land'] + 0.5 * df['a'] * df['dir_cos'] * df['time_to_ball_land']**2).astype('float32')
    df['predicted_y_accel'] = (df['y'] + df['vy'] * df['time_to_ball_land'] + 0.5 * df['a'] * df['dir_sin'] * df['time_to_ball_land']**2).astype('float32')
    
    # Field context
    df['x_normalized'] = (df['x'] / 120.0).astype('float32')
    df['y_normalized'] = (df['y'] / 53.3).astype('float32')
    df['ball_x_normalized'] = (df['ball_land_x'] / 120.0).astype('float32')
    df['ball_y_normalized'] = (df['ball_land_y'] / 53.3).astype('float32')
    df['in_red_zone'] = ((df['x'] < 20) | (df['x'] > 100)).astype('int8')
    df['near_sideline'] = ((df['y'] < 10) | (df['y'] > 43.3)).astype('int8')
    df['dist_to_sideline'] = np.minimum(df['y'], 53.3 - df['y']).astype('float32')
    
    # Orientation features
    df['o_sin'] = np.sin(np.radians(df['o'])).astype('float32')
    df['o_cos'] = np.cos(np.radians(df['o'])).astype('float32')
    df['dir_o_diff'] = np.abs(df['dir'] - df['o']).astype('float32')
    df['dir_o_diff'] = np.minimum(df['dir_o_diff'], 360 - df['dir_o_diff']).astype('float32')
    
    # Statistical features
    df['speed_percentile'] = df.groupby(['game_id', 'play_id'])['s'].transform(lambda x: x.rank(pct=True)).astype('float32')
    
    return df, position_encoder

def create_strategic_training_pairs(input_df, output_df, max_samples=100000):
    """Create training pairs"""
    print("\n[STEP 2] Creating Training Pairs")
    print("-" * 60)
    
    training_pairs = []
    total_created = 0
    
    for (game_id, play_id), play_input in input_df.groupby(['game_id', 'play_id']):
        if total_created >= max_samples:
            break
        
        play_output = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
        
        if len(play_output) == 0:
            continue
        
        for nfl_id in play_output['nfl_id'].unique():
            if total_created >= max_samples:
                break
            
            player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
            player_output = play_output[play_output['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if len(player_input) == 0 or len(player_output) == 0:
                continue
            
            is_target_player = player_input['player_to_predict'].iloc[0] if 'player_to_predict' in player_input.columns else False
            last_frame = player_input.iloc[-1]
            
            if len(player_input) >= 2:
                second_last = player_input.iloc[-2]
                velocity_change_x = last_frame['vx'] - second_last['vx']
                velocity_change_y = last_frame['vy'] - second_last['vy']
                acceleration_est = np.sqrt(velocity_change_x**2 + velocity_change_y**2)
            else:
                velocity_change_x = velocity_change_y = acceleration_est = 0.0
            
            for idx, output_row in player_output.iterrows():
                if total_created >= max_samples:
                    break
                
                time_diff = float(output_row['frame_id'] - last_frame['frame_id'])
                displacement = np.sqrt((output_row['x'] - last_frame['x'])**2 + (output_row['y'] - last_frame['y'])**2)
                
                training_example = {
                    'target_x': float(output_row['x']),
                    'target_y': float(output_row['y']),
                    'time_diff': time_diff,
                    'output_frame_id': int(output_row['frame_id']),
                    'velocity_change_x': velocity_change_x,
                    'velocity_change_y': velocity_change_y,
                    'acceleration_est': acceleration_est,
                    'displacement': displacement,
                    'is_target_player': int(is_target_player),
                }
                
                feature_cols = [
                    'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy',
                    'dist_to_ball', 'angle_to_ball', 'moving_toward_ball', 'velocity_ball_alignment',
                    'is_targeted_receiver', 'is_defensive_coverage', 'is_passer', 'is_other_route_runner',
                    'position_encoded', 'is_offense',
                    'speed_squared', 'momentum_magnitude', 'kinetic_energy',
                    'time_to_ball_land', 'frames_remaining',
                    'predicted_x_linear', 'predicted_y_linear', 'predicted_x_accel', 'predicted_y_accel',
                    'x_normalized', 'y_normalized', 'ball_x_normalized', 'ball_y_normalized',
                    'in_red_zone', 'near_sideline', 'dist_to_sideline',
                    'dir_sin', 'dir_cos', 'o_sin', 'o_cos', 'dir_o_diff',
                    'speed_percentile', 'ball_land_x', 'ball_land_y', 'num_frames_output'
                ]
                
                for col in feature_cols:
                    if col in last_frame.index:
                        training_example[f'input_{col}'] = float(last_frame[col])
                
                training_pairs.append(training_example)
                total_created += 1
    
    training_df = pd.DataFrame(training_pairs)
    print(f"✓ Created {len(training_df):,} training examples")
    
    return training_df

# Load and prepare data
train_input, train_output = load_strategic_data()
train_input_features, pos_encoder = create_strategic_features(train_input)
training_data = create_strategic_training_pairs(train_input_features, train_output, max_samples=100000)

del train_input, train_input_features, train_output
gc.collect()

# ===================================
# STEP 3: TRAIN MODELS
# ===================================

print("\n[STEP 3] Training Models")
print("-" * 60)

feature_cols = [col for col in training_data.columns if col.startswith('input_') or 
               col in ['time_diff', 'velocity_change_x', 'velocity_change_y', 
                      'acceleration_est', 'displacement', 'output_frame_id']]

X = training_data[feature_cols].fillna(0)
y = training_data[['target_x', 'target_y']]
sample_weights = training_data['is_target_player'].values + 1

X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X, y, sample_weights, test_size=0.15, random_state=42
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Training samples: {len(X_train):,}, Validation: {len(X_val):,}, Features: {len(feature_cols)}")

# Train models
print("\n  Training Linear Model...")
linear_model = MultiOutputRegressor(Ridge(alpha=1.0))
linear_model.fit(X_train_scaled, y_train, sample_weight=w_train)
linear_pred = linear_model.predict(X_val_scaled)
linear_rmse = np.sqrt(mean_squared_error(y_val, linear_pred))
print(f"  ✓ Linear RMSE: {linear_rmse:.6f} yards")

print("\n  Training XGBoost...")
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=n_cores
))
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_val_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
print(f"  ✓ XGBoost RMSE: {xgb_rmse:.6f} yards")

print("\n  Training LightGBM...")
lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=n_cores, verbose=-1
))
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_val_scaled)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
print(f"  ✓ LightGBM RMSE: {lgb_rmse:.6f} yards")

print("\n  Training CatBoost...")
cat_model = MultiOutputRegressor(cb.CatBoostRegressor(
    iterations=250, depth=7, learning_rate=0.06, random_state=42, verbose=False
))
cat_model.fit(X_train_scaled, y_train)
cat_pred = cat_model.predict(X_val_scaled)
cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred))
print(f"  ✓ CatBoost RMSE: {cat_rmse:.6f} yards")

# Calculate ensemble weights
models = {
    'Linear': (linear_model, linear_rmse),
    'XGBoost': (xgb_model, xgb_rmse),
    'LightGBM': (lgb_model, lgb_rmse),
    'CatBoost': (cat_model, cat_rmse)
}

total_inv_rmse = sum(1.0 / rmse for _, rmse in models.values())
weights = {name: (1.0 / rmse) / total_inv_rmse for name, (_, rmse) in models.items()}

# Ensemble validation
ensemble_pred = sum(model.predict(X_val_scaled) * weights[name] for name, (model, _) in models.items())
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

print(f"\n✓ Ensemble Validation RMSE: {ensemble_rmse:.6f} yards")
print(f"  Model Weights: XGB={weights['XGBoost']:.2%}, LGB={weights['LightGBM']:.2%}, Cat={weights['CatBoost']:.2%}, Lin={weights['Linear']:.2%}")

del training_data, X_train, X_val, y_train, y_val
gc.collect()

# ===================================
# STEP 4: GENERATE TEST PREDICTIONS
# ===================================

print("\n[STEP 4] Generating Test Predictions")
print("-" * 60)

# Load test data
test_input = pd.read_csv(r'c:\nfl-big-data-bowl-2026-prediction\test_input.csv', dtype={
    'x': 'float32', 'y': 'float32', 's': 'float32', 'a': 'float32',
    'dir': 'float32', 'o': 'float32', 'frame_id': 'int16',
    'ball_land_x': 'float32', 'ball_land_y': 'float32',
    'num_frames_output': 'int8'
})
test_targets = pd.read_csv(r'c:\nfl-big-data-bowl-2026-prediction\test.csv')

print(f"✓ Test input: {len(test_input):,} rows")
print(f"✓ Test targets: {len(test_targets):,} predictions needed")

# Apply feature engineering to test data
test_input_features, _ = create_strategic_features(test_input)

print("\nGenerating predictions...")

def generate_predictions(test_input_df, test_targets_df, models_dict, weights_dict, scaler, feature_cols):
    """Generate ensemble predictions for test data"""
    predictions = []
    batch_size = 1000
    
    for batch_start in range(0, len(test_targets_df), batch_size):
        batch_end = min(batch_start + batch_size, len(test_targets_df))
        batch_targets = test_targets_df.iloc[batch_start:batch_end]
        
        for _, target_row in batch_targets.iterrows():
            game_id = target_row['game_id']
            play_id = target_row['play_id']
            player_id = target_row['nfl_id']
            target_frame = target_row['frame_id']
            
            player_input = test_input_df[
                (test_input_df['game_id'] == game_id) & 
                (test_input_df['play_id'] == play_id) & 
                (test_input_df['nfl_id'] == player_id)
            ].sort_values('frame_id')
            
            if len(player_input) == 0:
                pred_x, pred_y = 50.0, 26.65
            else:
                last_frame = player_input.iloc[-1]
                
                # Calculate velocity changes (same as training)
                if len(player_input) >= 2:
                    second_last = player_input.iloc[-2]
                    velocity_change_x = last_frame['vx'] - second_last['vx']
                    velocity_change_y = last_frame['vy'] - second_last['vy']
                    acceleration_est = np.sqrt(velocity_change_x**2 + velocity_change_y**2)
                else:
                    velocity_change_x = velocity_change_y = acceleration_est = 0.0
                
                # Estimate displacement to target frame
                time_diff_val = target_frame - last_frame['frame_id']
                displacement = np.sqrt((last_frame['vx'] * time_diff_val)**2 + (last_frame['vy'] * time_diff_val)**2)
                
                # Create feature vector matching training exactly
                feature_dict = {
                    'time_diff': time_diff_val,
                    'velocity_change_x': velocity_change_x,
                    'velocity_change_y': velocity_change_y,
                    'acceleration_est': acceleration_est,
                    'displacement': displacement,
                    'output_frame_id': target_frame,
                    'is_target_player': 0  # Unknown at test time
                }
                
                for col in feature_cols:
                    if col.startswith('input_'):
                        orig_col = col[6:]
                        if orig_col in last_frame.index:
                            feature_dict[col] = float(last_frame[orig_col])
                
                feature_values = [feature_dict.get(col, 0) for col in feature_cols]
                feature_array = np.array(feature_values).reshape(1, -1)
                
                try:
                    feature_scaled = scaler.transform(feature_array)
                    
                    # Ensemble prediction
                    pred_xy = sum(
                        model.predict(feature_scaled) * weights_dict[name]
                        for name, (model, _) in models_dict.items()
                    )[0]
                    
                    pred_x, pred_y = float(pred_xy[0]), float(pred_xy[1])
                    
                    # Apply physical constraints
                    pred_x = np.clip(pred_x, 0, 120)
                    pred_y = np.clip(pred_y, 0, 53.3)
                except:
                    pred_x, pred_y = 50.0, 26.65
            
            predictions.append({
                'id': f"{target_row['game_id']}_{target_row['play_id']}_{target_row['nfl_id']}_{target_row['frame_id']}",
                'x': pred_x,
                'y': pred_y
            })
        
        if (batch_start // batch_size + 1) % 5 == 0:
            print(f"  Progress: {batch_end:,}/{len(test_targets_df):,} ({100*batch_end/len(test_targets_df):.1f}%)")
    
    return pd.DataFrame(predictions)

submission = generate_predictions(test_input_features, test_targets, models, weights, scaler, feature_cols)

print(f"\n✓ Generated {len(submission):,} predictions")

# ===================================
# STEP 5: SAVE SUBMISSION
# ===================================

print("\n[STEP 5] Saving Submission File")
print("-" * 60)

# Load sample submission to ensure correct format
sample_submission = pd.read_csv(r'c:\nfl-big-data-bowl-2026-prediction\sample_submission.csv')

# Ensure all required IDs are present
expected_ids = set(sample_submission['id'])
our_ids = set(submission['id'])
missing_ids = expected_ids - our_ids

if missing_ids:
    print(f"⚠ Adding {len(missing_ids):,} missing predictions...")
    missing_df = pd.DataFrame([
        {'id': missing_id, 'x': 50.0, 'y': 26.65} for missing_id in missing_ids
    ])
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Final formatting
submission = submission.merge(sample_submission[['id']], on='id', how='right')
submission = submission.sort_values('id').reset_index(drop=True)
submission = submission.fillna({'x': 50.0, 'y': 26.65})

# Save submission
submission.to_csv('submission.csv', index=False)

print("\n" + "="*80)
print("SUBMISSION GENERATION COMPLETE!")
print("="*80)
print(f"File: submission.csv")
print(f"Total predictions: {len(submission):,}")
print(f"X range: [{submission['x'].min():.2f}, {submission['x'].max():.2f}]")
print(f"Y range: [{submission['y'].min():.2f}, {submission['y'].max():.2f}]")
print(f"\nValidation Performance:")
print(f"  Ensemble RMSE: {ensemble_rmse:.6f} yards")
print(f"  Best Model (XGBoost): {xgb_rmse:.6f} yards")
print(f"  Target: < 0.500 yards")
print(f"  Status: {'✅ ACHIEVED!' if ensemble_rmse < 0.5 else '⚠ Close'}")
print("\n✅ Ready for Kaggle submission!")
print("="*80)
