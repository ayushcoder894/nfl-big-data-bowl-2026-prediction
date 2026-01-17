# NFL Big Data Bowl 2026 - Memory-Optimized Ensemble Learning
# This code handles large datasets with memory-efficient processing

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
import multiprocessing
import gc
from typing import List, Tuple

# Memory optimization settings
pd.set_option('mode.chained_assignment', None)
os.environ['PYTHONHASHSEED'] = '42'

# Set number of threads for optimal CPU performance
n_cores = min(multiprocessing.cpu_count(), 8)  # Limit to 8 cores to save memory
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)

print(f"Memory-Optimized Ensemble Learning - Using {n_cores} CPU cores")

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import MiniBatchKMeans  # Memory-efficient clustering

# Set random seed for reproducibility
np.random.seed(42)

print("Memory-optimized libraries imported successfully!")

# ===================================
# 1. MEMORY-EFFICIENT DATA LOADING
# ===================================

def load_data_efficiently():
    """Load data with memory optimization"""
    
    train_folder = Path(r'c:\nfl-big-data-bowl-2026-prediction\train')
    
    if not train_folder.exists():
        raise FileNotFoundError(f"Training folder not found: {train_folder}")
    
    print(f"Loading training data efficiently from {train_folder}...")
    
    # Load only a subset of weeks to manage memory
    weeks_to_use = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09']  # First 9 weeks
    input_dfs = []
    output_dfs = []
    
    for week in weeks_to_use:
        input_file = train_folder / f"input_2023_{week}.csv"
        output_file = train_folder / f"output_2023_{week}.csv"
        
        if input_file.exists() and output_file.exists():
            try:
                # Load with optimized dtypes
                input_df = pd.read_csv(input_file, dtype={
                    'x': 'float32', 'y': 'float32', 's': 'float32', 'a': 'float32',
                    'dir': 'float32', 'o': 'float32', 'frame_id': 'int16',
                    'ball_land_x': 'float32', 'ball_land_y': 'float32',
                    'num_frames_output': 'int8'
                })
                output_df = pd.read_csv(output_file, dtype={
                    'x': 'float32', 'y': 'float32', 'frame_id': 'int16'
                })
                
                input_dfs.append(input_df)
                output_dfs.append(output_df)
                print(f"Loaded {week}: {len(input_df)} input rows, {len(output_df)} output rows")
                
            except Exception as e:
                print(f"Error loading {week}: {e}")
        else:
            print(f"Files not found for {week}")
    
    if len(input_dfs) == 0:
        raise ValueError("No training data could be loaded.")
    
    # Combine and optimize memory
    train_input = pd.concat(input_dfs, ignore_index=True)
    train_output = pd.concat(output_dfs, ignore_index=True)
    
    # Clear intermediate data
    del input_dfs, output_dfs
    gc.collect()
    
    print(f"Total training data: Input={len(train_input)}, Output={len(train_output)}")
    return train_input, train_output

# Load data
train_input, train_output = load_data_efficiently()

# ===================================
# 2. MEMORY-EFFICIENT FEATURE ENGINEERING
# ===================================

def create_efficient_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Create essential features with memory optimization"""
    
    print("Creating memory-efficient features...")
    
    # Basic velocity components (using float32)
    df['vx'] = (df['s'] * np.cos(np.radians(df['dir']))).astype('float32')
    df['vy'] = (df['s'] * np.sin(np.radians(df['dir']))).astype('float32')
    
    # Essential physics features
    df['speed_squared'] = (df['s'] ** 2).astype('float32')
    df['velocity_magnitude'] = np.sqrt(df['vx']**2 + df['vy']**2).astype('float32')
    
    # Direction features (key ones only)
    df['dir_sin'] = np.sin(np.radians(df['dir'])).astype('float32')
    df['dir_cos'] = np.cos(np.radians(df['dir'])).astype('float32')
    df['o_sin'] = np.sin(np.radians(df['o'])).astype('float32')
    df['o_cos'] = np.cos(np.radians(df['o'])).astype('float32')
    
    # Ball interaction features
    df['dist_to_ball'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + 
                                (df['y'] - df['ball_land_y'])**2).astype('float32')
    df['dir_to_ball'] = np.degrees(np.arctan2(df['ball_land_y'] - df['y'], 
                                             df['ball_land_x'] - df['x'])).astype('float32')
    
    # Normalized positions
    df['x_normalized'] = (df['x'] / 120.0).astype('float32')
    df['y_normalized'] = (df['y'] / 53.3).astype('float32')
    
    # Time-based predictions
    df['time_to_ball_land'] = (df['num_frames_output'] / 10.0).astype('float32')
    df['predicted_x_linear'] = (df['x'] + df['vx'] * df['time_to_ball_land']).astype('float32')
    df['predicted_y_linear'] = (df['y'] + df['vy'] * df['time_to_ball_land']).astype('float32')
    
    # Player side encoding
    df['is_offense'] = (df['player_side'] == 'Offense').astype('int8')
    df['is_defense'] = (df['player_side'] == 'Defense').astype('int8')
    
    # Position encoding
    position_encoder = LabelEncoder()
    df['position_encoded'] = position_encoder.fit_transform(df['player_position']).astype('int8')
    
    role_encoder = LabelEncoder()
    df['role_encoded'] = role_encoder.fit_transform(df['player_role']).astype('int8')
    
    # Essential position groups
    df['is_skill_position'] = df['player_position'].isin(['WR', 'RB', 'TE', 'QB']).astype('int8')
    df['is_lineman'] = df['player_position'].isin(['C', 'G', 'T', 'DE', 'DT', 'NT']).astype('int8')
    
    # Statistical percentiles (memory efficient)
    df['speed_percentile'] = df.groupby(['game_id', 'play_id'])['s'].transform(lambda x: x.rank(pct=True)).astype('float32')
    
    # Simple clustering (memory efficient)
    try:
        position_features = df[['x_normalized', 'y_normalized', 'vx', 'vy']].fillna(0)
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=10000)
        df['position_cluster'] = kmeans.fit_predict(position_features).astype('int8')
    except:
        df['position_cluster'] = 0
    
    print(f"Created features. Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df, position_encoder, role_encoder

# Apply feature engineering
train_input_features, pos_encoder, role_encoder = create_efficient_features(train_input)

# Clear original data to save memory
del train_input
gc.collect()

# ===================================
# 3. MEMORY-EFFICIENT TRAINING PAIRS
# ===================================

def create_training_pairs_efficient(input_df: pd.DataFrame, output_df: pd.DataFrame, sample_size: int = 100000):
    """Create training pairs with memory optimization and sampling"""
    
    print(f"Creating training pairs (sampling {sample_size} examples)...")
    
    training_pairs = []
    total_examples = 0
    
    for (game_id, play_id), play_input in input_df.groupby(['game_id', 'play_id']):
        if total_examples >= sample_size:
            break
            
        play_output = output_df[(output_df['game_id'] == game_id) & 
                               (output_df['play_id'] == play_id)]
        
        if len(play_output) == 0:
            continue
        
        for nfl_id in play_output['nfl_id'].unique():
            if total_examples >= sample_size:
                break
                
            player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
            player_output = play_output[play_output['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if len(player_input) == 0 or len(player_output) == 0:
                continue
            
            # Use only the last frame for efficiency
            last_frame = player_input.iloc[-1]
            
            # Create training examples (limit per player)
            for _, output_row in player_output.head(3).iterrows():  # Max 3 examples per player
                if total_examples >= sample_size:
                    break
                
                time_diff = output_row['frame_id'] - last_frame['frame_id']
                
                training_example = {
                    'time_diff': time_diff,
                    'target_x': output_row['x'],
                    'target_y': output_row['y'],
                }
                
                # Add essential input features
                essential_features = [
                    'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy', 'speed_squared',
                    'velocity_magnitude', 'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
                    'dist_to_ball', 'dir_to_ball', 'x_normalized', 'y_normalized',
                    'time_to_ball_land', 'predicted_x_linear', 'predicted_y_linear',
                    'is_offense', 'is_defense', 'position_encoded', 'role_encoded',
                    'is_skill_position', 'is_lineman', 'speed_percentile', 'position_cluster',
                    'ball_land_x', 'ball_land_y', 'frame_id', 'num_frames_output'
                ]
                
                for feature in essential_features:
                    if feature in last_frame.index:
                        training_example[f'input_{feature}'] = last_frame[feature]
                
                training_pairs.append(training_example)
                total_examples += 1
    
    training_df = pd.DataFrame(training_pairs)
    
    # Convert to efficient dtypes
    for col in training_df.columns:
        if training_df[col].dtype == 'float64':
            training_df[col] = training_df[col].astype('float32')
        elif training_df[col].dtype == 'int64':
            training_df[col] = training_df[col].astype('int32')
    
    print(f"Created {len(training_df)} efficient training examples")
    return training_df

# Create training dataset
training_data = create_training_pairs_efficient(train_input_features, train_output, sample_size=150000)

# Clear more memory
del train_input_features, train_output
gc.collect()

# ===================================
# 4. MEMORY-EFFICIENT MODEL TRAINING
# ===================================

print("Preparing features for memory-efficient training...")

# Select features
feature_cols = [col for col in training_data.columns if col.startswith('input_') or col == 'time_diff']
X = training_data[feature_cols].fillna(0)
y = training_data[['target_x', 'target_y']]

print(f"Training with {len(feature_cols)} features and {len(X)} samples")

# Memory-efficient feature selection
print("Performing memory-efficient feature selection...")
selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))  # Limit features
X_selected = selector.fit_transform(X, y.iloc[:, 0])
selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

X = pd.DataFrame(X_selected, columns=selected_features)
print(f"Selected {len(selected_features)} features")

# Simple preprocessing
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training memory-efficient ensemble models...")

# ===================================
# MEMORY-EFFICIENT ENSEMBLE MODELS
# ===================================

models = {}

# Model 1: Lightweight XGBoost
print("Training lightweight XGBoost...")
models['XGBoost'] = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=200,  # Reduced
    max_depth=6,       # Reduced
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=n_cores,
    tree_method='hist'  # Memory efficient
))
models['XGBoost'].fit(X_train, y_train)

# Model 2: Lightweight LightGBM  
print("Training lightweight LightGBM...")
models['LightGBM'] = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=200,  # Reduced
    max_depth=6,       # Reduced
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=n_cores,
    verbose=-1
))
models['LightGBM'].fit(X_train, y_train)

# Model 3: Lightweight CatBoost
print("Training lightweight CatBoost...")
models['CatBoost'] = MultiOutputRegressor(cb.CatBoostRegressor(
    iterations=150,    # Reduced
    depth=6,           # Reduced
    learning_rate=0.1,
    random_state=42,
    verbose=False,
    thread_count=n_cores
))
models['CatBoost'].fit(X_train, y_train)

# Model 4: Random Forest
print("Training Random Forest...")
models['RandomForest'] = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=100,  # Reduced
    max_depth=10,      # Reduced
    random_state=42,
    n_jobs=n_cores
))
models['RandomForest'].fit(X_train, y_train)

# Model 5: Extra Trees
print("Training Extra Trees...")
models['ExtraTrees'] = MultiOutputRegressor(ExtraTreesRegressor(
    n_estimators=100,  # Reduced
    max_depth=10,      # Reduced
    random_state=42,
    n_jobs=n_cores
))
models['ExtraTrees'].fit(X_train, y_train)

# Model 6: Linear models (memory efficient)
print("Training linear models...")
models['Ridge'] = MultiOutputRegressor(Ridge(alpha=1.0))
models['Ridge'].fit(X_train, y_train)

models['ElasticNet'] = MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
models['ElasticNet'].fit(X_train, y_train)

models['BayesianRidge'] = MultiOutputRegressor(BayesianRidge())
models['BayesianRidge'].fit(X_train, y_train)

# Evaluate models
print("\nEvaluating models...")
model_scores = {}
for name, model in models.items():
    try:
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        model_scores[name] = rmse
        print(f"{name}: RMSE = {rmse:.6f} yards")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# ===================================
# SIMPLE ENSEMBLE
# ===================================

print("\nCreating simple weighted ensemble...")

# Weight models by inverse RMSE
weights = {}
total_weight = 0
for name, rmse in model_scores.items():
    weight = 1.0 / rmse
    weights[name] = weight
    total_weight += weight

# Normalize weights
for name in weights:
    weights[name] /= total_weight

print("Model weights:")
for name, weight in weights.items():
    print(f"{name}: {weight:.4f}")

def ensemble_predict(X_test):
    """Simple weighted ensemble prediction"""
    predictions = []
    total_weight = 0
    
    for name, model in models.items():
        if name in weights:
            try:
                pred = model.predict(X_test)
                predictions.append(pred * weights[name])
                total_weight += weights[name]
            except:
                continue
    
    if predictions and total_weight > 0:
        return np.sum(predictions, axis=0) / total_weight
    else:
        return np.zeros((len(X_test), 2))

# Evaluate ensemble
ensemble_pred = ensemble_predict(X_val)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

print(f"\n🎯 MEMORY-EFFICIENT ENSEMBLE RESULTS:")
print(f"Ensemble RMSE: {ensemble_rmse:.6f} yards")
print(f"X-coordinate RMSE: {np.sqrt(mean_squared_error(y_val.iloc[:, 0], ensemble_pred[:, 0])):.6f} yards")
print(f"Y-coordinate RMSE: {np.sqrt(mean_squared_error(y_val.iloc[:, 1], ensemble_pred[:, 1])):.6f} yards")

# Store ensemble info
ensemble_info = {
    'models': models,
    'weights': weights,
    'scaler': scaler,
    'selector': selector,
    'selected_features': selected_features,
    'feature_cols': feature_cols
}

# ===================================
# 5. GENERATE PREDICTIONS ON TEST DATA
# ===================================

print("\nLoading test data...")

test_input_path = r'c:\nfl-big-data-bowl-2026-prediction\test_input.csv'
test_path = r'c:\nfl-big-data-bowl-2026-prediction\test.csv'
sample_submission_path = r'c:\nfl-big-data-bowl-2026-prediction\sample_submission.csv'

if not os.path.exists(test_input_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Test files not found.")

# Load test data with optimized dtypes
test_input = pd.read_csv(test_input_path, dtype={
    'x': 'float32', 'y': 'float32', 's': 'float32', 'a': 'float32',
    'dir': 'float32', 'o': 'float32', 'frame_id': 'int16',
    'ball_land_x': 'float32', 'ball_land_y': 'float32',
    'num_frames_output': 'int8'
})
test_targets = pd.read_csv(test_path, dtype={
    'frame_id': 'int16'
})

print(f"Test input: {len(test_input)} rows")
print(f"Test targets: {len(test_targets)} rows")

# Apply feature engineering to test data
test_input_features, _, _ = create_efficient_features(test_input)

def generate_memory_efficient_predictions(test_input_df, test_targets_df, ensemble_predict_func, ensemble_info):
    """Generate predictions with memory efficiency"""
    
    print("Generating memory-efficient predictions...")
    predictions = []
    
    # Process in smaller batches
    batch_size = 1000
    total_targets = len(test_targets_df)
    
    for batch_start in range(0, total_targets, batch_size):
        batch_end = min(batch_start + batch_size, total_targets)
        batch_targets = test_targets_df.iloc[batch_start:batch_end]
        
        for _, target_row in batch_targets.iterrows():
            game_id = target_row['game_id']
            play_id = target_row['play_id']
            player_id = target_row['nfl_id']
            target_frame = target_row['frame_id']
            
            # Get player input data
            player_input = test_input_df[
                (test_input_df['game_id'] == game_id) & 
                (test_input_df['play_id'] == play_id) & 
                (test_input_df['nfl_id'] == player_id)
            ].sort_values('frame_id')
            
            if len(player_input) == 0:
                pred_x, pred_y = 50.0, 26.65
            else:
                # Use last frame
                last_frame = player_input.iloc[-1]
                
                # Create feature vector
                feature_dict = {'time_diff': target_frame - last_frame['frame_id']}
                
                # Add essential features
                essential_features = [
                    'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy', 'speed_squared',
                    'velocity_magnitude', 'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
                    'dist_to_ball', 'dir_to_ball', 'x_normalized', 'y_normalized',
                    'time_to_ball_land', 'predicted_x_linear', 'predicted_y_linear',
                    'is_offense', 'is_defense', 'position_encoded', 'role_encoded',
                    'is_skill_position', 'is_lineman', 'speed_percentile', 'position_cluster',
                    'ball_land_x', 'ball_land_y', 'frame_id', 'num_frames_output'
                ]
                
                for feature in essential_features:
                    if feature in last_frame.index:
                        feature_dict[f'input_{feature}'] = last_frame[feature]
                
                # Create feature vector for prediction
                feature_values = []
                for col in ensemble_info['feature_cols']:
                    feature_values.append(feature_dict.get(col, 0))
                
                try:
                    # Transform features
                    feature_array = np.array(feature_values).reshape(1, -1)
                    feature_selected = ensemble_info['selector'].transform(feature_array)
                    feature_scaled = ensemble_info['scaler'].transform(feature_selected)
                    
                    # Predict
                    pred_xy = ensemble_predict_func(feature_scaled)[0]
                    pred_x, pred_y = float(pred_xy[0]), float(pred_xy[1])
                    
                    # Apply constraints
                    pred_x = np.clip(pred_x, 0, 120)
                    pred_y = np.clip(pred_y, 0, 53.3)
                    
                except Exception as e:
                    pred_x, pred_y = 50.0, 26.65
            
            predictions.append({
                'id': f"{target_row['game_id']}_{target_row['play_id']}_{target_row['nfl_id']}_{target_row['frame_id']}",
                'x': pred_x,
                'y': pred_y
            })
        
        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"Processed {batch_end}/{total_targets} predictions...")
    
    return pd.DataFrame(predictions)

# Generate predictions
submission = generate_memory_efficient_predictions(test_input_features, test_targets, ensemble_predict, ensemble_info)

print(f"Generated {len(submission)} predictions")

# ===================================
# 6. CREATE SUBMISSION FILE
# ===================================

if not os.path.exists(sample_submission_path):
    raise FileNotFoundError("sample_submission.csv not found")

sample_submission = pd.read_csv(sample_submission_path)

print("Validating submission...")
expected_ids = set(sample_submission['id'])
our_ids = set(submission['id'])

missing_ids = expected_ids - our_ids
if missing_ids:
    print(f"Adding {len(missing_ids)} missing predictions...")
    missing_df = pd.DataFrame([
        {'id': missing_id, 'x': 50.0, 'y': 26.65} for missing_id in missing_ids
    ])
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Ensure correct order and clean up
submission = submission.merge(sample_submission[['id']], on='id', how='right')
submission = submission.sort_values('id').reset_index(drop=True)
submission = submission.fillna({'x': 50.0, 'y': 26.65})

# Save submission
submission.to_csv('submission_memory_efficient.csv', index=False)

print(f"\n🏆 MEMORY-EFFICIENT SUBMISSION SUMMARY:")
print(f"Total predictions: {len(submission)}")
print(f"X coordinate range: {submission['x'].min():.1f} to {submission['x'].max():.1f}")
print(f"Y coordinate range: {submission['y'].min():.1f} to {submission['y'].max():.1f}")
print(f"Average X: {submission['x'].mean():.1f}")
print(f"Average Y: {submission['y'].mean():.1f}")

print("\n" + "="*70)
print("🚀 MEMORY-EFFICIENT ENSEMBLE SUBMISSION CREATED SUCCESSFULLY! 🚀")
print("="*70)
print("File saved as: submission_memory_efficient.csv")
print("✅ Optimized for large datasets")
print("✅ Reduced memory usage") 
print("✅ 8-model ensemble")
print("✅ Efficient feature selection")
print("Ready for submission!")