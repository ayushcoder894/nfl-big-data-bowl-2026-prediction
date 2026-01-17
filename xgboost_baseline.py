"""
XGBoost Baseline with Advanced Techniques
Evaluates both training and validation RMSE
"""

import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb

np.random.seed(42)

print("=" * 80)
print("XGBOOST BASELINE - TRAINING & VALIDATION RMSE")
print("=" * 80)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(folder: Path, weeks: List[str], phase_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data with memory optimization."""
    print(f"\n[{phase_name}] Loading Data ({len(weeks)} weeks)")
    print("-" * 60)
    
    input_frames, output_frames = [], []
    
    for week in weeks:
        input_file = folder / f'input_2023_{week}.csv'
        output_file = folder / f'output_2023_{week}.csv'
        
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
            
            input_frames.append(input_df)
            output_frames.append(output_df)
            print(f"  ✓ {week}: {len(input_df):,} input, {len(output_df):,} output rows")
    
    if not input_frames:
        raise RuntimeError(f'No data loaded for {phase_name}.')
    
    data_input = pd.concat(input_frames, ignore_index=True)
    data_output = pd.concat(output_frames, ignore_index=True)
    
    print(f"✓ Total: {len(data_input):,} input rows, {len(data_output):,} output rows")
    
    del input_frames, output_frames
    gc.collect()
    
    return data_input, data_output


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering for XGBoost."""
    print("\n[FEATURE ENGINEERING]")
    print("-" * 60)
    
    feats = df.copy()
    
    # Velocity decomposition
    print("  → Velocity vectors...")
    feats['vx'] = feats['s'] * np.cos(np.radians(feats['dir']))
    feats['vy'] = feats['s'] * np.sin(np.radians(feats['dir']))
    
    # Ball-centric features
    print("  → Ball landing features...")
    feats['dist_to_ball'] = np.sqrt(
        (feats['x'] - feats['ball_land_x']) ** 2 +
        (feats['y'] - feats['ball_land_y']) ** 2
    )
    
    dx_to_ball = feats['ball_land_x'] - feats['x']
    dy_to_ball = feats['ball_land_y'] - feats['y']
    feats['angle_to_ball'] = np.degrees(np.arctan2(dy_to_ball, dx_to_ball))
    
    alignment = feats['dir'] - feats['angle_to_ball']
    feats['velocity_ball_alignment'] = np.cos(np.radians(alignment))
    feats['moving_toward_ball'] = (feats['velocity_ball_alignment'] > 0).astype('int8')
    
    ball_dist_safe = feats['dist_to_ball'] + 1e-6
    feats['ball_direction_x'] = dx_to_ball / ball_dist_safe
    feats['ball_direction_y'] = dy_to_ball / ball_dist_safe
    
    # Physics features
    print("  → Physics features...")
    feats['speed_squared'] = feats['s'] ** 2
    feats['kinetic_energy'] = 0.5 * feats['speed_squared']
    feats['momentum'] = feats['s'] * 200.0
    
    feats['dir_sin'] = np.sin(np.radians(feats['dir']))
    feats['dir_cos'] = np.cos(np.radians(feats['dir']))
    feats['o_sin'] = np.sin(np.radians(feats['o']))
    feats['o_cos'] = np.cos(np.radians(feats['o']))
    
    diff = np.abs(feats['dir'] - feats['o'])
    feats['dir_o_diff'] = np.minimum(diff, 360 - diff)
    feats['body_aligned'] = (feats['dir_o_diff'] < 30).astype('int8')
    
    # Role-specific features
    print("  → Role-specific features...")
    feats['is_targeted_receiver'] = (feats['player_role'] == 'Targeted Receiver').astype('int8')
    feats['is_defensive_coverage'] = (feats['player_role'] == 'Defensive Coverage').astype('int8')
    feats['is_passer'] = (feats['player_role'] == 'Passer').astype('int8')
    feats['is_offense'] = (feats['player_side'] == 'Offense').astype('int8')
    feats['position_encoded'] = feats['player_position'].astype('category').cat.codes.astype('int16')
    
    # Spatial context
    print("  → Spatial features...")
    feats['x_norm'] = feats['x'] / 120.0
    feats['y_norm'] = feats['y'] / 53.3
    feats['ball_x_norm'] = feats['ball_land_x'] / 120.0
    feats['ball_y_norm'] = feats['ball_land_y'] / 53.3
    feats['dist_to_sideline'] = np.minimum(feats['y'], 53.3 - feats['y'])
    
    # Temporal features
    print("  → Temporal features...")
    feats['frames_remaining'] = feats['num_frames_output']
    feats['time_to_ball'] = feats['frames_remaining'] / 10.0
    
    feats['predicted_x_linear'] = feats['x'] + feats['vx'] * feats['time_to_ball']
    feats['predicted_y_linear'] = feats['y'] + feats['vy'] * feats['time_to_ball']
    
    feats['predicted_x_accel'] = (
        feats['predicted_x_linear'] +
        0.5 * feats['a'] * feats['dir_cos'] * feats['time_to_ball'] ** 2
    )
    feats['predicted_y_accel'] = (
        feats['predicted_y_linear'] +
        0.5 * feats['a'] * feats['dir_sin'] * feats['time_to_ball'] ** 2
    )
    
    # Advanced trajectory features
    print("  → Advanced trajectory features...")
    feats['speed_toward_ball'] = (
        feats['vx'] * feats['ball_direction_x'] + 
        feats['vy'] * feats['ball_direction_y']
    )
    feats['speed_perpendicular_ball'] = (
        feats['vx'] * feats['ball_direction_y'] - 
        feats['vy'] * feats['ball_direction_x']
    )
    
    feats['dist_to_ball_per_time'] = feats['dist_to_ball'] / (feats['time_to_ball'] + 0.1)
    feats['speed_ratio_to_required'] = feats['s'] / (feats['dist_to_ball_per_time'] + 0.1)
    
    # Interaction features
    print("  → Interaction features...")
    feats['speed_percentile'] = feats.groupby(['game_id', 'play_id'])['s'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Polynomial features
    print("  → Polynomial features...")
    feats['speed_dist_product'] = feats['s'] * feats['dist_to_ball']
    feats['time_dist_product'] = feats['time_to_ball'] * feats['dist_to_ball']
    feats['dist_to_ball_squared'] = feats['dist_to_ball'] ** 2
    
    # Log features
    print("  → Log features...")
    feats['log_dist_to_ball'] = np.log1p(feats['dist_to_ball'])
    feats['log_time_to_ball'] = np.log1p(feats['time_to_ball'])
    
    # Exponential decay features
    feats['exp_neg_dist'] = np.exp(-feats['dist_to_ball'] / 10.0)
    
    # Play-level context
    print("  → Play-level context...")
    play_group = feats.groupby(['game_id', 'play_id'])
    feats['play_mean_speed'] = play_group['s'].transform('mean')
    feats['play_mean_dist_to_ball'] = play_group['dist_to_ball'].transform('mean')
    
    side_group = feats.groupby(['game_id', 'play_id', 'player_side'])
    feats['team_mean_x'] = side_group['x'].transform('mean')
    feats['team_mean_y'] = side_group['y'].transform('mean')
    feats['team_speed_mean'] = side_group['s'].transform('mean')
    feats[['team_mean_x', 'team_mean_y', 'team_speed_mean']] = (
        feats[['team_mean_x', 'team_mean_y', 'team_speed_mean']].fillna(0.0)
    )
    
    feats['offset_team_center_x'] = feats['x'] - feats['team_mean_x']
    feats['offset_team_center_y'] = feats['y'] - feats['team_mean_y']
    feats['dist_to_team_center'] = np.sqrt(
        feats['offset_team_center_x'] ** 2 + feats['offset_team_center_y'] ** 2
    )
    
    # Ball dominance
    print("  → Ball dominance features...")
    feats['catchability_score'] = (
        (1.0 / (1.0 + feats['dist_to_ball'])) *
        feats['velocity_ball_alignment'] *
        (1.0 - feats['speed_percentile'] * 0.3)
    )
    
    feats['ball_dominance'] = (
        feats['is_targeted_receiver'] * 
        feats['exp_neg_dist'] * 
        feats['moving_toward_ball']
    )
    
    print(f"✓ Feature engineering complete: {len(feats.columns)} total features")
    
    return feats


# Feature columns
FEATURE_COLUMNS = [
    'x', 'y', 's', 'a', 'frame_id',
    'vx', 'vy', 'speed_squared', 'kinetic_energy', 'momentum',
    'dir_sin', 'dir_cos', 'o_sin', 'o_cos', 'dir_o_diff', 'body_aligned',
    'ball_land_x', 'ball_land_y', 'dist_to_ball', 'angle_to_ball',
    'velocity_ball_alignment', 'moving_toward_ball',
    'ball_direction_x', 'ball_direction_y',
    'is_targeted_receiver', 'is_defensive_coverage', 'is_passer',
    'is_offense', 'position_encoded',
    'x_norm', 'y_norm', 'ball_x_norm', 'ball_y_norm',
    'dist_to_sideline',
    'frames_remaining', 'time_to_ball',
    'predicted_x_linear', 'predicted_y_linear',
    'predicted_x_accel', 'predicted_y_accel',
    'speed_toward_ball', 'speed_perpendicular_ball',
    'dist_to_ball_per_time', 'speed_ratio_to_required',
    'speed_percentile',
    'speed_dist_product', 'time_dist_product', 'dist_to_ball_squared',
    'log_dist_to_ball', 'log_time_to_ball',
    'exp_neg_dist',
    'play_mean_speed', 'play_mean_dist_to_ball',
    'team_mean_x', 'team_mean_y', 'team_speed_mean',
    'offset_team_center_x', 'offset_team_center_y', 'dist_to_team_center',
    'catchability_score', 'ball_dominance'
]


# ============================================================================
# TRAINING PAIR GENERATION
# ============================================================================

def create_pairs(input_df: pd.DataFrame, output_df: pd.DataFrame, max_samples: int = None) -> pd.DataFrame:
    """Create training pairs."""
    print("\n[CREATING PAIRS]")
    print("-" * 60)
    
    pairs = []
    created = 0
    
    for (game_id, play_id), play_in in input_df.groupby(['game_id', 'play_id']):
        if max_samples and created >= max_samples:
            break
        
        play_out = output_df[
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id)
        ]
        
        if play_out.empty:
            continue
        
        for nfl_id in play_out['nfl_id'].unique():
            if max_samples and created >= max_samples:
                break
            
            player_in = play_in[play_in['nfl_id'] == nfl_id].sort_values('frame_id')
            player_out = play_out[play_out['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if player_in.empty or player_out.empty:
                continue
            
            last_frame = player_in.iloc[-1]
            is_target = bool(last_frame['player_to_predict']) if 'player_to_predict' in last_frame.index else False
            
            # Multi-frame trajectory
            if len(player_in) >= 2:
                prev_frame = player_in.iloc[-2]
                vel_dx = last_frame['vx'] - prev_frame['vx']
                vel_dy = last_frame['vy'] - prev_frame['vy']
                
                if len(player_in) >= 3:
                    frame_minus_2 = player_in.iloc[-3]
                    trajectory_curvature = np.hypot(
                        last_frame['x'] - 2 * prev_frame['x'] + frame_minus_2['x'],
                        last_frame['y'] - 2 * prev_frame['y'] + frame_minus_2['y']
                    )
                else:
                    trajectory_curvature = 0.0
            else:
                vel_dx = vel_dy = trajectory_curvature = 0.0
            
            accel_est = np.hypot(vel_dx, vel_dy)
            
            for _, out_row in player_out.iterrows():
                if max_samples and created >= max_samples:
                    break
                
                time_diff = out_row['frame_id'] - last_frame['frame_id']
                
                sample = {
                    'target_x': float(out_row['x']),
                    'target_y': float(out_row['y']),
                    'output_frame_id': int(out_row['frame_id']),
                    'time_diff': float(time_diff),
                    'velocity_change_x': float(vel_dx),
                    'velocity_change_y': float(vel_dy),
                    'acceleration_est': float(accel_est),
                    'trajectory_curvature': float(trajectory_curvature),
                    'is_target_player': int(is_target)
                }
                
                for col in FEATURE_COLUMNS:
                    if col in last_frame.index:
                        sample[f'input_{col}'] = float(last_frame[col])
                    else:
                        sample[f'input_{col}'] = 0.0
                
                pairs.append(sample)
                created += 1
    
    df = pd.DataFrame(pairs)
    print(f"✓ Created {len(df):,} pairs")
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    base_path = Path('c:/nfl-big-data-bowl-2026-prediction')
    train_folder = base_path / 'train'
    validation_folder = base_path / 'validation'
    
    # Load training data (weeks 1-15)
    train_weeks = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 
                   'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15']
    train_input, train_output = load_data(train_folder, train_weeks, "TRAINING")
    
    # Load validation data (weeks 16, 17, 18)
    val_weeks = ['w16', 'w17', 'w18']
    val_input, val_output = load_data(validation_folder, val_weeks, "VALIDATION")
    
    # Engineer features
    train_input = engineer_features(train_input)
    val_input = engineer_features(val_input)
    
    # Create pairs
    train_df = create_pairs(train_input, train_output, max_samples=300_000)
    val_df = create_pairs(val_input, val_output, max_samples=50_000)
    
    del train_input, train_output, val_input, val_output
    gc.collect()
    
    # Prepare features
    feature_columns = [col for col in train_df.columns if col.startswith('input_')]
    feature_columns += [
        'time_diff', 'output_frame_id',
        'velocity_change_x', 'velocity_change_y',
        'acceleration_est', 'trajectory_curvature',
        'is_target_player'
    ]
    
    X_train = train_df[feature_columns].fillna(0.0).values
    y_train = train_df[['target_x', 'target_y']].values
    train_weights = train_df['is_target_player'].values * 2.0 + 1.0
    
    X_val = val_df[feature_columns].fillna(0.0).values
    y_val = val_df[['target_x', 'target_y']].values
    
    print(f"\n[TRAINING CONFIGURATION]")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    
    del train_df, val_df
    gc.collect()
    
    # Scale features
    print("\n[SCALING FEATURES]")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train XGBoost with advanced settings
    print("\n[TRAINING XGBOOST]")
    print("-" * 60)
    
    model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=800,
        max_depth=12,
        learning_rate=0.025,
        subsample=0.88,
        colsample_bytree=0.88,
        colsample_bylevel=0.88,
        colsample_bynode=0.88,
        reg_lambda=2.5,
        reg_alpha=1.0,
        gamma=0.2,
        min_child_weight=5,
        max_delta_step=1,
        random_state=42,
        tree_method='hist',
        n_jobs=-1,
        verbosity=1
    ))
    
    print("  → Fitting model...")
    model.fit(X_train_scaled, y_train, sample_weight=train_weights)
    
    # Training predictions
    print("\n[EVALUATING PERFORMANCE]")
    print("-" * 60)
    
    train_pred = model.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_rmse_x = np.sqrt(mean_squared_error(y_train[:, 0], train_pred[:, 0]))
    train_rmse_y = np.sqrt(mean_squared_error(y_train[:, 1], train_pred[:, 1]))
    
    print(f"\n📊 TRAINING RMSE:")
    print(f"  Overall: {train_rmse:.4f} yards")
    print(f"  X-coord: {train_rmse_x:.4f} yards")
    print(f"  Y-coord: {train_rmse_y:.4f} yards")
    
    # Validation predictions
    val_pred = model.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], val_pred[:, 0]))
    val_rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], val_pred[:, 1]))
    
    print(f"\n📊 VALIDATION RMSE:")
    print(f"  Overall: {val_rmse:.4f} yards")
    print(f"  X-coord: {val_rmse_x:.4f} yards")
    print(f"  Y-coord: {val_rmse_y:.4f} yards")
    
    # Overfitting check
    overfit_margin = val_rmse - train_rmse
    print(f"\n📈 OVERFITTING ANALYSIS:")
    print(f"  Difference: {overfit_margin:.4f} yards")
    if overfit_margin < 0.1:
        print(f"  Status: ✅ Excellent generalization")
    elif overfit_margin < 0.3:
        print(f"  Status: ✅ Good generalization")
    elif overfit_margin < 0.5:
        print(f"  Status: ⚠️ Moderate overfitting")
    else:
        print(f"  Status: ❌ Significant overfitting")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
