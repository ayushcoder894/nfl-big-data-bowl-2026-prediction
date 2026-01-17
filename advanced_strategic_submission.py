"""
NFL Big Data Bowl 2026 - Advanced Strategic Framework
Comprehensive multi-agent spatiotemporal modeling approach
Target: RMSE < 0.5 yards
"""

import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

np.random.seed(42)

print("=" * 80)
print("NFL BIG DATA BOWL 2026 - ADVANCED STRATEGIC FRAMEWORK")
print("=" * 80)
print("Multi-Agent Spatiotemporal Modeling with Ensemble Strategy")
print("Target: RMSE < 0.5 yards")
print("=" * 80)


# ============================================================================
# 1. DATA UNDERSTANDING & PREPROCESSING
# ============================================================================

def detect_data_paths() -> Tuple[Path, Path, Path, Path]:
    """Detect data file locations across different environments."""
    base_candidates = [
        Path('/kaggle/input/nfl-big-data-bowl-2026-prediction'),
        Path('/kaggle/input'),
        Path('c:/nfl-big-data-bowl-2026-prediction')
    ]

    train_folder = None
    for base in base_candidates:
        if base.exists():
            candidate = base / 'train'
            if candidate.exists():
                train_folder = candidate
                break
            if any('input_2023' in f for f in os.listdir(base)):
                train_folder = base
                break

    if train_folder is None:
        raise FileNotFoundError('Unable to locate training data.')

    def find_file(name: str) -> Path:
        for base in base_candidates:
            if base.exists():
                candidate = base / name
                if candidate.exists():
                    return candidate
                for root, _, files in os.walk(base):
                    if name in files:
                        return Path(root) / name
        raise FileNotFoundError(f'Unable to locate {name}.')

    return (
        train_folder,
        find_file('test_input.csv'),
        find_file('test.csv'),
        find_file('sample_submission.csv')
    )


def load_training_data(train_folder: Path, weeks: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate training data with memory optimization."""
    print(f"\n[PHASE 1] Loading Training Data ({len(weeks)} weeks)")
    print("-" * 60)
    
    input_frames, output_frames = [], []
    
    for week in weeks:
        input_file = train_folder / f'input_2023_{week}.csv'
        output_file = train_folder / f'output_2023_{week}.csv'
        
        if input_file.exists() and output_file.exists():
            # Optimized dtypes for memory efficiency
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
        else:
            print(f"  ✗ {week}: Files not found")
    
    if not input_frames:
        raise RuntimeError('No training data loaded.')
    
    train_input = pd.concat(input_frames, ignore_index=True)
    train_output = pd.concat(output_frames, ignore_index=True)
    
    print(f"\n✓ Total: {len(train_input):,} input rows, {len(train_output):,} output rows")
    
    del input_frames, output_frames
    gc.collect()
    
    return train_input, train_output


# ============================================================================
# 2. CONTEXTUAL FEATURE ENGINEERING (Strategic Innovation)
# ============================================================================

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering incorporating:
    - Ball landing integration (PRIMARY SIGNAL)
    - Role-specific modeling
    - Multi-scale temporal features
    - Spatial priors and constraints
    - Player interaction dynamics
    """
    print("\n[PHASE 2] Engineering Advanced Features")
    print("-" * 60)
    
    feats = df.copy()
    
    # === A. CORE MOTION FEATURES ===
    print("  → Core motion vectors...")
    feats['vx'] = feats['s'] * np.cos(np.radians(feats['dir']))
    feats['vy'] = feats['s'] * np.sin(np.radians(feats['dir']))
    feats['speed_squared'] = feats['s'] ** 2
    feats['kinetic_energy'] = 0.5 * feats['speed_squared']
    
    # Directional encoding
    feats['dir_sin'] = np.sin(np.radians(feats['dir']))
    feats['dir_cos'] = np.cos(np.radians(feats['dir']))
    feats['o_sin'] = np.sin(np.radians(feats['o']))
    feats['o_cos'] = np.cos(np.radians(feats['o']))
    
    # Body orientation vs direction
    diff = np.abs(feats['dir'] - feats['o'])
    feats['dir_o_diff'] = np.minimum(diff, 360 - diff)
    feats['body_aligned'] = (feats['dir_o_diff'] < 30).astype('int8')
    
    # === B. BALL LANDING INTEGRATION (Primary Signal) ===
    print("  → Ball landing integration...")
    feats['dist_to_ball'] = np.sqrt(
        (feats['x'] - feats['ball_land_x']) ** 2 +
        (feats['y'] - feats['ball_land_y']) ** 2
    )
    feats['angle_to_ball'] = np.degrees(np.arctan2(
        feats['ball_land_y'] - feats['y'],
        feats['ball_land_x'] - feats['x']
    ))
    
    # Velocity alignment with ball direction
    alignment = feats['dir'] - feats['angle_to_ball']
    feats['velocity_ball_alignment'] = np.cos(np.radians(alignment))
    
    # Moving toward ball indicator
    feats['moving_toward_ball'] = (
        (feats['vx'] * (feats['ball_land_x'] - feats['x']) +
         feats['vy'] * (feats['ball_land_y'] - feats['y'])) > 0
    ).astype('int8')
    
    # Ball proximity tiers (critical vs distant)
    feats['ball_proximity_tier'] = pd.cut(
        feats['dist_to_ball'],
        bins=[0, 3, 10, 25, np.inf],
        labels=[3, 2, 1, 0]  # 3=very close, 0=far
    ).astype('int8')
    
    # === C. SPATIAL PRIORS & FIELD CONSTRAINTS ===
    print("  → Spatial priors and field constraints...")
    feats['x_norm'] = feats['x'] / 120.0
    feats['y_norm'] = feats['y'] / 53.3
    feats['ball_x_norm'] = feats['ball_land_x'] / 120.0
    feats['ball_y_norm'] = feats['ball_land_y'] / 53.3
    
    # Distance to boundaries
    feats['dist_to_sideline'] = np.minimum(feats['y'], 53.3 - feats['y'])
    feats['dist_to_endzone'] = np.minimum(feats['x'], 120 - feats['x'])
    
    # Field zones
    feats['field_zone'] = pd.cut(
        feats['x'],
        bins=[-np.inf, 40, 80, np.inf],
        labels=[0, 1, 2]
    ).astype('int8')
    
    feats['in_red_zone'] = ((feats['x'] < 20) | (feats['x'] > 100)).astype('int8')
    feats['near_sideline'] = ((feats['y'] < 10) | (feats['y'] > 43.3)).astype('int8')
    
    # === D. ROLE-SPECIFIC FEATURES ===
    print("  → Role-specific modeling...")
    feats['is_targeted_receiver'] = (feats['player_role'] == 'Targeted Receiver').astype('int8')
    feats['is_defensive_coverage'] = (feats['player_role'] == 'Defensive Coverage').astype('int8')
    feats['is_passer'] = (feats['player_role'] == 'Passer').astype('int8')
    feats['is_other_route_runner'] = (feats['player_role'] == 'Other Route Runner').astype('int8')
    
    # Side encoding
    feats['is_offense'] = (feats['player_side'] == 'Offense').astype('int8')
    feats['is_defense'] = (feats['player_side'] == 'Defense').astype('int8')
    
    # Position encoding
    feats['position_encoded'] = feats['player_position'].astype('category').cat.codes.astype('int16')
    
    # Position groups
    feats['is_skill_position'] = feats['player_position'].isin(
        ['WR', 'RB', 'TE', 'QB']
    ).astype('int8')
    feats['is_db'] = feats['player_position'].isin(
        ['CB', 'S', 'SS', 'FS', 'DB']
    ).astype('int8')
    feats['is_linebacker'] = feats['player_position'].isin(
        ['LB', 'ILB', 'OLB', 'MLB']
    ).astype('int8')
    
    # === E. MULTI-SCALE TEMPORAL FEATURES ===
    print("  → Multi-scale temporal dynamics...")
    feats['frames_remaining'] = feats['num_frames_output']
    feats['time_to_ball_land'] = feats['frames_remaining'] / 10.0
    
    # Short-term: Linear extrapolation (physics-based)
    feats['predicted_x_linear'] = feats['x'] + feats['vx'] * feats['time_to_ball_land']
    feats['predicted_y_linear'] = feats['y'] + feats['vy'] * feats['time_to_ball_land']
    
    # Medium-term: With acceleration
    feats['predicted_x_accel'] = (
        feats['predicted_x_linear'] +
        0.5 * feats['a'] * feats['dir_cos'] * feats['time_to_ball_land'] ** 2
    )
    feats['predicted_y_accel'] = (
        feats['predicted_y_linear'] +
        0.5 * feats['a'] * feats['dir_sin'] * feats['time_to_ball_land'] ** 2
    )
    
    # Long-term: Attraction to ball (intention-based)
    ball_direction_x = feats['ball_land_x'] - feats['x']
    ball_direction_y = feats['ball_land_y'] - feats['y']
    ball_dist = np.sqrt(ball_direction_x ** 2 + ball_direction_y ** 2) + 1e-6
    
    feats['ball_attraction_x'] = ball_direction_x / ball_dist
    feats['ball_attraction_y'] = ball_direction_y / ball_dist
    
    # === F. PLAYER INTERACTION DYNAMICS ===
    print("  → Player interaction features...")
    # Relative speed within play context
    feats['speed_percentile'] = feats.groupby(['game_id', 'play_id'])['s'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Acceleration intensity
    feats['accel_intensity'] = np.abs(feats['a'])
    feats['is_accelerating'] = (feats['a'] > 0.5).astype('int8')
    feats['is_decelerating'] = (feats['a'] < -0.5).astype('int8')
    
    # === G. MOMENTUM & PHYSICS ===
    print("  → Physics-based features...")
    feats['momentum_x'] = feats['vx'] * 200  # Approximate mass
    feats['momentum_y'] = feats['vy'] * 200
    feats['momentum_magnitude'] = np.sqrt(feats['momentum_x'] ** 2 + feats['momentum_y'] ** 2)
    
    # Change capacity (ability to change direction)
    feats['change_capacity'] = 1.0 / (1.0 + feats['s'])
    
    print(f"✓ Feature engineering complete: {len(feats.columns)} total features")
    
    return feats


# Feature column definitions
CORE_FEATURES = [
    'x', 'y', 's', 'a', 'dir', 'o', 'frame_id',
    'vx', 'vy', 'speed_squared', 'kinetic_energy',
    'dir_sin', 'dir_cos', 'o_sin', 'o_cos', 'dir_o_diff', 'body_aligned'
]

BALL_FEATURES = [
    'ball_land_x', 'ball_land_y', 'dist_to_ball', 'angle_to_ball',
    'velocity_ball_alignment', 'moving_toward_ball', 'ball_proximity_tier',
    'ball_attraction_x', 'ball_attraction_y'
]

SPATIAL_FEATURES = [
    'x_norm', 'y_norm', 'ball_x_norm', 'ball_y_norm',
    'dist_to_sideline', 'dist_to_endzone', 'field_zone',
    'in_red_zone', 'near_sideline'
]

ROLE_FEATURES = [
    'is_targeted_receiver', 'is_defensive_coverage', 'is_passer', 'is_other_route_runner',
    'is_offense', 'is_defense', 'position_encoded',
    'is_skill_position', 'is_db', 'is_linebacker'
]

TEMPORAL_FEATURES = [
    'frames_remaining', 'time_to_ball_land', 'num_frames_output',
    'predicted_x_linear', 'predicted_y_linear',
    'predicted_x_accel', 'predicted_y_accel'
]

INTERACTION_FEATURES = [
    'speed_percentile', 'accel_intensity', 'is_accelerating', 'is_decelerating',
    'momentum_x', 'momentum_y', 'momentum_magnitude', 'change_capacity'
]

ALL_BASE_FEATURES = (
    CORE_FEATURES + BALL_FEATURES + SPATIAL_FEATURES +
    ROLE_FEATURES + TEMPORAL_FEATURES + INTERACTION_FEATURES
)


# ============================================================================
# 3. TRAINING PAIR GENERATION WITH TEMPORAL DYNAMICS
# ============================================================================

def create_advanced_training_pairs(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    max_samples: int = 200_000,
    prioritize_targets: bool = True
) -> pd.DataFrame:
    """
    Create training pairs with:
    - Temporal sequence awareness
    - Player-to-predict prioritization
    - Multi-frame trajectory encoding
    """
    print("\n[PHASE 3] Creating Advanced Training Pairs")
    print("-" * 60)
    
    pairs = []
    created = 0
    target_count = 0
    
    for (game_id, play_id), play_in in input_df.groupby(['game_id', 'play_id']):
        if created >= max_samples:
            break
        
        play_out = output_df[
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id)
        ]
        
        if play_out.empty:
            continue
        
        # Prioritize player_to_predict if enabled
        if prioritize_targets and 'player_to_predict' in play_in.columns:
            priority_players = play_in[play_in['player_to_predict']]['nfl_id'].unique()
            other_players = play_in[~play_in['player_to_predict']]['nfl_id'].unique()
            player_order = list(priority_players) + list(other_players)
        else:
            player_order = play_out['nfl_id'].unique()
        
        for nfl_id in player_order:
            if created >= max_samples:
                break
            
            player_in = play_in[play_in['nfl_id'] == nfl_id].sort_values('frame_id')
            player_out = play_out[play_out['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if player_in.empty or player_out.empty:
                continue
            
            # Track if this is a target player
            is_target = bool(player_in['player_to_predict'].iloc[0]) if 'player_to_predict' in player_in.columns else False
            if is_target:
                target_count += 1
            
            # Use last input frame as reference
            last_frame = player_in.iloc[-1]
            
            # Calculate velocity changes (temporal dynamics)
            if len(player_in) >= 2:
                prev_frame = player_in.iloc[-2]
                vel_dx = last_frame['vx'] - prev_frame['vx']
                vel_dy = last_frame['vy'] - prev_frame['vy']
                
                # Multi-frame trajectory if available
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
            
            # Create training examples for each output frame
            for _, out_row in player_out.iterrows():
                if created >= max_samples:
                    break
                
                time_diff = out_row['frame_id'] - last_frame['frame_id']
                actual_displacement = np.hypot(
                    out_row['x'] - last_frame['x'],
                    out_row['y'] - last_frame['y']
                )
                
                # Expected displacement based on current velocity
                expected_displacement = last_frame['s'] * time_diff / 10.0
                displacement_ratio = (
                    actual_displacement / expected_displacement
                    if expected_displacement > 0.1 else 1.0
                )
                
                sample = {
                    'game_id': game_id,
                    'play_id': play_id,
                    'nfl_id': nfl_id,
                    'target_x': float(out_row['x']),
                    'target_y': float(out_row['y']),
                    'output_frame_id': int(out_row['frame_id']),
                    'time_diff': float(time_diff),
                    'velocity_change_x': float(vel_dx),
                    'velocity_change_y': float(vel_dy),
                    'acceleration_est': float(accel_est),
                    'actual_displacement': float(actual_displacement),
                    'displacement_ratio': float(displacement_ratio),
                    'trajectory_curvature': float(trajectory_curvature),
                    'is_target_player': int(is_target)
                }
                
                # Add all base features
                for col in ALL_BASE_FEATURES:
                    if col in last_frame.index:
                        sample[f'input_{col}'] = float(last_frame[col])
                    else:
                        sample[f'input_{col}'] = 0.0
                
                pairs.append(sample)
                created += 1
    
    training_df = pd.DataFrame(pairs)
    
    print(f"✓ Created {len(training_df):,} training pairs")
    print(f"  → Target players: {target_count:,}")
    print(f"  → Target pair ratio: {100 * training_df['is_target_player'].mean():.1f}%")
    
    return training_df


# ============================================================================
# 4. ENSEMBLE STRATEGY - DIVERSE MODEL TYPES
# ============================================================================

def train_ensemble_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Train diverse ensemble with role-specific emphasis:
    1. Linear Baseline - Simple motion extrapolation
    2. XGBoost - Captures complex interactions
    3. LightGBM - Fast gradient boosting
    4. CatBoost - Handles categoricals well
    """
    print("\n[PHASE 4] Training Ensemble Models")
    print("-" * 60)
    
    models = {}
    predictions = {}
    scores = {}
    
    # 1. LINEAR BASELINE (Physics-based extrapolation)
    print("  → Training Linear Ridge Baseline...")
    linear = MultiOutputRegressor(Ridge(alpha=2.0))
    linear.fit(X_train, y_train, sample_weight=sample_weights)
    linear_pred = linear.predict(X_val)
    linear_rmse = np.sqrt(mean_squared_error(y_val, linear_pred))
    
    models['Linear'] = linear
    predictions['Linear'] = linear_pred
    scores['Linear'] = linear_rmse
    print(f"     RMSE: {linear_rmse:.4f} yards")
    
    # 2. XGBOOST (Complex Interactions)
    print("  → Training XGBoost...")
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    ))
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    scores['XGBoost'] = xgb_rmse
    print(f"     RMSE: {xgb_rmse:.4f} yards")
    
    # 3. LIGHTGBM (Fast Gradient Boosting)
    print("  → Training LightGBM...")
    lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=350,
        max_depth=9,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
    
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = lgb_pred
    scores['LightGBM'] = lgb_rmse
    print(f"     RMSE: {lgb_rmse:.4f} yards")
    
    # 4. CATBOOST (Categorical Handling)
    print("  → Training CatBoost...")
    cat_model = MultiOutputRegressor(cb.CatBoostRegressor(
        iterations=280,
        depth=8,
        learning_rate=0.055,
        l2_leaf_reg=2.0,
        random_state=42,
        verbose=False
    ))
    cat_model.fit(X_train, y_train)
    cat_pred = cat_model.predict(X_val)
    cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred))
    
    models['CatBoost'] = cat_model
    predictions['CatBoost'] = cat_pred
    scores['CatBoost'] = cat_rmse
    print(f"     RMSE: {cat_rmse:.4f} yards")
    
    # Calculate ensemble weights (inverse RMSE)
    total_inv_rmse = sum(1.0 / rmse for rmse in scores.values())
    weights = {
        name: (1.0 / rmse) / total_inv_rmse
        for name, rmse in scores.items()
    }
    
    # Ensemble prediction
    ensemble_pred = sum(
        predictions[name] * weights[name]
        for name in models.keys()
    )
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    
    print("\n" + "=" * 60)
    print(f"ENSEMBLE VALIDATION RMSE: {ensemble_rmse:.4f} yards")
    print(f"  X-coordinate: {np.sqrt(mean_squared_error(y_val[:, 0], ensemble_pred[:, 0])):.4f}")
    print(f"  Y-coordinate: {np.sqrt(mean_squared_error(y_val[:, 1], ensemble_pred[:, 1])):.4f}")
    print("\nModel Weights:")
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}: {weight:6.2%} (RMSE: {scores[name]:.4f})")
    print("=" * 60)
    
    return {
        'models': models,
        'weights': weights,
        'scores': scores,
        'ensemble_rmse': ensemble_rmse
    }


# ============================================================================
# 5. PREDICTION WITH PHYSICAL CONSTRAINTS
# ============================================================================

def predict_with_constraints(
    player_df: pd.DataFrame,
    target_frame: int,
    models: Dict,
    weights: Dict,
    scaler: RobustScaler,
    feature_columns: List[str]
) -> Tuple[float, float]:
    """
    Generate prediction with physical constraints:
    - Field boundaries
    - Maximum acceleration limits
    - Smooth trajectory enforcement
    """
    player_df = player_df.sort_values('frame_id')
    last_frame = player_df.iloc[-1]
    
    # Temporal dynamics
    if len(player_df) >= 2:
        prev_frame = player_df.iloc[-2]
        vel_dx = last_frame['vx'] - prev_frame['vx']
        vel_dy = last_frame['vy'] - prev_frame['vy']
        
        if len(player_df) >= 3:
            frame_minus_2 = player_df.iloc[-3]
            trajectory_curvature = np.hypot(
                last_frame['x'] - 2 * prev_frame['x'] + frame_minus_2['x'],
                last_frame['y'] - 2 * prev_frame['y'] + frame_minus_2['y']
            )
        else:
            trajectory_curvature = 0.0
    else:
        vel_dx = vel_dy = trajectory_curvature = 0.0
    
    accel_est = np.hypot(vel_dx, vel_dy)
    time_diff = target_frame - last_frame['frame_id']
    
    # Expected displacement
    expected_disp = last_frame['s'] * time_diff / 10.0
    
    # Build feature vector
    feature_dict = {
        'time_diff': float(time_diff),
        'output_frame_id': float(target_frame),
        'velocity_change_x': float(vel_dx),
        'velocity_change_y': float(vel_dy),
        'acceleration_est': float(accel_est),
        'actual_displacement': float(expected_disp),  # Placeholder
        'displacement_ratio': 1.0,
        'trajectory_curvature': float(trajectory_curvature),
        'is_target_player': 0  # Unknown at test time
    }
    
    # Add input features
    for col in ALL_BASE_FEATURES:
        key = f'input_{col}'
        feature_dict[key] = float(last_frame[col]) if col in last_frame.index else 0.0
    
    # Order features correctly
    ordered_features = [feature_dict.get(col, 0.0) for col in feature_columns]
    features_scaled = scaler.transform([ordered_features])
    
    # Ensemble prediction
    pred = sum(
        models[name].predict(features_scaled) * weights[name]
        for name in models.keys()
    )
    
    x_pred, y_pred = pred[0]
    
    # Apply physical constraints
    # 1. Field boundaries
    x_pred = float(np.clip(x_pred, 0.0, 120.0))
    y_pred = float(np.clip(y_pred, 0.0, 53.3))
    
    # 2. Maximum acceleration constraint (roughly 10 yards/s²)
    max_displacement = last_frame['s'] * time_diff / 10.0 + 0.5 * 10.0 * (time_diff / 10.0) ** 2
    actual_pred_disp = np.hypot(x_pred - last_frame['x'], y_pred - last_frame['y'])
    
    if actual_pred_disp > max_displacement * 1.5:  # Allow some flexibility
        # Scale back to reasonable range
        scale = (max_displacement * 1.5) / actual_pred_disp
        x_pred = last_frame['x'] + (x_pred - last_frame['x']) * scale
        y_pred = last_frame['y'] + (y_pred - last_frame['y']) * scale
        
        # Re-clip
        x_pred = float(np.clip(x_pred, 0.0, 120.0))
        y_pred = float(np.clip(y_pred, 0.0, 53.3))
    
    return x_pred, y_pred


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    # Detect paths
    train_folder, test_input_path, test_targets_path, sample_sub_path = detect_data_paths()
    
    # Load training data (10 weeks for comprehensive coverage)
    weeks = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09', 'w10']
    train_input, train_output = load_training_data(train_folder, weeks)
    
    # Engineer features
    train_input = engineer_advanced_features(train_input)
    
    # Create training pairs
    training_df = create_advanced_training_pairs(
        train_input, train_output,
        max_samples=200_000,
        prioritize_targets=True
    )
    
    del train_input, train_output
    gc.collect()
    
    # Prepare features
    feature_columns = [col for col in training_df.columns if col.startswith('input_')]
    feature_columns += [
        'time_diff', 'output_frame_id',
        'velocity_change_x', 'velocity_change_y',
        'acceleration_est', 'actual_displacement',
        'displacement_ratio', 'trajectory_curvature',
        'is_target_player'
    ]
    
    X = training_df[feature_columns].fillna(0.0)
    y = training_df[['target_x', 'target_y']].values
    
    # Sample weights (emphasize target players)
    sample_weights = training_df['is_target_player'].values * 2.0 + 1.0
    
    print(f"\n[INFO] Training Configuration")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Samples: {len(X):,}")
    print(f"  Target player weight: 3x")
    
    # Train/val split
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, sample_weights, test_size=0.15, random_state=42
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train ensemble
    ensemble_results = train_ensemble_models(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        sample_weights=w_train
    )
    
    del training_df, X_train, X_val, y_train, y_val
    gc.collect()
    
    # Load test data
    print("\n[PHASE 5] Generating Test Predictions")
    print("-" * 60)
    
    test_input = pd.read_csv(test_input_path)
    test_targets = pd.read_csv(test_targets_path)
    
    print(f"  Test input: {len(test_input):,} rows")
    print(f"  Test targets: {len(test_targets):,} predictions")
    
    # Engineer test features
    test_input = engineer_advanced_features(test_input)
    
    # Generate predictions
    predictions = []
    
    for idx, (game_id, play_id) in enumerate(test_targets.groupby(['game_id', 'play_id']).groups.keys()):
        play_targets = test_targets[
            (test_targets['game_id'] == game_id) &
            (test_targets['play_id'] == play_id)
        ]
        play_input = test_input[
            (test_input['game_id'] == game_id) &
            (test_input['play_id'] == play_id)
        ]
        
        for _, target_row in play_targets.iterrows():
            nfl_id = target_row['nfl_id']
            target_frame = int(target_row['frame_id'])
            
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            
            if player_input.empty:
                pred_x, pred_y = 50.0, 26.65
            else:
                pred_x, pred_y = predict_with_constraints(
                    player_input, target_frame,
                    ensemble_results['models'],
                    ensemble_results['weights'],
                    scaler,
                    feature_columns
                )
            
            pred_id = f"{game_id}_{play_id}_{nfl_id}_{target_frame}"
            predictions.append({'id': pred_id, 'x': pred_x, 'y': pred_y})
        
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1} plays processed")
    
    submission = pd.DataFrame(predictions)
    
    # Align with sample submission
    sample_sub = pd.read_csv(sample_sub_path)
    expected_ids = set(sample_sub['id'])
    produced_ids = set(submission['id'])
    
    missing_ids = expected_ids - produced_ids
    if missing_ids:
        print(f"  → Filling {len(missing_ids):,} missing IDs")
        missing_df = pd.DataFrame({
            'id': list(missing_ids),
            'x': 50.0,
            'y': 26.65
        })
        submission = pd.concat([submission, missing_df], ignore_index=True)
    
    submission = submission.merge(sample_sub[['id']], on='id', how='right')
    submission = submission.fillna({'x': 50.0, 'y': 26.65})
    submission = submission.sort_values('id').reset_index(drop=True)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 80)
    print(f"File: submission.csv")
    print(f"Predictions: {len(submission):,}")
    print(f"X range: [{submission['x'].min():.2f}, {submission['x'].max():.2f}]")
    print(f"Y range: [{submission['y'].min():.2f}, {submission['y'].max():.2f}]")
    print(f"\nValidation RMSE: {ensemble_results['ensemble_rmse']:.4f} yards")
    print(f"Target: < 0.500 yards")
    print(f"Status: {'✅ ACHIEVED' if ensemble_results['ensemble_rmse'] < 0.5 else '⚠ CLOSE - Continue iteration'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
