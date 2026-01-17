"""
NFL Big Data Bowl 2026 - Optimized Gradient Boosting Ensemble
Focus on proven techniques with advanced feature engineering
Target: RMSE < 0.5 yards
"""

import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

np.random.seed(42)

print("=" * 80)
print("NFL BIG DATA BOWL 2026 - OPTIMIZED GRADIENT BOOSTING ENSEMBLE")
print("=" * 80)
print("Advanced Feature Engineering + Multi-Model Ensemble")
print("Target: RMSE < 0.5 yards")
print("=" * 80)


# ============================================================================
# DATA LOADING
# ============================================================================

def detect_data_paths() -> Tuple[Path, Path, Path, Path]:
    """Detect data file locations."""
    base_candidates = [
        Path('/kaggle/input/nfl-big-data-bowl-2026-prediction'),
        Path('/kaggle/input'),
        Path('c:/nfl-big-data-bowl-2026-prediction')
    ]

    train_folder = None
    validation_folder = None
    for base in base_candidates:
        if base.exists():
            candidate = base / 'train'
            val_candidate = base / 'validation'
            if candidate.exists():
                train_folder = candidate
            if val_candidate.exists():
                validation_folder = val_candidate
            if train_folder and validation_folder:
                break
            if any('input_2023' in f for f in os.listdir(base)):
                train_folder = base
            if 'validation' in os.listdir(base):
                validation_folder = base / 'validation'
            if train_folder and validation_folder:
                break

    if train_folder is None:
        raise FileNotFoundError('Unable to locate training data.')
    if validation_folder is None:
        print('Warning: Validation folder not found. Using only training data.')

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
        validation_folder,
        find_file('test_input.csv'),
        find_file('test.csv'),
        find_file('sample_submission.csv')
    )


def load_training_data(train_folder: Path, weeks: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data with memory optimization."""
    print(f"\n[PHASE 1] Loading Training Data ({len(weeks)} weeks)")
    print("-" * 60)
    
    input_frames, output_frames = [], []
    
    for week in weeks:
        input_file = train_folder / f'input_2023_{week}.csv'
        output_file = train_folder / f'output_2023_{week}.csv'
        
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
        raise RuntimeError('No training data loaded.')
    
    train_input = pd.concat(input_frames, ignore_index=True)
    train_output = pd.concat(output_frames, ignore_index=True)
    
    print(f"\n✓ Total: {len(train_input):,} input rows, {len(train_output):,} output rows")
    
    del input_frames, output_frames
    gc.collect()
    
    return train_input, train_output


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive feature engineering:
    - Ball landing integration (PRIMARY)
    - Physics-based features
    - Role-specific modeling
    - Temporal dynamics
    - Spatial context
    """
    print("\n[PHASE 2] Engineering Advanced Features")
    print("-" * 60)
    
    feats = df.copy()
    
    # === VELOCITY DECOMPOSITION ===
    print("  → Velocity vectors...")
    feats['vx'] = feats['s'] * np.cos(np.radians(feats['dir']))
    feats['vy'] = feats['s'] * np.sin(np.radians(feats['dir']))
    
    # === BALL-CENTRIC FEATURES (Critical for sub-0.5 RMSE) ===
    print("  → Ball landing integration...")
    feats['dist_to_ball'] = np.sqrt(
        (feats['x'] - feats['ball_land_x']) ** 2 +
        (feats['y'] - feats['ball_land_y']) ** 2
    )
    
    # Directional features to ball
    dx_to_ball = feats['ball_land_x'] - feats['x']
    dy_to_ball = feats['ball_land_y'] - feats['y']
    feats['angle_to_ball'] = np.degrees(np.arctan2(dy_to_ball, dx_to_ball))
    
    # Velocity alignment with ball direction
    alignment = feats['dir'] - feats['angle_to_ball']
    feats['velocity_ball_alignment'] = np.cos(np.radians(alignment))
    feats['moving_toward_ball'] = (feats['velocity_ball_alignment'] > 0).astype('int8')
    
    # Ball proximity tiers
    feats['very_close_to_ball'] = (feats['dist_to_ball'] < 5).astype('int8')
    feats['close_to_ball'] = (feats['dist_to_ball'] < 15).astype('int8')
    
    # Normalized ball direction components
    ball_dist_safe = feats['dist_to_ball'] + 1e-6
    feats['ball_direction_x'] = dx_to_ball / ball_dist_safe
    feats['ball_direction_y'] = dy_to_ball / ball_dist_safe
    
    # === PHYSICS FEATURES ===
    print("  → Physics-based features...")
    feats['speed_squared'] = feats['s'] ** 2
    feats['kinetic_energy'] = 0.5 * feats['speed_squared']
    feats['momentum'] = feats['s'] * 200.0  # Approximate mass
    
    # Directional encoding
    feats['dir_sin'] = np.sin(np.radians(feats['dir']))
    feats['dir_cos'] = np.cos(np.radians(feats['dir']))
    feats['o_sin'] = np.sin(np.radians(feats['o']))
    feats['o_cos'] = np.cos(np.radians(feats['o']))
    
    # Body alignment
    diff = np.abs(feats['dir'] - feats['o'])
    feats['dir_o_diff'] = np.minimum(diff, 360 - diff)
    feats['body_aligned'] = (feats['dir_o_diff'] < 30).astype('int8')
    
    # === ROLE-SPECIFIC FEATURES ===
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
    feats['is_skill_position'] = feats['player_position'].isin(['WR', 'RB', 'TE', 'QB']).astype('int8')
    feats['is_db'] = feats['player_position'].isin(['CB', 'S', 'SS', 'FS', 'DB']).astype('int8')
    feats['is_linebacker'] = feats['player_position'].isin(['LB', 'ILB', 'OLB', 'MLB']).astype('int8')
    
    # === SPATIAL CONTEXT ===
    print("  → Spatial context...")
    feats['x_norm'] = feats['x'] / 120.0
    feats['y_norm'] = feats['y'] / 53.3
    feats['ball_x_norm'] = feats['ball_land_x'] / 120.0
    feats['ball_y_norm'] = feats['ball_land_y'] / 53.3
    
    # Field boundaries
    feats['dist_to_sideline'] = np.minimum(feats['y'], 53.3 - feats['y'])
    feats['dist_to_endzone'] = np.minimum(feats['x'], 120 - feats['x'])
    feats['near_sideline'] = (feats['dist_to_sideline'] < 10).astype('int8')
    feats['in_red_zone'] = ((feats['x'] < 20) | (feats['x'] > 100)).astype('int8')
    
    # Field zones
    feats['field_third'] = pd.cut(feats['x'], bins=[-np.inf, 40, 80, np.inf], labels=[0, 1, 2]).astype('int8')
    
    # === TEMPORAL FEATURES ===
    print("  → Temporal dynamics...")
    feats['frames_remaining'] = feats['num_frames_output']
    feats['time_to_ball'] = feats['frames_remaining'] / 10.0
    
    # Linear extrapolation (physics-based prediction)
    feats['predicted_x_linear'] = feats['x'] + feats['vx'] * feats['time_to_ball']
    feats['predicted_y_linear'] = feats['y'] + feats['vy'] * feats['time_to_ball']
    
    # With acceleration
    feats['predicted_x_accel'] = (
        feats['predicted_x_linear'] +
        0.5 * feats['a'] * feats['dir_cos'] * feats['time_to_ball'] ** 2
    )
    feats['predicted_y_accel'] = (
        feats['predicted_y_linear'] +
        0.5 * feats['a'] * feats['dir_sin'] * feats['time_to_ball'] ** 2
    )
    
    # Attraction to ball (for receivers)
    feats['ball_attraction_x'] = feats['ball_direction_x'] * feats['is_targeted_receiver']
    feats['ball_attraction_y'] = feats['ball_direction_y'] * feats['is_targeted_receiver']
    
    # === INTERACTION FEATURES ===
    print("  → Player interaction features...")
    feats['speed_percentile'] = feats.groupby(['game_id', 'play_id'])['s'].transform(
        lambda x: x.rank(pct=True)
    )
    feats['accel_intensity'] = np.abs(feats['a'])
    feats['is_accelerating'] = (feats['a'] > 0.5).astype('int8')
    feats['is_decelerating'] = (feats['a'] < -0.5).astype('int8')
    
    # Change capacity (ability to turn)
    feats['change_capacity'] = 1.0 / (1.0 + feats['s'])
    
    # === ADVANCED TRAJECTORY FEATURES ===
    print("  → Advanced trajectory prediction...")
    # Distance from linear prediction to ball
    feats['linear_pred_to_ball_dist'] = np.sqrt(
        (feats['predicted_x_linear'] - feats['ball_land_x']) ** 2 +
        (feats['predicted_y_linear'] - feats['ball_land_y']) ** 2
    )
    
    # Distance from accel prediction to ball
    feats['accel_pred_to_ball_dist'] = np.sqrt(
        (feats['predicted_x_accel'] - feats['ball_land_x']) ** 2 +
        (feats['predicted_y_accel'] - feats['ball_land_y']) ** 2
    )
    
    # Relative positioning features
    feats['x_relative_to_ball'] = feats['x'] - feats['ball_land_x']
    feats['y_relative_to_ball'] = feats['y'] - feats['ball_land_y']
    
    # Speed components relative to ball direction
    feats['speed_toward_ball'] = (
        feats['vx'] * feats['ball_direction_x'] + 
        feats['vy'] * feats['ball_direction_y']
    )
    feats['speed_perpendicular_ball'] = (
        feats['vx'] * feats['ball_direction_y'] - 
        feats['vy'] * feats['ball_direction_x']
    )
    
    # Time-scaled features
    feats['dist_to_ball_per_time'] = feats['dist_to_ball'] / (feats['time_to_ball'] + 0.1)
    feats['speed_ratio_to_required'] = feats['s'] / (feats['dist_to_ball_per_time'] + 0.1)
    
    # Role-specific ball features
    feats['receiver_ball_product'] = feats['is_targeted_receiver'] * feats['dist_to_ball']
    feats['defender_ball_product'] = feats['is_defensive_coverage'] * feats['dist_to_ball']
    
    # Momentum toward ball
    feats['momentum_toward_ball'] = feats['momentum'] * feats['velocity_ball_alignment']
    
    # Field position interactions
    feats['x_ball_interaction'] = feats['x_norm'] * feats['ball_x_norm']
    feats['y_ball_interaction'] = feats['y_norm'] * feats['ball_y_norm']
    
    # Quadrant features (field divided into 4 zones)
    feats['in_left_half'] = (feats['y'] < 26.65).astype('int8')
    feats['in_offensive_half'] = (feats['x'] > 60).astype('int8')
    
    # Trajectory confidence (how aligned is current movement with ball)
    feats['trajectory_confidence'] = (
        feats['velocity_ball_alignment'] * feats['moving_toward_ball'] * 
        (1.0 / (1.0 + feats['dir_o_diff'] / 180.0))
    )
    
    # === POLYNOMIAL & INTERACTION FEATURES ===
    print("  → Polynomial & interaction features...")
    # Speed-distance interactions
    feats['speed_dist_product'] = feats['s'] * feats['dist_to_ball']
    feats['speed_squared_dist'] = feats['speed_squared'] * feats['dist_to_ball']
    
    # Time-space interactions
    feats['time_x_product'] = feats['time_to_ball'] * feats['x_norm']
    feats['time_y_product'] = feats['time_to_ball'] * feats['y_norm']
    feats['time_dist_product'] = feats['time_to_ball'] * feats['dist_to_ball']
    
    # Role-position interactions
    feats['receiver_x_interaction'] = feats['is_targeted_receiver'] * feats['x_norm']
    feats['receiver_y_interaction'] = feats['is_targeted_receiver'] * feats['y_norm']
    feats['defender_x_interaction'] = feats['is_defensive_coverage'] * feats['x_norm']
    feats['defender_y_interaction'] = feats['is_defensive_coverage'] * feats['y_norm']
    
    # Velocity-acceleration synergy
    feats['vel_accel_product'] = feats['s'] * feats['a']
    feats['vx_accel_product'] = feats['vx'] * feats['a']
    feats['vy_accel_product'] = feats['vy'] * feats['a']
    
    # Distance squared features
    feats['dist_to_ball_squared'] = feats['dist_to_ball'] ** 2
    feats['dist_to_sideline_squared'] = feats['dist_to_sideline'] ** 2
    
    # === EXPONENTIAL & LOG FEATURES ===
    print("  → Exponential & log transformations...")
    # Log distance (helps with scale)
    feats['log_dist_to_ball'] = np.log1p(feats['dist_to_ball'])
    feats['log_time_to_ball'] = np.log1p(feats['time_to_ball'])
    
    # Exponential decay features (importance decreases with distance/time)
    feats['exp_neg_dist'] = np.exp(-feats['dist_to_ball'] / 10.0)
    feats['exp_neg_time'] = np.exp(-feats['time_to_ball'])
    
    # === CUMULATIVE & STATISTICAL FEATURES ===
    print("  → Statistical aggregation features...")
    # Distance percentiles within play
    feats['dist_ball_percentile'] = feats.groupby(['game_id', 'play_id'])['dist_to_ball'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Position variance (spread of players)
    feats['x_std_in_play'] = feats.groupby(['game_id', 'play_id'])['x'].transform('std').fillna(0)
    feats['y_std_in_play'] = feats.groupby(['game_id', 'play_id'])['y'].transform('std').fillna(0)
    
    # === DIRECTIONAL INTENSITY FEATURES ===
    print("  → Directional intensity features...")
    # Combined directional strength
    feats['directional_strength'] = np.sqrt(feats['dir_cos']**2 + feats['dir_sin']**2) * feats['s']
    feats['orientation_strength'] = np.sqrt(feats['o_cos']**2 + feats['o_sin']**2) * feats['s']
    
    # Angular momentum approximation
    feats['angular_momentum'] = feats['s'] * feats['dir_o_diff'] / (180.0 + 1e-6)
    
    # === RATIO FEATURES ===
    print("  → Advanced ratio features...")
    feats['vx_vy_ratio'] = feats['vx'] / (np.abs(feats['vy']) + 0.1)
    feats['x_y_ratio'] = feats['x'] / (feats['y'] + 0.1)
    feats['ball_x_y_ratio'] = feats['ball_land_x'] / (feats['ball_land_y'] + 0.1)
    
    # Acceleration to speed ratio (agility indicator)
    feats['accel_speed_ratio'] = feats['a'] / (feats['s'] + 0.1)
    
    # === PLAY-LEVEL CONTEXT FEATURES ===
    print("  → Play-level context features...")
    play_group = feats.groupby(['game_id', 'play_id'])
    feats['play_mean_speed'] = play_group['s'].transform('mean')
    feats['play_max_speed'] = play_group['s'].transform('max')
    feats['play_speed_std'] = play_group['s'].transform('std').fillna(0.0)
    feats['play_mean_dist_to_ball'] = play_group['dist_to_ball'].transform('mean')
    feats['play_min_dist_to_ball'] = play_group['dist_to_ball'].transform('min')
    feats['play_max_dist_to_ball'] = play_group['dist_to_ball'].transform('max')
    feats['play_density'] = play_group['nfl_id'].transform('count')

    side_group = feats.groupby(['game_id', 'play_id', 'player_side'])
    feats['team_mean_x'] = side_group['x'].transform('mean')
    feats['team_mean_y'] = side_group['y'].transform('mean')
    feats['team_speed_mean'] = side_group['s'].transform('mean')
    feats['team_speed_std'] = side_group['s'].transform('std').fillna(0.0)

    feats[['team_mean_x', 'team_mean_y', 'team_speed_mean', 'team_speed_std']] = (
        feats[['team_mean_x', 'team_mean_y', 'team_speed_mean', 'team_speed_std']].fillna(0.0)
    )

    feats['offset_team_center_x'] = feats['x'] - feats['team_mean_x']
    feats['offset_team_center_y'] = feats['y'] - feats['team_mean_y']
    feats['dist_to_team_center'] = np.sqrt(
        feats['offset_team_center_x'] ** 2 + feats['offset_team_center_y'] ** 2
    )
    feats['speed_vs_team_mean'] = feats['s'] - feats['team_speed_mean']
    feats['ball_offset_team_center_x'] = feats['ball_land_x'] - feats['team_mean_x']
    feats['ball_offset_team_center_y'] = feats['ball_land_y'] - feats['team_mean_y']
    
    # === BALL DOMINANCE FEATURES ===
    print("  → Ball dominance & catchability features...")
    # Catchability score (distance, alignment, speed)
    feats['catchability_score'] = (
        (1.0 / (1.0 + feats['dist_to_ball'])) *
        feats['velocity_ball_alignment'] *
        (1.0 - feats['speed_percentile'] * 0.3)  # Moderate speed is better
    )
    
    # Ball dominance (for receivers)
    feats['ball_dominance'] = (
        feats['is_targeted_receiver'] * 
        feats['exp_neg_dist'] * 
        feats['moving_toward_ball']
    )
    
    # Interception potential (for defenders)
    feats['interception_potential'] = (
        feats['is_defensive_coverage'] *
        feats['exp_neg_dist'] *
        feats['moving_toward_ball'] *
        feats['speed_percentile']
    )
    
    print(f"✓ Feature engineering complete: {len(feats.columns)} total features")
    
    return feats


# Feature columns for modeling
FEATURE_COLUMNS = [
    'x', 'y', 's', 'a', 'frame_id',
    'vx', 'vy', 'speed_squared', 'kinetic_energy', 'momentum',
    'dir_sin', 'dir_cos', 'o_sin', 'o_cos', 'dir_o_diff', 'body_aligned',
    'ball_land_x', 'ball_land_y', 'dist_to_ball', 'angle_to_ball',
    'velocity_ball_alignment', 'moving_toward_ball',
    'very_close_to_ball', 'close_to_ball',
    'ball_direction_x', 'ball_direction_y',
    'is_targeted_receiver', 'is_defensive_coverage', 'is_passer', 'is_other_route_runner',
    'is_offense', 'is_defense', 'position_encoded',
    'is_skill_position', 'is_db', 'is_linebacker',
    'x_norm', 'y_norm', 'ball_x_norm', 'ball_y_norm',
    'dist_to_sideline', 'dist_to_endzone', 'near_sideline', 'in_red_zone', 'field_third',
    'frames_remaining', 'time_to_ball',
    'predicted_x_linear', 'predicted_y_linear',
    'predicted_x_accel', 'predicted_y_accel',
    'ball_attraction_x', 'ball_attraction_y',
    'speed_percentile', 'accel_intensity', 'is_accelerating', 'is_decelerating',
    'change_capacity',
    # New advanced features
    'linear_pred_to_ball_dist', 'accel_pred_to_ball_dist',
    'x_relative_to_ball', 'y_relative_to_ball',
    'speed_toward_ball', 'speed_perpendicular_ball',
    'dist_to_ball_per_time', 'speed_ratio_to_required',
    'receiver_ball_product', 'defender_ball_product',
    'momentum_toward_ball',
    'x_ball_interaction', 'y_ball_interaction',
    'in_left_half', 'in_offensive_half',
    'trajectory_confidence',
    # Polynomial & interaction features
    'speed_dist_product', 'speed_squared_dist',
    'time_x_product', 'time_y_product', 'time_dist_product',
    'receiver_x_interaction', 'receiver_y_interaction',
    'defender_x_interaction', 'defender_y_interaction',
    'vel_accel_product', 'vx_accel_product', 'vy_accel_product',
    'dist_to_ball_squared', 'dist_to_sideline_squared',
    # Exponential & log features
    'log_dist_to_ball', 'log_time_to_ball',
    'exp_neg_dist', 'exp_neg_time',
    # Statistical features
    'dist_ball_percentile', 'x_std_in_play', 'y_std_in_play',
    # Directional intensity
    'directional_strength', 'orientation_strength', 'angular_momentum',
    # Ratio features
    'vx_vy_ratio', 'x_y_ratio', 'ball_x_y_ratio', 'accel_speed_ratio',
    # Play-level context
    'play_mean_speed', 'play_max_speed', 'play_speed_std',
    'play_mean_dist_to_ball', 'play_min_dist_to_ball', 'play_max_dist_to_ball',
    'play_density',
    'team_mean_x', 'team_mean_y', 'team_speed_mean', 'team_speed_std',
    'offset_team_center_x', 'offset_team_center_y', 'dist_to_team_center',
    'speed_vs_team_mean', 'ball_offset_team_center_x', 'ball_offset_team_center_y',
    # Ball dominance
    'catchability_score', 'ball_dominance', 'interception_potential'
]


# ============================================================================
# TRAINING PAIR GENERATION
# ============================================================================

def create_training_pairs(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    max_samples: int = 400_000
) -> pd.DataFrame:
    """Create training pairs with multi-frame trajectory encoding."""
    print("\n[PHASE 3] Creating Training Pairs")
    print("-" * 60)
    
    pairs = []
    created = 0
    
    for (game_id, play_id), play_in in input_df.groupby(['game_id', 'play_id']):
        if created >= max_samples:
            break
        
        play_out = output_df[
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id)
        ]
        
        if play_out.empty:
            continue
        
        for nfl_id in play_out['nfl_id'].unique():
            if created >= max_samples:
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
                if created >= max_samples:
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
                
                # Add all base features
                for col in FEATURE_COLUMNS:
                    if col in last_frame.index:
                        sample[f'input_{col}'] = float(last_frame[col])
                    else:
                        sample[f'input_{col}'] = 0.0
                
                pairs.append(sample)
                created += 1
    
    training_df = pd.DataFrame(pairs)
    
    print(f"✓ Created {len(training_df):,} training pairs")
    print(f"  Target player ratio: {100 * training_df['is_target_player'].mean():.1f}%")
    
    return training_df


# ============================================================================
# OPTIMIZED ENSEMBLE TRAINING
# ============================================================================

def train_optimized_ensemble(
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weights,
    meta_feature_indices=None
):
    """Train advanced stacked ensemble with cross-validated meta-learning and residual boosting."""
    print("\n[PHASE 4] Training Advanced Stacked Ensemble")
    print("-" * 60)

    base_model_builders = {
        'XGBoost': lambda: MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=600,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=2.0,
            reg_alpha=0.8,
            gamma=0.15,
            min_child_weight=5,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )),
        'LightGBM': lambda: MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=600,
            max_depth=11,
            learning_rate=0.028,
            subsample=0.92,
            colsample_bytree=0.92,
            reg_lambda=1.8,
            reg_alpha=0.6,
            min_child_samples=25,
            num_leaves=100,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )),
        'CatBoost': lambda: MultiOutputRegressor(cb.CatBoostRegressor(
            iterations=500,
            depth=10,
            learning_rate=0.035,
            l2_leaf_reg=3.0,
            subsample=0.9,
            random_state=42,
            verbose=False
        )),
        'Ridge': lambda: MultiOutputRegressor(Ridge(alpha=2.5)),
        'ExtraTrees': lambda: MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )),
        'GradientBoosting': lambda: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42
        ))
    }
    base_model_names = list(base_model_builders.keys())
    display_names = {
        'XGBoost': 'XGBoost (Enhanced)',
        'LightGBM': 'LightGBM (Enhanced)',
        'CatBoost': 'CatBoost (Enhanced)',
        'Ridge': 'Ridge Regression',
        'ExtraTrees': 'ExtraTrees Ensemble',
        'GradientBoosting': 'Gradient Boosting'
    }

    if meta_feature_indices:
        X_train_meta = X_train[:, meta_feature_indices]
        X_val_meta = X_val[:, meta_feature_indices]
    else:
        X_train_meta = None
        X_val_meta = None

    # Cross-validated out-of-fold predictions for robust meta-learning
    print("  → Generating cross-validated meta-features (3-fold)...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    oof_predictions = {
        name: np.zeros((X_train.shape[0], y_train.shape[1]), dtype=np.float32)
        for name in base_model_names
    }

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        print(f"     Fold {fold_idx}/3")
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        w_tr = sample_weights[tr_idx] if sample_weights is not None else None

        for name, builder in base_model_builders.items():
            model = builder()
            fit_kwargs = {}
            if w_tr is not None:
                fit_kwargs['sample_weight'] = w_tr
            model.fit(X_tr, y_tr, **fit_kwargs)
            oof_predictions[name][va_idx] = model.predict(X_va)

    meta_train_input = np.hstack([oof_predictions[name] for name in base_model_names])
    if X_train_meta is not None and X_train_meta.size:
        meta_train_input = np.hstack([meta_train_input, X_train_meta])

    # Level 1: Train final base learners on full training split
    models = {}
    predictions = {}
    scores = {}

    for name in base_model_names:
        print(f"  → Training {display_names[name]}...")
        model = base_model_builders[name]()
        fit_kwargs = {}
        if sample_weights is not None:
            fit_kwargs['sample_weight'] = sample_weights
        model.fit(X_train, y_train, **fit_kwargs)
        val_pred = model.predict(X_val)
        models[name] = model
        predictions[name] = val_pred
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        scores[name] = rmse
        print(f"     RMSE: {rmse:.4f} yards")

    print("\n  → Calculating weighted ensemble...")
    total_inv_rmse = sum(1.0 / scores[name] for name in base_model_names)
    weights = {name: (1.0 / scores[name]) / total_inv_rmse for name in base_model_names}
    ensemble_pred = sum(predictions[name] * weights[name] for name in base_model_names)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))

    # Level 2: Meta learner with gating features
    print("\n  → Training Stacking Meta-Learner...")
    meta_val_input = np.hstack([predictions[name] for name in base_model_names])
    if X_val_meta is not None and X_val_meta.size:
        meta_val_input = np.hstack([meta_val_input, X_val_meta])

    meta_model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        alpha=1e-4,
        learning_rate_init=0.0015,
        batch_size=4096,
        max_iter=350,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False
    )
    meta_model.fit(meta_train_input, y_train)
    stacked_val_pred = meta_model.predict(meta_val_input)
    if stacked_val_pred.ndim == 1:
        stacked_val_pred = stacked_val_pred.reshape(-1, 1)
    stacked_rmse = np.sqrt(mean_squared_error(y_val, stacked_val_pred))
    print(f"     Stacked RMSE: {stacked_rmse:.4f} yards")

    # Residual correction using gradient boosted residuals
    print("  → Training residual booster...")
    residual_target = y_train - meta_model.predict(meta_train_input)
    if residual_target.ndim == 1:
        residual_target = residual_target.reshape(-1, 1)
    residual_model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.2,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    ))
    residual_fit_kwargs = {}
    if sample_weights is not None:
        residual_fit_kwargs['sample_weight'] = sample_weights
    residual_model.fit(meta_train_input, residual_target, **residual_fit_kwargs)
    residual_val_pred = residual_model.predict(meta_val_input)
    if residual_val_pred.ndim == 1:
        residual_val_pred = residual_val_pred.reshape(-1, 1)
    boosted_val_pred = stacked_val_pred + residual_val_pred
    boosted_rmse = np.sqrt(mean_squared_error(y_val, boosted_val_pred))
    print(f"     Residual-boosted RMSE: {boosted_rmse:.4f} yards")

    print("\n  → Evaluating ensemble strategies...")
    strategies = [
        ('Weighted Average', ensemble_pred, ensemble_rmse, False, False),
        ('Stacked Meta-Learner', stacked_val_pred, stacked_rmse, True, False),
        ('Stacked + Residual Booster', boosted_val_pred, boosted_rmse, True, True)
    ]
    strategies.sort(key=lambda tpl: tpl[2])
    best_name, final_pred, final_rmse, use_stacking, use_residual = strategies[0]
    print(f"     Selected: {best_name} (RMSE: {final_rmse:.4f} yards)")

    models['meta_model'] = meta_model
    if use_residual:
        models['residual_model'] = residual_model

    print("\n" + "=" * 60)
    print(f"FINAL ENSEMBLE VALIDATION RMSE: {final_rmse:.4f} yards")
    print(f"  X-coordinate: {np.sqrt(mean_squared_error(y_val[:, 0], final_pred[:, 0])):.4f}")
    print(f"  Y-coordinate: {np.sqrt(mean_squared_error(y_val[:, 1], final_pred[:, 1])):.4f}")
    print(f"\nEnsemble Type: {best_name}")
    print("\nBase Model Performance:")
    for name in base_model_names:
        print(f"  {name:18s}: RMSE {scores[name]:.4f} yards, Weight {weights[name]:6.2%}")
    print("=" * 60)

    ensemble_context = {
        'base_model_names': base_model_names,
        'meta_feature_indices': meta_feature_indices or [],
        'use_residual': use_residual
    }

    return models, weights, scores, final_rmse, use_stacking, ensemble_context


# ============================================================================
# PREDICTION
# ============================================================================

def predict_single_player(
    player_df,
    target_frame,
    models,
    weights,
    scaler,
    feature_columns,
    ensemble_context=None,
    use_stacking=False
):
    """Generate prediction with physical constraints and stacking support."""
    player_df = player_df.sort_values('frame_id')
    last_frame = player_df.iloc[-1]
    context = ensemble_context or {}
    base_model_names = context.get(
        'base_model_names',
        [name for name in models.keys() if name not in {'meta_model', 'residual_model'}]
    )
    meta_feature_indices = context.get('meta_feature_indices', [])
    use_residual = context.get('use_residual', False)
    
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
    
    # Build feature vector
    feature_dict = {
        'time_diff': float(time_diff),
        'output_frame_id': float(target_frame),
        'velocity_change_x': float(vel_dx),
        'velocity_change_y': float(vel_dy),
        'acceleration_est': float(accel_est),
        'trajectory_curvature': float(trajectory_curvature),
        'is_target_player': 0
    }
    
    for col in FEATURE_COLUMNS:
        key = f'input_{col}'
        feature_dict[key] = float(last_frame[col]) if col in last_frame.index else 0.0
    
    # Order features
    ordered_features = [feature_dict.get(col, 0.0) for col in feature_columns]
    features_scaled = scaler.transform([ordered_features])
    
    # Generate predictions
    if use_stacking and 'meta_model' in models:
        # Stacked ensemble: get predictions from all base models
        base_preds = [models[name].predict(features_scaled)[0] for name in base_model_names]

        meta_features = np.array(base_preds).reshape(1, -1)
        if meta_feature_indices:
            meta_features = np.hstack([meta_features, features_scaled[:, meta_feature_indices]])

        pred = models['meta_model'].predict(meta_features)
        if np.ndim(pred) == 1:
            pred = pred.reshape(1, -1)
        if use_residual and 'residual_model' in models:
            residual_adjustment = models['residual_model'].predict(meta_features)
            pred = pred + residual_adjustment
        pred = pred[0]
    else:
        # Weighted ensemble
        pred = sum(
            models[name].predict(features_scaled) * weights[name]
            for name in base_model_names
        )[0]
    
    x_pred, y_pred = pred[0], pred[1]
    
    # Physical constraints
    x_pred = float(np.clip(x_pred, 0.0, 120.0))
    y_pred = float(np.clip(y_pred, 0.0, 53.3))
    
    return x_pred, y_pred


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Detect paths

    train_folder, validation_folder, test_input_path, test_targets_path, sample_sub_path = detect_data_paths()

    # Load training data (ALL 18 weeks for maximum coverage)
    weeks = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09', 
             'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18']
    train_input, train_output = load_training_data(train_folder, weeks)

    # Engineer features
    train_input = engineer_advanced_features(train_input)

    # Create training pairs (increased for better coverage)
    training_df = create_training_pairs(train_input, train_output, max_samples=400_000)

    del train_input, train_output
    gc.collect()

    # Prepare features
    feature_columns = [col for col in training_df.columns if col.startswith('input_')]
    feature_columns += [
        'time_diff', 'output_frame_id',
        'velocity_change_x', 'velocity_change_y',
        'acceleration_est', 'trajectory_curvature',
        'is_target_player'
    ]

    X_train = training_df[feature_columns].fillna(0.0)
    y_train = training_df[['target_x', 'target_y']].values
    sample_weights = training_df['is_target_player'].values * 2.0 + 1.0

    meta_feature_names = [
        'input_dist_to_ball',
        'input_speed_toward_ball',
        'input_speed_perpendicular_ball',
        'input_speed_ratio_to_required',
        'input_momentum_toward_ball',
        'input_dist_to_team_center',
        'input_speed_vs_team_mean',
        'input_play_mean_speed',
        'input_play_mean_dist_to_ball',
        'time_diff',
        'is_target_player'
    ]
    meta_feature_indices = [
        feature_columns.index(name) for name in meta_feature_names if name in feature_columns
    ]

    # Load validation data if available
    if validation_folder is not None and validation_folder.exists():
        val_weeks = [f.name[-6:-4] for f in validation_folder.glob('input_2023_w*.csv')]
        if not val_weeks:
            val_weeks = ['w01']  # fallback
        val_input, val_output = load_training_data(validation_folder, val_weeks)
        val_input = engineer_advanced_features(val_input)
        validation_df = create_training_pairs(val_input, val_output, max_samples=50_000)
        X_val = validation_df[feature_columns].fillna(0.0)
        y_val = validation_df[['target_x', 'target_y']].values
        w_val = validation_df['is_target_player'].values * 2.0 + 1.0
        print(f"\n[INFO] Validation Configuration")
        print(f"  Features: {len(feature_columns)}")
        print(f"  Samples: {len(X_val):,}")
    else:
        print("\n[INFO] No validation folder found, using train/val split.")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_train, y_train, sample_weights, test_size=0.15, random_state=42
        )

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train ensemble
    models, weights, scores, ensemble_rmse, use_stacking, ensemble_context = train_optimized_ensemble(
        X_train_scaled, y_train, X_val_scaled, y_val, w_val, meta_feature_indices
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
                pred_x, pred_y = 60.0, 26.65
            else:
                pred_x, pred_y = predict_single_player(
                    player_input,
                    target_frame,
                    models,
                    weights,
                    scaler,
                    feature_columns,
                    ensemble_context,
                    use_stacking
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
        missing_df = pd.DataFrame({'id': list(missing_ids), 'x': 60.0, 'y': 26.65})
        submission = pd.concat([submission, missing_df], ignore_index=True)
    
    submission = submission.merge(sample_sub[['id']], on='id', how='right')
    submission = submission.fillna({'x': 60.0, 'y': 26.65})
    submission = submission.sort_values('id').reset_index(drop=True)
    
    # Save
    submission.to_csv('submission.csv', index=False)
    
    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 80)
    print(f"File: submission.csv")
    print(f"Predictions: {len(submission):,}")
    print(f"X range: [{submission['x'].min():.2f}, {submission['x'].max():.2f}]")
    print(f"Y range: [{submission['y'].min():.2f}, {submission['y'].max():.2f}]")
    print(f"\nValidation RMSE: {ensemble_rmse:.4f} yards")
    print(f"Target: < 0.500 yards")
    print(f"Status: {'✅ ACHIEVED!' if ensemble_rmse < 0.5 else '⚠ Close - Further tuning recommended'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
