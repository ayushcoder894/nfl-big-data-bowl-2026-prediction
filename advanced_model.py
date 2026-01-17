"""
Advanced NFL Trajectory Prediction with State-of-the-Art Techniques
Target: Validation RMSE < 0.5 yards

Advanced Techniques:
1. Deep feature engineering (200+ features)
2. Multi-model stacking ensemble
3. Cross-validation with out-of-fold predictions
4. Feature selection and interaction learning
5. Residual learning and error correction
6. Separate models for X and Y coordinates
7. Physics-informed predictions
"""

import os
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

np.random.seed(42)

print("=" * 80)
print("ADVANCED NFL TRAJECTORY PREDICTION")
print("=" * 80)
print("Target: Validation RMSE < 0.5 yards")
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
# DEEP FEATURE ENGINEERING
# ============================================================================

def engineer_deep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deep feature engineering with 200+ features."""
    print("\n[DEEP FEATURE ENGINEERING]")
    print("-" * 60)
    
    feats = df.copy()
    
    # === VELOCITY DECOMPOSITION ===
    print("  → Velocity vectors...")
    feats['vx'] = feats['s'] * np.cos(np.radians(feats['dir']))
    feats['vy'] = feats['s'] * np.sin(np.radians(feats['dir']))
    
    # === BALL-CENTRIC FEATURES ===
    print("  → Ball-centric features...")
    feats['dist_to_ball'] = np.sqrt(
        (feats['x'] - feats['ball_land_x']) ** 2 +
        (feats['y'] - feats['ball_land_y']) ** 2
    )
    
    dx_to_ball = feats['ball_land_x'] - feats['x']
    dy_to_ball = feats['ball_land_y'] - feats['y']
    feats['dx_to_ball'] = dx_to_ball
    feats['dy_to_ball'] = dy_to_ball
    
    feats['angle_to_ball'] = np.degrees(np.arctan2(dy_to_ball, dx_to_ball))
    feats['angle_to_ball_rad'] = np.arctan2(dy_to_ball, dx_to_ball)
    
    alignment = feats['dir'] - feats['angle_to_ball']
    feats['velocity_ball_alignment'] = np.cos(np.radians(alignment))
    feats['velocity_ball_alignment_sin'] = np.sin(np.radians(alignment))
    feats['moving_toward_ball'] = (feats['velocity_ball_alignment'] > 0).astype('int8')
    
    ball_dist_safe = feats['dist_to_ball'] + 1e-6
    feats['ball_direction_x'] = dx_to_ball / ball_dist_safe
    feats['ball_direction_y'] = dy_to_ball / ball_dist_safe
    
    # Ball proximity tiers (multi-scale)
    feats['very_close_to_ball'] = (feats['dist_to_ball'] < 3).astype('int8')
    feats['close_to_ball'] = (feats['dist_to_ball'] < 8).astype('int8')
    feats['near_ball'] = (feats['dist_to_ball'] < 15).astype('int8')
    feats['far_from_ball'] = (feats['dist_to_ball'] > 30).astype('int8')
    
    # === PHYSICS FEATURES ===
    print("  → Physics features...")
    feats['speed_squared'] = feats['s'] ** 2
    feats['speed_cubed'] = feats['s'] ** 3
    feats['kinetic_energy'] = 0.5 * feats['speed_squared']
    feats['momentum'] = feats['s'] * 200.0
    
    # Directional encoding (multiple representations)
    feats['dir_sin'] = np.sin(np.radians(feats['dir']))
    feats['dir_cos'] = np.cos(np.radians(feats['dir']))
    feats['o_sin'] = np.sin(np.radians(feats['o']))
    feats['o_cos'] = np.cos(np.radians(feats['o']))
    
    # Double angle encoding for periodic features
    feats['dir_sin2'] = np.sin(2 * np.radians(feats['dir']))
    feats['dir_cos2'] = np.cos(2 * np.radians(feats['dir']))
    
    # Body alignment
    diff = np.abs(feats['dir'] - feats['o'])
    feats['dir_o_diff'] = np.minimum(diff, 360 - diff)
    feats['dir_o_diff_rad'] = np.radians(feats['dir_o_diff'])
    feats['body_aligned'] = (feats['dir_o_diff'] < 30).astype('int8')
    feats['body_perpendicular'] = ((feats['dir_o_diff'] > 60) & (feats['dir_o_diff'] < 120)).astype('int8')
    
    # Acceleration features
    feats['accel_intensity'] = np.abs(feats['a'])
    feats['is_accelerating'] = (feats['a'] > 0.5).astype('int8')
    feats['is_decelerating'] = (feats['a'] < -0.5).astype('int8')
    feats['is_constant_speed'] = (np.abs(feats['a']) < 0.3).astype('int8')
    
    # === ROLE-SPECIFIC FEATURES ===
    print("  → Role-specific features...")
    feats['is_targeted_receiver'] = (feats['player_role'] == 'Targeted Receiver').astype('int8')
    feats['is_defensive_coverage'] = (feats['player_role'] == 'Defensive Coverage').astype('int8')
    feats['is_passer'] = (feats['player_role'] == 'Passer').astype('int8')
    feats['is_other_route_runner'] = (feats['player_role'] == 'Other Route Runner').astype('int8')
    feats['is_pass_rush'] = (feats['player_role'] == 'Pass Rush').astype('int8')
    
    feats['is_offense'] = (feats['player_side'] == 'Offense').astype('int8')
    feats['is_defense'] = (feats['player_side'] == 'Defense').astype('int8')
    
    # Position encoding
    feats['position_encoded'] = feats['player_position'].astype('category').cat.codes.astype('int16')
    
    # Position groups
    feats['is_wr'] = (feats['player_position'] == 'WR').astype('int8')
    feats['is_te'] = (feats['player_position'] == 'TE').astype('int8')
    feats['is_rb'] = (feats['player_position'] == 'RB').astype('int8')
    feats['is_qb'] = (feats['player_position'] == 'QB').astype('int8')
    feats['is_cb'] = (feats['player_position'] == 'CB').astype('int8')
    feats['is_safety'] = feats['player_position'].isin(['S', 'SS', 'FS']).astype('int8')
    feats['is_linebacker'] = feats['player_position'].isin(['LB', 'ILB', 'OLB', 'MLB']).astype('int8')
    
    # === SPATIAL CONTEXT ===
    print("  → Spatial features...")
    feats['x_norm'] = feats['x'] / 120.0
    feats['y_norm'] = feats['y'] / 53.3
    feats['ball_x_norm'] = feats['ball_land_x'] / 120.0
    feats['ball_y_norm'] = feats['ball_land_y'] / 53.3
    
    # Field boundaries
    feats['dist_to_left_sideline'] = feats['y']
    feats['dist_to_right_sideline'] = 53.3 - feats['y']
    feats['dist_to_sideline'] = np.minimum(feats['y'], 53.3 - feats['y'])
    feats['dist_to_near_endzone'] = np.minimum(feats['x'], 120 - feats['x'])
    feats['dist_to_offense_endzone'] = 120 - feats['x']
    
    feats['near_sideline'] = (feats['dist_to_sideline'] < 5).astype('int8')
    feats['in_red_zone'] = ((feats['x'] < 20) | (feats['x'] > 100)).astype('int8')
    feats['in_middle_field'] = ((feats['y'] > 15) & (feats['y'] < 38.3)).astype('int8')
    
    # Field zones (multiple granularities)
    feats['field_third'] = pd.cut(feats['x'], bins=[-np.inf, 40, 80, np.inf], labels=[0, 1, 2]).astype('int8')
    feats['field_quarter'] = pd.cut(feats['x'], bins=[-np.inf, 30, 60, 90, np.inf], labels=[0, 1, 2, 3]).astype('int8')
    feats['lateral_zone'] = pd.cut(feats['y'], bins=[-np.inf, 17.77, 35.53, np.inf], labels=[0, 1, 2]).astype('int8')
    
    # === TEMPORAL FEATURES ===
    print("  → Temporal features...")
    feats['frames_remaining'] = feats['num_frames_output']
    feats['time_to_ball'] = feats['frames_remaining'] / 10.0
    feats['time_to_ball_squared'] = feats['time_to_ball'] ** 2
    
    # Physics-based predictions (multiple orders)
    feats['predicted_x_linear'] = feats['x'] + feats['vx'] * feats['time_to_ball']
    feats['predicted_y_linear'] = feats['y'] + feats['vy'] * feats['time_to_ball']
    
    feats['predicted_x_accel'] = (
        feats['predicted_x_linear'] +
        0.5 * feats['a'] * feats['dir_cos'] * feats['time_to_ball_squared']
    )
    feats['predicted_y_accel'] = (
        feats['predicted_y_linear'] +
        0.5 * feats['a'] * feats['dir_sin'] * feats['time_to_ball_squared']
    )
    
    # Predictions with ball attraction
    ball_pull_strength = 0.3
    feats['predicted_x_ball_pull'] = (
        feats['predicted_x_linear'] + 
        ball_pull_strength * dx_to_ball * feats['is_targeted_receiver']
    )
    feats['predicted_y_ball_pull'] = (
        feats['predicted_y_linear'] + 
        ball_pull_strength * dy_to_ball * feats['is_targeted_receiver']
    )
    
    # === ADVANCED TRAJECTORY FEATURES ===
    print("  → Advanced trajectory features...")
    # Distance from predictions to ball
    feats['linear_pred_to_ball_dist'] = np.sqrt(
        (feats['predicted_x_linear'] - feats['ball_land_x']) ** 2 +
        (feats['predicted_y_linear'] - feats['ball_land_y']) ** 2
    )
    feats['accel_pred_to_ball_dist'] = np.sqrt(
        (feats['predicted_x_accel'] - feats['ball_land_x']) ** 2 +
        (feats['predicted_y_accel'] - feats['ball_land_y']) ** 2
    )
    
    # Relative positioning
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
    
    # Required speed to reach ball
    feats['dist_to_ball_per_time'] = feats['dist_to_ball'] / (feats['time_to_ball'] + 0.1)
    feats['speed_ratio_to_required'] = feats['s'] / (feats['dist_to_ball_per_time'] + 0.1)
    feats['can_reach_ball'] = (feats['speed_ratio_to_required'] > 0.8).astype('int8')
    
    # === INTERACTION FEATURES ===
    print("  → Interaction features...")
    # Play-level statistics
    play_group = feats.groupby(['game_id', 'play_id'])
    feats['speed_percentile'] = play_group['s'].transform(lambda x: x.rank(pct=True))
    feats['dist_to_ball_percentile'] = play_group['dist_to_ball'].transform(lambda x: x.rank(pct=True))
    feats['x_percentile'] = play_group['x'].transform(lambda x: x.rank(pct=True))
    
    feats['play_mean_speed'] = play_group['s'].transform('mean')
    feats['play_max_speed'] = play_group['s'].transform('max')
    feats['play_min_speed'] = play_group['s'].transform('min')
    feats['play_std_speed'] = play_group['s'].transform('std').fillna(0)
    
    feats['play_mean_dist_to_ball'] = play_group['dist_to_ball'].transform('mean')
    feats['play_min_dist_to_ball'] = play_group['dist_to_ball'].transform('min')
    feats['play_max_dist_to_ball'] = play_group['dist_to_ball'].transform('max')
    
    feats['play_density'] = play_group['nfl_id'].transform('count')
    feats['play_mean_x'] = play_group['x'].transform('mean')
    feats['play_std_x'] = play_group['x'].transform('std').fillna(0)
    feats['play_mean_y'] = play_group['y'].transform('mean')
    feats['play_std_y'] = play_group['y'].transform('std').fillna(0)
    
    # Team-level statistics
    side_group = feats.groupby(['game_id', 'play_id', 'player_side'])
    feats['team_mean_x'] = side_group['x'].transform('mean')
    feats['team_mean_y'] = side_group['y'].transform('mean')
    feats['team_speed_mean'] = side_group['s'].transform('mean')
    feats['team_speed_max'] = side_group['s'].transform('max')
    feats['team_speed_std'] = side_group['s'].transform('std').fillna(0)
    feats['team_size'] = side_group['nfl_id'].transform('count')
    
    feats[['team_mean_x', 'team_mean_y', 'team_speed_mean', 'team_speed_max', 'team_speed_std', 'team_size']] = (
        feats[['team_mean_x', 'team_mean_y', 'team_speed_mean', 'team_speed_max', 'team_speed_std', 'team_size']].fillna(0.0)
    )
    
    feats['offset_team_center_x'] = feats['x'] - feats['team_mean_x']
    feats['offset_team_center_y'] = feats['y'] - feats['team_mean_y']
    feats['dist_to_team_center'] = np.sqrt(
        feats['offset_team_center_x'] ** 2 + feats['offset_team_center_y'] ** 2
    )
    
    feats['speed_vs_team_mean'] = feats['s'] - feats['team_speed_mean']
    feats['faster_than_team'] = (feats['speed_vs_team_mean'] > 0).astype('int8')
    
    # === POLYNOMIAL & INTERACTION FEATURES ===
    print("  → Polynomial features...")
    # Speed-distance interactions
    feats['speed_dist_product'] = feats['s'] * feats['dist_to_ball']
    feats['speed_squared_dist'] = feats['speed_squared'] * feats['dist_to_ball']
    feats['vx_dist'] = feats['vx'] * feats['dist_to_ball']
    feats['vy_dist'] = feats['vy'] * feats['dist_to_ball']
    
    # Time-space interactions
    feats['time_x_product'] = feats['time_to_ball'] * feats['x_norm']
    feats['time_y_product'] = feats['time_to_ball'] * feats['y_norm']
    feats['time_dist_product'] = feats['time_to_ball'] * feats['dist_to_ball']
    feats['time_squared_dist'] = feats['time_to_ball_squared'] * feats['dist_to_ball']
    
    # Role-position interactions
    feats['receiver_x'] = feats['is_targeted_receiver'] * feats['x_norm']
    feats['receiver_y'] = feats['is_targeted_receiver'] * feats['y_norm']
    feats['receiver_dist'] = feats['is_targeted_receiver'] * feats['dist_to_ball']
    feats['receiver_speed'] = feats['is_targeted_receiver'] * feats['s']
    
    feats['defender_x'] = feats['is_defensive_coverage'] * feats['x_norm']
    feats['defender_y'] = feats['is_defensive_coverage'] * feats['y_norm']
    feats['defender_dist'] = feats['is_defensive_coverage'] * feats['dist_to_ball']
    
    # Velocity-acceleration synergy
    feats['vel_accel_product'] = feats['s'] * feats['a']
    feats['vx_accel'] = feats['vx'] * feats['a']
    feats['vy_accel'] = feats['vy'] * feats['a']
    
    # === EXPONENTIAL & LOG FEATURES ===
    print("  → Exponential & log features...")
    feats['log_dist_to_ball'] = np.log1p(feats['dist_to_ball'])
    feats['log_time_to_ball'] = np.log1p(feats['time_to_ball'])
    feats['log_speed'] = np.log1p(feats['s'])
    feats['log1p_x'] = np.log1p(feats['x'])
    feats['log1p_y'] = np.log1p(feats['y'])
    
    # Exponential decay (importance decreases with distance/time)
    feats['exp_neg_dist'] = np.exp(-feats['dist_to_ball'] / 10.0)
    feats['exp_neg_dist_fast'] = np.exp(-feats['dist_to_ball'] / 5.0)
    feats['exp_neg_time'] = np.exp(-feats['time_to_ball'])
    
    # Squared features
    feats['dist_to_ball_squared'] = feats['dist_to_ball'] ** 2
    feats['dist_to_sideline_squared'] = feats['dist_to_sideline'] ** 2
    feats['x_squared'] = feats['x'] ** 2
    feats['y_squared'] = feats['y'] ** 2
    
    # === RATIO FEATURES ===
    print("  → Ratio features...")
    feats['vx_vy_ratio'] = feats['vx'] / (np.abs(feats['vy']) + 0.1)
    feats['vy_vx_ratio'] = feats['vy'] / (np.abs(feats['vx']) + 0.1)
    feats['x_y_ratio'] = feats['x'] / (feats['y'] + 0.1)
    feats['ball_x_y_ratio'] = feats['ball_land_x'] / (feats['ball_land_y'] + 0.1)
    feats['accel_speed_ratio'] = feats['a'] / (feats['s'] + 0.1)
    feats['dist_speed_ratio'] = feats['dist_to_ball'] / (feats['s'] + 0.1)
    
    # === BALL DOMINANCE & CATCHABILITY ===
    print("  → Ball dominance features...")
    feats['catchability_score'] = (
        (1.0 / (1.0 + feats['dist_to_ball'])) *
        (feats['velocity_ball_alignment'] + 1.0) / 2.0 *
        (1.0 - feats['speed_percentile'] * 0.3)
    )
    
    feats['ball_dominance'] = (
        feats['is_targeted_receiver'] * 
        feats['exp_neg_dist'] * 
        (feats['moving_toward_ball'] + 1) / 2
    )
    
    feats['interception_potential'] = (
        feats['is_defensive_coverage'] *
        feats['exp_neg_dist'] *
        feats['moving_toward_ball'] *
        feats['speed_percentile']
    )
    
    # === DIRECTIONAL INTENSITY ===
    print("  → Directional intensity...")
    feats['directional_strength'] = np.sqrt(feats['dir_cos']**2 + feats['dir_sin']**2) * feats['s']
    feats['orientation_strength'] = np.sqrt(feats['o_cos']**2 + feats['o_sin']**2) * feats['s']
    feats['angular_momentum'] = feats['s'] * feats['dir_o_diff_rad']
    
    # Momentum features
    feats['momentum_toward_ball'] = feats['momentum'] * feats['velocity_ball_alignment']
    feats['momentum_x'] = feats['momentum'] * feats['dir_cos']
    feats['momentum_y'] = feats['momentum'] * feats['dir_sin']
    
    # === CROSS FEATURES ===
    print("  → Cross features...")
    feats['x_ball_interaction'] = feats['x_norm'] * feats['ball_x_norm']
    feats['y_ball_interaction'] = feats['y_norm'] * feats['ball_y_norm']
    feats['speed_alignment'] = feats['s'] * feats['velocity_ball_alignment']
    
    # Quadrant features
    feats['in_left_half'] = (feats['y'] < 26.65).astype('int8')
    feats['in_offensive_half'] = (feats['x'] > 60).astype('int8')
    feats['ball_in_left_half'] = (feats['ball_land_y'] < 26.65).astype('int8')
    feats['ball_in_offensive_half'] = (feats['ball_land_x'] > 60).astype('int8')
    
    # Same quadrant as ball
    feats['same_lateral_zone'] = (
        (feats['in_left_half'] == feats['ball_in_left_half'])
    ).astype('int8')
    feats['same_field_half'] = (
        (feats['in_offensive_half'] == feats['ball_in_offensive_half'])
    ).astype('int8')
    
    # Trajectory confidence
    feats['trajectory_confidence'] = (
        feats['velocity_ball_alignment'] * 
        feats['moving_toward_ball'] * 
        (1.0 / (1.0 + feats['dir_o_diff'] / 180.0))
    )

    # === COVERAGE & PRESSURE INTELLIGENCE ===
    print("  → Coverage intelligence...")

    def compute_frame_level_coverage(frame_df: pd.DataFrame) -> pd.DataFrame:
        frame = frame_df.copy()
        defense_mask = frame['is_defense'].values.astype(bool)
        offense_mask = frame['is_offense'].values.astype(bool)
        coords = frame[['x', 'y']].values.astype('float32')

        if defense_mask.any():
            defense_coords = coords[defense_mask]
            diffs = coords[:, None, :] - defense_coords[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            nearest_defender = dists.min(axis=1).astype('float32')
        else:
            nearest_defender = np.full(len(frame), np.nan, dtype='float32')

        frame['nearest_defender_dist'] = nearest_defender
        frame['receiver_nearest_defender'] = np.where(offense_mask, nearest_defender, np.nan).astype('float32')

        if defense_mask.any():
            defense_dist_to_ball = frame.loc[defense_mask, 'dist_to_ball'].values.astype('float32')
            close_defenders = float((defense_dist_to_ball < 5.0).sum())
            zone_defenders = float((defense_dist_to_ball < 10.0).sum())
            zone_area = np.pi * (10.0 ** 2)
            pressure_density = zone_defenders / zone_area if zone_area > 0 else 0.0
        else:
            close_defenders = 0.0
            pressure_density = 0.0

        frame['defensive_pressure'] = np.full(len(frame), close_defenders, dtype='float32')
        frame['defensive_pressure_density'] = np.full(len(frame), pressure_density, dtype='float32')

        target_mask = frame['is_targeted_receiver'].values.astype(bool)

        if target_mask.any() and defense_mask.any():
            target_row = frame[target_mask].iloc[0]
            target_vec = np.array([
                target_row['ball_land_x'] - target_row['x'],
                target_row['ball_land_y'] - target_row['y']
            ], dtype='float32')
            target_norm = np.linalg.norm(target_vec) + 1e-6
            target_unit = target_vec / target_norm

            defense_vecs = np.column_stack([
                frame.loc[defense_mask, 'ball_land_x'].values.astype('float32') - frame.loc[defense_mask, 'x'].values.astype('float32'),
                frame.loc[defense_mask, 'ball_land_y'].values.astype('float32') - frame.loc[defense_mask, 'y'].values.astype('float32')
            ])
            defense_norms = np.linalg.norm(defense_vecs, axis=1, keepdims=True) + 1e-6
            defense_unit = defense_vecs / defense_norms
            cos_angles = np.clip(defense_unit @ target_unit, -1.0, 1.0)
            angles_deg = np.degrees(np.arccos(cos_angles)).astype('float32')
            min_angle_deg = float(angles_deg.min())
            coverage_score = 1.0 - min_angle_deg / 180.0

            frame['angle_coverage_score'] = np.full(len(frame), coverage_score, dtype='float32')
            frame['angle_coverage_min_deg'] = np.full(len(frame), min_angle_deg, dtype='float32')
            frame.loc[defense_mask, 'defender_intercept_alignment'] = cos_angles.astype('float32')
            frame.loc[defense_mask, 'defender_angle_to_target_deg'] = angles_deg
        else:
            frame['angle_coverage_score'] = np.full(len(frame), np.nan, dtype='float32')
            frame['angle_coverage_min_deg'] = np.full(len(frame), np.nan, dtype='float32')
            frame['defender_intercept_alignment'] = np.full(len(frame), np.nan, dtype='float32')
            frame['defender_angle_to_target_deg'] = np.full(len(frame), np.nan, dtype='float32')

        return frame

    feats = feats.groupby(['game_id', 'play_id', 'frame_id'], group_keys=False).apply(compute_frame_level_coverage)

    # === ROUTE SIMILARITY FEATURES ===
    print("  → Route similarity features...")
    player_mean_v = feats.groupby('nfl_id')[['vx', 'vy']].transform('mean').astype('float32')
    current_vec = feats[['vx', 'vy']].values.astype('float32')
    mean_vec = player_mean_v.values.astype('float32')
    current_norm = np.linalg.norm(current_vec, axis=1) + 1e-6
    mean_norm = np.linalg.norm(mean_vec, axis=1) + 1e-6
    dot_product = (current_vec * mean_vec).sum(axis=1)
    feats['route_similarity'] = (dot_product / (current_norm * mean_norm)).astype('float32')
    feats['route_similarity'] = feats['route_similarity'].clip(-1.0, 1.0)

    mean_speed = feats.groupby('nfl_id')['s'].transform('mean').astype('float32')
    feats['route_speed_delta'] = (feats['s'] - mean_speed).astype('float32')

    mean_dir_sin = feats.groupby('nfl_id')['dir_sin'].transform('mean').astype('float32')
    mean_dir_cos = feats.groupby('nfl_id')['dir_cos'].transform('mean').astype('float32')
    current_dir_sin = feats['dir_sin'].values.astype('float32')
    current_dir_cos = feats['dir_cos'].values.astype('float32')
    dir_dot = current_dir_sin * mean_dir_sin + current_dir_cos * mean_dir_cos
    dir_norms = np.sqrt((current_dir_sin ** 2 + current_dir_cos ** 2) * (mean_dir_sin ** 2 + mean_dir_cos ** 2)) + 1e-6
    feats['route_direction_similarity'] = (dir_dot / dir_norms).astype('float32')
    feats['route_direction_similarity'] = feats['route_direction_similarity'].clip(-1.0, 1.0)
    
    print(f"✓ Feature engineering complete: {len(feats.columns)} total features")
    
    return feats


# ============================================================================
# TRAINING PAIR GENERATION
# ============================================================================

def create_pairs(input_df: pd.DataFrame, output_df: pd.DataFrame, max_samples: int = None) -> pd.DataFrame:
    """Create training pairs with enhanced trajectory encoding."""
    print("\n[CREATING TRAINING PAIRS]")
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
            
            # Multi-frame trajectory analysis
            if len(player_in) >= 2:
                prev_frame = player_in.iloc[-2]
                vel_dx = last_frame['vx'] - prev_frame['vx']
                vel_dy = last_frame['vy'] - prev_frame['vy']
                pos_dx = last_frame['x'] - prev_frame['x']
                pos_dy = last_frame['y'] - prev_frame['y']
                
                if len(player_in) >= 3:
                    frame_minus_2 = player_in.iloc[-3]
                    trajectory_curvature = np.hypot(
                        last_frame['x'] - 2 * prev_frame['x'] + frame_minus_2['x'],
                        last_frame['y'] - 2 * prev_frame['y'] + frame_minus_2['y']
                    )
                    
                    # Second derivative (jerk)
                    prev_vel_dx = prev_frame['vx'] - frame_minus_2['vx']
                    prev_vel_dy = prev_frame['vy'] - frame_minus_2['vy']
                    jerk_x = vel_dx - prev_vel_dx
                    jerk_y = vel_dy - prev_vel_dy
                else:
                    trajectory_curvature = 0.0
                    jerk_x = jerk_y = 0.0
            else:
                vel_dx = vel_dy = pos_dx = pos_dy = trajectory_curvature = jerk_x = jerk_y = 0.0
            
            accel_est = np.hypot(vel_dx, vel_dy)
            jerk_est = np.hypot(jerk_x, jerk_y)
            
            for _, out_row in player_out.iterrows():
                if max_samples and created >= max_samples:
                    break
                
                time_diff = out_row['frame_id'] - last_frame['frame_id']
                
                sample = {
                    'target_x': float(out_row['x']),
                    'target_y': float(out_row['y']),
                    'output_frame_id': int(out_row['frame_id']),
                    'time_diff': float(time_diff),
                    'game_id': int(game_id),
                    'play_id': int(play_id),
                    'nfl_id': int(nfl_id),
                    'velocity_change_x': float(vel_dx),
                    'velocity_change_y': float(vel_dy),
                    'position_change_x': float(pos_dx),
                    'position_change_y': float(pos_dy),
                    'acceleration_est': float(accel_est),
                    'trajectory_curvature': float(trajectory_curvature),
                    'jerk_x': float(jerk_x),
                    'jerk_y': float(jerk_y),
                    'jerk_est': float(jerk_est),
                    'is_target_player': int(is_target)
                }
                
                # Add all engineered features (only numeric)
                for col in last_frame.index:
                    if col not in ['game_id', 'play_id', 'nfl_id', 'player_position', 
                                   'player_role', 'player_side', 'direction']:
                        try:
                            sample[f'input_{col}'] = float(last_frame[col])
                        except (ValueError, TypeError):
                            # Skip non-numeric columns
                            pass
                
                pairs.append(sample)
                created += 1
    
    df = pd.DataFrame(pairs)
    print(f"✓ Created {len(df):,} pairs")
    if 'is_target_player' in df.columns:
        print(f"  Target player ratio: {100 * df['is_target_player'].mean():.1f}%")
    
    return df


# ============================================================================
# PAIR-LEVEL FEATURE AUGMENTATION
# ============================================================================

def augment_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich pair-level features with horizon-aware signals."""
    df = df.copy()
    df['time_diff_squared'] = df['time_diff'] ** 2
    df['time_diff_cubed'] = df['time_diff'] ** 3
    time_nonneg = np.maximum(df['time_diff'].astype(float), 0.0)
    df['time_diff_log'] = np.log1p(time_nonneg)
    df['time_diff_inv'] = 1.0 / (1.0 + time_nonneg)
    df['time_weight_feature'] = np.exp(-time_nonneg / 6.0)
    df['time_diff_normalized'] = time_nonneg / (time_nonneg.max() + 1e-6)
    horizon_bucket = pd.cut(
        df['time_diff'],
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=[0, 1, 2, 3]
    ).astype('int8')
    df['time_horizon_bucket'] = horizon_bucket.astype('float32')
    df['is_short_horizon'] = (df['time_diff'] <= 3).astype('int8')
    df['is_mid_horizon'] = ((df['time_diff'] > 3) & (df['time_diff'] <= 7)).astype('int8')
    df['is_long_horizon'] = (df['time_diff'] > 7).astype('int8')
    return df


# ============================================================================
# FEATURE SELECTION UTILITIES
# ============================================================================

def select_features_via_permutation(
    features: pd.DataFrame,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: List[str],
    random_state: int = 42
) -> List[str]:
    """Perform permutation-based feature selection on a representative sample."""
    print("\n[FEATURE SELECTION]")
    print("  → Permutation importance (LightGBM baseline)...")

    n_samples = len(features)
    if n_samples == 0:
        return feature_names

    sample_size = min(30000, n_samples)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

    X_sample = features.iloc[sample_idx]
    w_sample = sample_weights[sample_idx]
    y_sample = targets[sample_idx]

    model_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )

    model_x = lgb.LGBMRegressor(**model_params)
    model_x.fit(X_sample, y_sample[:, 0], sample_weight=w_sample)
    perm_x = permutation_importance(
        model_x,
        X_sample,
        y_sample[:, 0],
        scoring='neg_mean_squared_error',
        n_repeats=4,
        random_state=random_state,
        n_jobs=-1,
        sample_weight=w_sample
    )

    model_y = lgb.LGBMRegressor(**model_params)
    model_y.fit(X_sample, y_sample[:, 1], sample_weight=w_sample)
    perm_y = permutation_importance(
        model_y,
        X_sample,
        y_sample[:, 1],
        scoring='neg_mean_squared_error',
        n_repeats=4,
        random_state=random_state + 7,
        n_jobs=-1,
        sample_weight=w_sample
    )

    importance_scores = (
        np.maximum(perm_x.importances_mean, 0.0) +
        np.maximum(perm_y.importances_mean, 0.0)
    ) / 2.0

    threshold = np.percentile(importance_scores, 20)
    keep_mask = importance_scores >= threshold
    keep_features = [name for name, keep in zip(feature_names, keep_mask) if keep]

    # Ensure we retain a reasonable number of predictors
    min_features = max(80, int(len(feature_names) * 0.5))
    if len(keep_features) < min_features:
        threshold = np.percentile(importance_scores, 40)
        keep_mask = importance_scores >= threshold
        keep_features = [name for name, keep in zip(feature_names, keep_mask) if keep]

    if len(keep_features) < min_features:
        keep_features = feature_names
        print("  → Retaining all features (importance scores too uniform)")
    else:
        removed = len(feature_names) - len(keep_features)
        print(f"  → Selected {len(keep_features)} / {len(feature_names)} features (removed {removed})")

    return keep_features


# ============================================================================
# ADVANCED ENSEMBLE WITH CROSS-VALIDATION
# ============================================================================

def train_advanced_ensemble(X_train, y_train, X_val, y_val, sample_weights, base_val, train_groups):
    """
    Advanced ensemble with:
    - Cross-validated out-of-fold predictions
    - Stacking with neural network meta-learner
    - Residual learning
    - Separate X and Y models for better precision
    """
    print("\n[TRAINING ADVANCED ENSEMBLE]")
    print("-" * 60)
    
    n_folds = 5
    kf = GroupKFold(n_splits=n_folds)
    splits = list(kf.split(X_train, y_train[:, 0], groups=train_groups))
    
    def clip_delta(preds_delta: np.ndarray) -> np.ndarray:
        abs_preds = preds_delta + base_val
        abs_preds[:, 0] = np.clip(abs_preds[:, 0], 0.0, 120.0)
        abs_preds[:, 1] = np.clip(abs_preds[:, 1], 0.0, 53.3)
        return abs_preds - base_val

    # Base models with different strengths
    base_models = {
        'XGB1': xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=1.5,
            gamma=0.3,
            min_child_weight=7,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        ),
        'XGB2': xgb.XGBRegressor(
            n_estimators=800,
            max_depth=12,
            learning_rate=0.025,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=2.0,
            reg_alpha=0.8,
            gamma=0.15,
            min_child_weight=5,
            random_state=43,
            tree_method='hist',
            n_jobs=-1
        ),
        'LGBM1': lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=9,
            learning_rate=0.02,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=2.5,
            reg_alpha=1.0,
            min_child_samples=30,
            num_leaves=80,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'LGBM2': lgb.LGBMRegressor(
            n_estimators=800,
            max_depth=11,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            reg_alpha=0.7,
            min_child_samples=25,
            num_leaves=100,
            random_state=43,
            n_jobs=-1,
            verbose=-1
        ),
        'CAT': cb.CatBoostRegressor(
            iterations=800,
            depth=9,
            learning_rate=0.03,
            l2_leaf_reg=3.5,
            subsample=0.88,
            random_state=42,
            verbose=False
        ),
    }

    model_names = list(base_models.keys())
    n_samples = X_train.shape[0]
    
    # Train separate models for X and Y coordinates for better precision
    print("\n  → Training models for X-coordinate...")
    models_x = {}
    oof_preds_x = {name: np.zeros(n_samples) for name in model_names}
    val_preds_x = {}
    
    for name, base_model in base_models.items():
        print(f"    - {name}")
        val_fold_preds = []
        
        for fold, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx, 0], y_train[va_idx, 0]
            w_tr = sample_weights[tr_idx]
            
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            
            oof_preds_x[name][va_idx] = model.predict(X_va)
            val_fold_preds.append(model.predict(X_val))
        
        val_preds_x[name] = np.mean(val_fold_preds, axis=0)
        
        # Train final model on full training data
        final_model = base_model.__class__(**base_model.get_params())
        final_model.fit(X_train, y_train[:, 0], sample_weight=sample_weights)
        models_x[name] = final_model
    
    print("\n  → Training models for Y-coordinate...")
    models_y = {}
    oof_preds_y = {name: np.zeros(n_samples) for name in model_names}
    val_preds_y = {}
    
    for name, base_model in base_models.items():
        print(f"    - {name}")
        val_fold_preds = []
        
        for fold, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx, 1], y_train[va_idx, 1]
            w_tr = sample_weights[tr_idx]
            
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            
            oof_preds_y[name][va_idx] = model.predict(X_va)
            val_fold_preds.append(model.predict(X_val))
        
        val_preds_y[name] = np.mean(val_fold_preds, axis=0)
        
        # Train final model on full training data
        final_model = base_model.__class__(**base_model.get_params())
        final_model.fit(X_train, y_train[:, 1], sample_weight=sample_weights)
        models_y[name] = final_model
    
    # Stack predictions for meta-learner
    print("\n  → Stacking predictions...")
    meta_train_x = np.column_stack([oof_preds_x[name] for name in model_names])
    meta_val_x = np.column_stack([val_preds_x[name] for name in model_names])
    
    meta_train_y = np.column_stack([oof_preds_y[name] for name in model_names])
    meta_val_y = np.column_stack([val_preds_y[name] for name in model_names])
    
    # Simple weighted average ensemble
    print("\n  → Computing weighted ensemble...")
    ensemble_val_x = np.mean(meta_val_x, axis=1)
    ensemble_val_y = np.mean(meta_val_y, axis=1)
    ensemble_val = clip_delta(np.column_stack([ensemble_val_x, ensemble_val_y]))
    ensemble_val_x = ensemble_val[:, 0]
    ensemble_val_y = ensemble_val[:, 1]
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
    ensemble_rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], ensemble_val_x))
    ensemble_rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], ensemble_val_y))
    
    print(f"\n    Ensemble RMSE: {ensemble_rmse:.4f} yards")
    print(f"      X: {ensemble_rmse_x:.4f} yards")
    print(f"      Y: {ensemble_rmse_y:.4f} yards")
    
    # Train meta-learner (Ridge regression for stability)
    print("\n  → Training meta-learner...")
    meta_model_x = Ridge(alpha=10.0)
    meta_model_y = Ridge(alpha=10.0)
    
    # Use just the average for meta features to avoid overfitting
    meta_model_x.fit(meta_train_x, y_train[:, 0])
    meta_model_y.fit(meta_train_y, y_train[:, 1])
    
    stacked_val_x = meta_model_x.predict(meta_val_x)
    stacked_val_y = meta_model_y.predict(meta_val_y)
    stacked_val = clip_delta(np.column_stack([stacked_val_x, stacked_val_y]))
    stacked_val_x = stacked_val[:, 0]
    stacked_val_y = stacked_val[:, 1]
    
    stacked_rmse = np.sqrt(mean_squared_error(y_val, stacked_val))
    stacked_rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], stacked_val_x))
    stacked_rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], stacked_val_y))
    
    print(f"\n    Stacked RMSE: {stacked_rmse:.4f} yards")
    print(f"      X: {stacked_rmse_x:.4f} yards")
    print(f"      Y: {stacked_rmse_y:.4f} yards")
    
    # Choose best approach
    if stacked_rmse < ensemble_rmse - 1e-4:
        print(f"\n  ✓ Using stacked meta-learner (improvement: {ensemble_rmse - stacked_rmse:.4f} yards)")
        final_pred = stacked_val
        final_rmse = stacked_rmse
        final_rmse_x = stacked_rmse_x
        final_rmse_y = stacked_rmse_y
        use_stacking = True
        train_primary_pred_x = meta_model_x.predict(meta_train_x)
        train_primary_pred_y = meta_model_y.predict(meta_train_y)
        val_primary_pred_x = stacked_val_x
        val_primary_pred_y = stacked_val_y
    else:
        print(f"\n  ✓ Using weighted ensemble (better by {stacked_rmse - ensemble_rmse:.4f} yards)")
        final_pred = ensemble_val
        final_rmse = ensemble_rmse
        final_rmse_x = ensemble_rmse_x
        final_rmse_y = ensemble_rmse_y
        use_stacking = False
        train_primary_pred_x = np.mean(meta_train_x, axis=1)
        train_primary_pred_y = np.mean(meta_train_y, axis=1)
        val_primary_pred_x = ensemble_val_x
        val_primary_pred_y = ensemble_val_y

    # Residual refinement to capture missed dynamics
    print("\n  → Residual refinement with gradient boosting...")
    residual_model_x = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective='huber',
        random_state=101,
        n_jobs=-1,
        verbose=-1
    )
    residual_model_y = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective='huber',
        random_state=202,
        n_jobs=-1,
        verbose=-1
    )

    residual_target_x = y_train[:, 0] - train_primary_pred_x
    residual_target_y = y_train[:, 1] - train_primary_pred_y

    residual_model_x.fit(X_train, residual_target_x, sample_weight=sample_weights)
    residual_model_y.fit(X_train, residual_target_y, sample_weight=sample_weights)

    residual_val_x = residual_model_x.predict(X_val)
    residual_val_y = residual_model_y.predict(X_val)

    corrected_val = clip_delta(
        np.column_stack([
            val_primary_pred_x + residual_val_x,
            val_primary_pred_y + residual_val_y
        ])
    )
    corrected_val_x = corrected_val[:, 0]
    corrected_val_y = corrected_val[:, 1]

    corrected_rmse = np.sqrt(mean_squared_error(y_val, corrected_val))
    corrected_rmse_x = np.sqrt(mean_squared_error(y_val[:, 0], corrected_val_x))
    corrected_rmse_y = np.sqrt(mean_squared_error(y_val[:, 1], corrected_val_y))

    if corrected_rmse < final_rmse - 1e-4:
        print(f"    ✓ Residual booster improved RMSE by {final_rmse - corrected_rmse:.4f} yards")
        final_pred = corrected_val
        final_rmse = corrected_rmse
        final_rmse_x = corrected_rmse_x
        final_rmse_y = corrected_rmse_y
        use_residuals = True
    else:
        print(f"    → Residual booster did not improve (Δ={final_rmse - corrected_rmse:.4f}); keeping primary ensemble")
        residual_model_x = None
        residual_model_y = None
        use_residuals = False
    
    return {
        'models_x': models_x,
        'models_y': models_y,
        'meta_model_x': meta_model_x if use_stacking else None,
        'meta_model_y': meta_model_y if use_stacking else None,
        'use_stacking': use_stacking,
        'use_residuals': use_residuals,
        'residual_model_x': residual_model_x,
        'residual_model_y': residual_model_y,
        'val_rmse': final_rmse,
        'val_rmse_x': final_rmse_x,
        'val_rmse_y': final_rmse_y,
        'model_names': model_names
    }


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
    
    # Engineer deep features
    train_input = engineer_deep_features(train_input)
    val_input = engineer_deep_features(val_input)
    
    # Create training pairs
    train_df = create_pairs(train_input, train_output, max_samples=500_000)
    val_df = create_pairs(val_input, val_output, max_samples=100_000)

    train_df = augment_pair_features(train_df)
    val_df = augment_pair_features(val_df)
    
    del train_input, train_output, val_input, val_output
    gc.collect()
    
    # Prepare features
    feature_columns = [col for col in train_df.columns if col.startswith('input_')]
    feature_columns += [
        'time_diff', 'output_frame_id',
        'velocity_change_x', 'velocity_change_y',
        'position_change_x', 'position_change_y',
        'acceleration_est', 'trajectory_curvature',
        'jerk_x', 'jerk_y', 'jerk_est',
        'is_target_player'
    ]

    pair_feature_columns = [
        'time_diff_squared', 'time_diff_cubed', 'time_diff_log',
        'time_diff_inv', 'time_weight_feature', 'time_diff_normalized',
        'time_horizon_bucket', 'is_short_horizon',
        'is_mid_horizon', 'is_long_horizon'
    ]
    feature_columns += pair_feature_columns
    feature_columns = list(dict.fromkeys(feature_columns))
    
    train_features_df = train_df[feature_columns].fillna(0.0)
    val_features_df = val_df[feature_columns].fillna(0.0)

    base_train = train_df[['input_x', 'input_y']].values
    y_train = train_df[['target_x', 'target_y']].values - base_train
    time_importance = train_df['time_weight_feature'].values
    train_weights = 1.0 + 2.0 * time_importance + train_df['is_target_player'].values
    train_groups = train_df['game_id'].values

    base_val = val_df[['input_x', 'input_y']].values
    y_val = val_df[['target_x', 'target_y']].values - base_val

    selected_features = select_features_via_permutation(
        train_features_df,
        y_train,
        train_weights,
        feature_columns
    )

    train_features_df = train_features_df[selected_features]
    val_features_df = val_features_df[selected_features]
    feature_columns = selected_features

    X_train = train_features_df.values
    X_val = val_features_df.values
    
    print(f"\n[CONFIGURATION]")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    
    del train_df, val_df
    gc.collect()
    
    # Scale features (two-stage scaling for better distribution)
    print("\n[FEATURE SCALING]")
    print("  → Robust scaling...")
    scaler1 = RobustScaler()
    X_train = scaler1.fit_transform(X_train)
    X_val = scaler1.transform(X_val)
    
    print("  → Quantile transformation...")
    scaler2 = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    X_train = scaler2.fit_transform(X_train)
    X_val = scaler2.transform(X_val)
    
    # Train advanced ensemble
    results = train_advanced_ensemble(X_train, y_train, X_val, y_val, train_weights, base_val, train_groups)
    
    # Calculate training RMSE
    print("\n[FINAL EVALUATION]")
    print("=" * 60)
    
    # Training predictions (on a sample for speed)
    sample_size = min(50000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[sample_idx]
    y_train_sample = y_train[sample_idx]
    base_train_sample = base_train[sample_idx]
    
    model_order = results['model_names']
    models_x = results['models_x']
    models_y = results['models_y']
    
    base_preds_x = np.column_stack([models_x[name].predict(X_train_sample) for name in model_order])
    base_preds_y = np.column_stack([models_y[name].predict(X_train_sample) for name in model_order])

    if results['use_stacking'] and results['meta_model_x'] is not None:
        train_pred_x = results['meta_model_x'].predict(base_preds_x)
        train_pred_y = results['meta_model_y'].predict(base_preds_y)
    else:
        train_pred_x = np.mean(base_preds_x, axis=1)
        train_pred_y = np.mean(base_preds_y, axis=1)

    if results['use_residuals'] and results['residual_model_x'] is not None:
        train_pred_x += results['residual_model_x'].predict(X_train_sample)
        train_pred_y += results['residual_model_y'].predict(X_train_sample)

    train_pred_delta = np.column_stack([train_pred_x, train_pred_y])
    train_pred_abs = train_pred_delta + base_train_sample
    train_pred_abs[:, 0] = np.clip(train_pred_abs[:, 0], 0.0, 120.0)
    train_pred_abs[:, 1] = np.clip(train_pred_abs[:, 1], 0.0, 53.3)
    train_pred = train_pred_abs - base_train_sample
    train_rmse = np.sqrt(mean_squared_error(y_train_sample, train_pred))
    train_rmse_x = np.sqrt(mean_squared_error(y_train_sample[:, 0], train_pred[:, 0]))
    train_rmse_y = np.sqrt(mean_squared_error(y_train_sample[:, 1], train_pred[:, 1]))
    
    print(f"\n📊 TRAINING RMSE (on {sample_size:,} samples):")
    print(f"  Overall: {train_rmse:.4f} yards")
    print(f"  X-coord: {train_rmse_x:.4f} yards")
    print(f"  Y-coord: {train_rmse_y:.4f} yards")
    
    print(f"\n📊 VALIDATION RMSE:")
    print(f"  Overall: {results['val_rmse']:.4f} yards")
    print(f"  X-coord: {results['val_rmse_x']:.4f} yards")
    print(f"  Y-coord: {results['val_rmse_y']:.4f} yards")

    if results['use_residuals']:
        print("\n✨ Residual booster: active (gradient boosted correction applied)")
    else:
        print("\nℹ️ Residual booster: inactive (primary ensemble retained)")
    
    # Overfitting analysis
    overfit_margin = results['val_rmse'] - train_rmse
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
    
    # Target achievement
    print(f"\n🎯 TARGET STATUS:")
    if results['val_rmse'] < 0.5:
        print(f"  ✅ TARGET ACHIEVED! Validation RMSE < 0.5 yards")
        print(f"  Margin: {0.5 - results['val_rmse']:.4f} yards below target")
    else:
        print(f"  ⚠️ Close to target. Need {results['val_rmse'] - 0.5:.4f} yards improvement")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
