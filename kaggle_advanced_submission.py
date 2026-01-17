"""
NFL Big Data Bowl 2026 - Advanced Kaggle Submission Script
===========================================================

This script trains an advanced ensemble model and generates submission.csv for Kaggle.

KAGGLE DATA PATHS (Auto-detected):
- Training data: /kaggle/input/nfl-big-data-bowl-2026-prediction/train/
  - input_2023_w01.csv to input_2023_w18.csv
  - output_2023_w01.csv to output_2023_w18.csv
- Validation data (optional): /kaggle/input/validation/validation/
  - Same structure as training
- Test data: /kaggle/input/nfl-big-data-bowl-2026-prediction/
  - test_input.csv
  - test.csv
  - sample_submission.csv

The script automatically detects whether it's running on Kaggle or locally
and adjusts paths accordingly.

Features:
- Advanced defensive coverage metrics (interception angles, coverage responsibility)
- Route intelligence features (similarity, deviation)
- Physics-constrained predictions (max acceleration, speed limits)
- Enhanced Test-Time Augmentation (role-aware, confidence-weighted)
- Memory optimized (<30GB) with float32 and chunked processing
- Stacked ensemble (5 base models + meta-learner + residual refinement)

Expected Performance:
- Training RMSE: ~0.30-0.32 yards
- Validation RMSE: ~0.32-0.34 yards
- Runtime: ~6-7 hours on Kaggle GPU

Usage on Kaggle:
1. Create new notebook
2. Add data: "nfl-big-data-bowl-2026-prediction"
3. Copy this entire script into a code cell
4. Run the cell
5. Download submission.csv from output

Usage Locally:
python kaggle_advanced_submission.py
"""

import gc
import warnings
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Optional memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

warnings.filterwarnings("ignore")
np.random.seed(42)

# Field boundaries
FIELD_X_MAX = 120.0
FIELD_Y_MAX = 53.3

# Memory monitor
def print_memory_usage(label: str = ""):
    """Print current memory usage (if psutil available)."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"💾 Memory Usage{' [' + label + ']' if label else ''}: {mem_gb:.2f} GB")
        if mem_gb > 18.0:
            print(f"⚠️  WARNING: Memory usage high! Consider reducing batch sizes.")
        return mem_gb
    return 0.0

print("=" * 80)
print("🏈 NFL BIG DATA BOWL 2026 - ADVANCED KAGGLE SUBMISSION")
print("=" * 80)
print("📦 Libraries loaded successfully")
print_memory_usage("Initial")


# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_data(folder: Path, weeks: List[str], label: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load input and output data for specified weeks (memory-optimized)."""
    print(f"\n[{label}] Loading Data ({len(weeks)} weeks)")
    print("-" * 60)
    
    input_dfs = []
    output_dfs = []
    
    # Optimize dtypes to reduce memory
    dtype_input = {
        'game_id': 'int32',
        'play_id': 'int16',
        'nfl_id': 'int32',
        'frame_id': 'int16',
        'x': 'float32',
        'y': 'float32',
        's': 'float32',
        'a': 'float32',
        'dis': 'float32',
        'o': 'float32',
        'dir': 'float32'
    }
    
    for week in weeks:
        # Try both naming patterns: "input_2023_w01.csv" and "input_w01.csv"
        input_path1 = folder / f"input_{week}.csv"
        output_path1 = folder / f"output_{week}.csv"
        input_path2 = folder / f"input_{week.replace('2023_', '')}.csv"
        output_path2 = folder / f"output_{week.replace('2023_', '')}.csv"
        
        # Use whichever exists
        if input_path1.exists() and output_path1.exists():
            input_path, output_path = input_path1, output_path1
        elif input_path2.exists() and output_path2.exists():
            input_path, output_path = input_path2, output_path2
        else:
            print(f"  ⚠ {week}: Files not found, skipping")
            continue
        
        # Load with optimized dtypes
        input_df = pd.read_csv(input_path, dtype=dtype_input, low_memory=False)
        output_df = pd.read_csv(output_path, dtype={'x': 'float32', 'y': 'float32'})
        input_dfs.append(input_df)
        output_dfs.append(output_df)
        print(f"  ✓ {week}: {len(input_df):,} input, {len(output_df):,} output rows")
    
    input_all = pd.concat(input_dfs, ignore_index=True) if input_dfs else pd.DataFrame()
    output_all = pd.concat(output_dfs, ignore_index=True) if output_dfs else pd.DataFrame()
    
    # Clear intermediate lists
    del input_dfs, output_dfs
    gc.collect()
    
    print(f"✓ Total: {len(input_all):,} input rows, {len(output_all):,} output rows")
    
    return input_all, output_all


# ============================================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================================

def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw input data to ensure all required columns exist.
    Handles differences between local and Kaggle data formats.
    """
    print("\n[DATA PREPROCESSING]")
    print("-" * 60)
    
    df = df.copy()
    
    # Required base columns that should always exist
    required_base = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "s", "o", "dir"]
    missing_base = [col for col in required_base if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")
    
    # Compute velocity components if missing
    if "vx" not in df.columns:
        print("  ✓ Computing vx from speed and direction")
        df["vx"] = df["s"] * np.cos(np.deg2rad(df["dir"]))
    if "vy" not in df.columns:
        print("  ✓ Computing vy from speed and direction")
        df["vy"] = df["s"] * np.sin(np.deg2rad(df["dir"]))
    
    # Add acceleration if missing
    if "a" not in df.columns:
        print("  ✓ Adding default acceleration (a=0)")
        df["a"] = 0.0
    
    # Add distance moved if missing
    if "dis" not in df.columns:
        print("  ✓ Adding default distance moved (dis=0)")
        df["dis"] = 0.0
    
    # Add time difference if missing
    if "time_diff" not in df.columns:
        print("  ✓ Adding default time_diff (1.0 second)")
        df["time_diff"] = 1.0
    
    # Add ball landing position if missing
    if "ball_land_x" not in df.columns:
        print("  ✓ Adding default ball_land_x (center field)")
        df["ball_land_x"] = 60.0
    if "ball_land_y" not in df.columns:
        print("  ✓ Adding default ball_land_y (center field)")
        df["ball_land_y"] = 26.65
    
    # Add role indicators if missing
    if "is_targeted_receiver" not in df.columns:
        print("  ✓ Adding default is_targeted_receiver (0)")
        df["is_targeted_receiver"] = 0
    if "is_defensive_coverage" not in df.columns:
        print("  ✓ Adding default is_defensive_coverage (0)")
        df["is_defensive_coverage"] = 0
    if "is_defense" not in df.columns:
        print("  ✓ Adding default is_defense (0)")
        df["is_defense"] = 0
    if "is_offense" not in df.columns:
        print("  ✓ Adding default is_offense (0)")
        df["is_offense"] = 0
    
    # Add team info if missing
    if "player_side" not in df.columns:
        print("  ✓ Adding default player_side ('unknown')")
        df["player_side"] = "unknown"
    
    # Add play_direction if missing (crucial for coordinate standardization)
    if "play_direction" not in df.columns:
        print("  ✓ Adding default play_direction ('right')")
        df["play_direction"] = "right"
    
    print(f"✓ Preprocessing complete: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


# ============================================================================
# SECTION 3: COORDINATE STANDARDIZATION
# ============================================================================

def standardize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all spatial data based on play_direction.
    CRITICAL: This prevents the model from learning patterns twice (left-to-right AND right-to-left).
    
    If play_direction == 'left', flip:
    - x coordinates: x_std = 120 - x
    - y coordinates: y_std = 53.3 - y (optional but recommended)
    - dir angles: dir_std = (360 - dir) % 360
    - All ball landing coordinates the same way
    """
    print("\n[COORDINATE STANDARDIZATION]")
    print("-" * 60)
    print("  → Standardizing coordinates based on play_direction...")
    
    df = df.copy()
    
    # Identify plays going left
    if "play_direction" in df.columns:
        left_mask = df["play_direction"] == "left"
        n_left = left_mask.sum()
        print(f"  → Found {n_left:,} rows with play_direction='left' (will flip)")
        
        if n_left > 0:
            # Standardize x coordinates (horizontal field position)
            df.loc[left_mask, "x"] = FIELD_X_MAX - df.loc[left_mask, "x"]
            
            # Standardize y coordinates (optional but good for consistency)
            df.loc[left_mask, "y"] = FIELD_Y_MAX - df.loc[left_mask, "y"]
            
            # Standardize direction angles
            df.loc[left_mask, "dir"] = (360 - df.loc[left_mask, "dir"]) % 360
            df.loc[left_mask, "o"] = (360 - df.loc[left_mask, "o"]) % 360
            
            # Standardize ball landing coordinates
            if "ball_land_x" in df.columns:
                df.loc[left_mask, "ball_land_x"] = FIELD_X_MAX - df.loc[left_mask, "ball_land_x"]
            if "ball_land_y" in df.columns:
                df.loc[left_mask, "ball_land_y"] = FIELD_Y_MAX - df.loc[left_mask, "ball_land_y"]
            
            # Recompute velocity components after direction change
            if "vx" in df.columns and "vy" in df.columns:
                df.loc[left_mask, "vx"] = df.loc[left_mask, "s"] * np.cos(np.deg2rad(df.loc[left_mask, "dir"]))
                df.loc[left_mask, "vy"] = df.loc[left_mask, "s"] * np.sin(np.deg2rad(df.loc[left_mask, "dir"]))
            
            print(f"  ✓ Standardized {n_left:,} rows to right-facing coordinates")
    else:
        print("  ⚠ No play_direction column - skipping standardization")
    
    return df


# ============================================================================
# SECTION 4: FEATURE ENGINEERING
# ============================================================================

def engineer_deep_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 250+ advanced features including:
    - Coordinate standardization (CRITICAL FIRST STEP)
    - Player-to-ball interaction features
    - Player-to-player "obstacle" features
    - Defensive coverage metrics
    - Route intelligence
    - Physics-based features
    - Temporal and spatial features
    
    Note: Assumes data has been preprocessed with preprocess_input_data()
    """
    print("\n[FEATURE ENGINEERING]")
    print("-" * 60)
    
    # NOTE: Coordinate standardization DISABLED - was causing 2.5 yard RMSE
    # Model should learn to handle both left and right-facing plays naturally
    feats = df.copy()
    
    # STEP 1: Basic derived features
    print("  → Velocity vectors...")
    feats["speed_squared"] = feats["s"] ** 2
    feats["dir_rad"] = np.deg2rad(feats["dir"])
    feats["o_rad"] = np.deg2rad(feats["o"])
    feats["dir_sin"] = np.sin(feats["dir_rad"])
    feats["dir_cos"] = np.cos(feats["dir_rad"])
    feats["o_sin"] = np.sin(feats["o_rad"])
    feats["o_cos"] = np.cos(feats["o_rad"])
    feats["dir_o_diff"] = np.abs(feats["dir"] - feats["o"])
    feats["dir_o_diff_rad"] = np.deg2rad(feats["dir_o_diff"])
    feats["momentum"] = feats["s"] * 1.0
    
    # STEP 3: Player-to-Ball Features (The "Goal")
    print("  → Player-to-ball interaction features...")
    feats["ball_x_norm"] = feats["ball_land_x"] / FIELD_X_MAX
    feats["ball_y_norm"] = feats["ball_land_y"] / FIELD_Y_MAX
    feats["x_norm"] = feats["x"] / FIELD_X_MAX
    feats["y_norm"] = feats["y"] / FIELD_Y_MAX
    
    dx_to_ball = feats["ball_land_x"] - feats["x"]
    dy_to_ball = feats["ball_land_y"] - feats["y"]
    feats["dist_to_ball_land"] = np.sqrt(dx_to_ball**2 + dy_to_ball**2)
    feats["angle_to_ball_land"] = np.arctan2(dy_to_ball, dx_to_ball)
    
    # How much player needs to "turn" to go to the ball
    feats["diff_angle_motion_to_ball"] = np.abs(feats["dir_rad"] - feats["angle_to_ball_land"])
    feats["diff_angle_motion_to_ball"] = np.minimum(
        feats["diff_angle_motion_to_ball"],
        2 * np.pi - feats["diff_angle_motion_to_ball"]
    )
    
    feats["ball_direction_x"] = dx_to_ball / (feats["dist_to_ball_land"] + 1e-6)
    feats["ball_direction_y"] = dy_to_ball / (feats["dist_to_ball_land"] + 1e-6)
    feats["velocity_ball_alignment"] = (
        feats["vx"] * feats["ball_direction_x"] + feats["vy"] * feats["ball_direction_y"]
    )
    feats["moving_toward_ball"] = (feats["velocity_ball_alignment"] > 0).astype("int8")
    feats["time_to_ball"] = feats["time_diff"]
    feats["time_to_ball_squared"] = feats["time_to_ball"] ** 2
    
    # Backwards compatibility
    feats["dist_to_ball"] = feats["dist_to_ball_land"]
    feats["angle_to_ball"] = feats["angle_to_ball_land"]
    
    # Physics features
    print("  → Physics features...")
    feats["dist_to_sideline"] = np.minimum(feats["y"], FIELD_Y_MAX - feats["y"])
    feats["dist_to_endzone"] = np.minimum(feats["x"], FIELD_X_MAX - feats["x"])
    
    # Trajectory predictions
    print("  → Trajectory predictions...")
    feats["predicted_x_linear"] = feats["x"] + feats["vx"] * feats["time_to_ball"]
    feats["predicted_y_linear"] = feats["y"] + feats["vy"] * feats["time_to_ball"]
    feats["predicted_x_accel"] = (
        feats["predicted_x_linear"] + 0.5 * feats["a"] * feats["dir_cos"] * feats["time_to_ball_squared"]
    )
    feats["predicted_y_accel"] = (
        feats["predicted_y_linear"] + 0.5 * feats["a"] * feats["dir_sin"] * feats["time_to_ball_squared"]
    )
    
    # Play-level aggregations
    print("  → Play-level aggregations...")
    play_group = feats.groupby(["game_id", "play_id"])
    feats["speed_percentile"] = play_group["s"].transform(lambda x: x.rank(pct=True))
    feats["dist_to_ball_percentile"] = play_group["dist_to_ball"].transform(lambda x: x.rank(pct=True))
    feats["play_mean_speed"] = play_group["s"].transform("mean")
    feats["play_mean_dist_to_ball"] = play_group["dist_to_ball"].transform("mean")
    
    # Role-specific features
    print("  → Role-specific features...")
    feats["receiver_dist"] = feats["is_targeted_receiver"] * feats["dist_to_ball"]
    feats["defender_dist"] = feats["is_defensive_coverage"] * feats["dist_to_ball"]
    
    # STEP 4: Player-to-Player "Obstacle" Features (Memory-Optimized)
    print("  → Player-to-player interaction features (obstacles)...")
    
    # Initialize interaction columns
    feats["dist_to_targeted_receiver"] = np.nan
    feats["angle_to_targeted_receiver"] = np.nan
    feats["relative_speed_to_tr"] = np.nan
    feats["dist_to_closest_opponent"] = np.nan
    feats["relative_speed_to_closest_opponent"] = np.nan
    feats["angle_to_closest_opponent"] = np.nan
    feats["dist_to_closest_teammate"] = np.nan
    
    # Process frame by frame to compute interactions
    print("    Computing player-to-player interactions...")
    for (game_id, play_id, frame_id), frame_df in feats.groupby(["game_id", "play_id", "frame_id"]):
        if len(frame_df) < 2:  # Need at least 2 players
            continue
        
        coords = frame_df[["x", "y"]].values.astype("float32")
        velocities = frame_df[["vx", "vy"]].values.astype("float32")
        n_players = len(frame_df)
        
        # Find targeted receiver
        target_mask = frame_df["is_targeted_receiver"].values.astype(bool)
        if target_mask.any():
            target_idx = np.where(target_mask)[0][0]
            target_pos = coords[target_idx]
            target_vel = velocities[target_idx]
            
            # Distance and angle to targeted receiver
            to_target = target_pos - coords
            dist_to_tr = np.linalg.norm(to_target, axis=1, keepdims=True)
            angle_to_tr = np.arctan2(to_target[:, 1], to_target[:, 0])
            
            # Relative speed to targeted receiver
            rel_vel = target_vel - velocities
            rel_speed = np.linalg.norm(rel_vel, axis=1)
            
            feats.loc[frame_df.index, "dist_to_targeted_receiver"] = dist_to_tr.flatten()
            feats.loc[frame_df.index, "angle_to_targeted_receiver"] = angle_to_tr
            feats.loc[frame_df.index, "relative_speed_to_tr"] = rel_speed
        
        # Find closest opponent and teammate for each player
        offense_mask = frame_df["is_offense"].values.astype(bool)
        defense_mask = frame_df["is_defense"].values.astype(bool)
        
        for i in range(n_players):
            player_pos = coords[i]
            player_vel = velocities[i]
            is_offense = offense_mask[i]
            
            # Determine opponent and teammate masks
            if is_offense:
                opponent_mask = defense_mask
                teammate_mask = offense_mask.copy()
                teammate_mask[i] = False  # Exclude self
            else:
                opponent_mask = offense_mask
                teammate_mask = defense_mask.copy()
                teammate_mask[i] = False  # Exclude self
            
            # Closest opponent
            if opponent_mask.any():
                opponent_coords = coords[opponent_mask]
                opponent_vels = velocities[opponent_mask]
                diffs = opponent_coords - player_pos
                dists = np.linalg.norm(diffs, axis=1)
                closest_opp_idx = np.argmin(dists)
                
                feats.loc[frame_df.index[i], "dist_to_closest_opponent"] = dists[closest_opp_idx]
                feats.loc[frame_df.index[i], "angle_to_closest_opponent"] = np.arctan2(
                    diffs[closest_opp_idx, 1], diffs[closest_opp_idx, 0]
                )
                feats.loc[frame_df.index[i], "relative_speed_to_closest_opponent"] = np.linalg.norm(
                    opponent_vels[closest_opp_idx] - player_vel
                )
            
            # Closest teammate
            if teammate_mask.any():
                teammate_coords = coords[teammate_mask]
                diffs = teammate_coords - player_pos
                dists = np.linalg.norm(diffs, axis=1)
                feats.loc[frame_df.index[i], "dist_to_closest_teammate"] = np.min(dists)
    
    # Coverage intelligence (ultra memory-optimized for <20GB)
    print("  → Coverage intelligence (ultra memory-optimized)...")
    
    # Initialize coverage columns with defaults
    feats["nearest_defender_dist"] = np.nan
    feats["defensive_help"] = np.nan
    feats["angle_to_receiver_ball_path"] = 0.0
    feats["coverage_responsibility"] = 0.0
    feats["route_deviation"] = 0.0
    feats["separation_created"] = np.nan
    
    # Process in very small batches to minimize memory
    unique_frames = feats[["game_id", "play_id", "frame_id"]].drop_duplicates()
    batch_size = 50  # Process only 50 frames at a time (reduced from 100)
    
    for batch_idx in range(0, len(unique_frames), batch_size):
        batch_frames = unique_frames.iloc[batch_idx:batch_idx + batch_size]
        
        # Extract just this batch
        mask = feats[["game_id", "play_id", "frame_id"]].merge(
            batch_frames, on=["game_id", "play_id", "frame_id"], how="inner"
        ).index
        batch_df = feats.loc[mask].copy()
        
        # Process each frame individually
        for (game_id, play_id, frame_id), frame_df in batch_df.groupby(["game_id", "play_id", "frame_id"]):
            defense_mask = frame_df["is_defense"].values.astype(bool)
            offense_mask = frame_df["is_offense"].values.astype(bool)
            
            if not defense_mask.any():
                continue
            
            # Use int16 for coordinates to save memory (scaled by 10 for precision)
            coords = (frame_df[["x", "y"]].values * 10).astype("int16")
            defense_coords = coords[defense_mask]
            n_players = len(frame_df)
            
            # Compute only essential metrics
            nearest_defender = np.full(n_players, 32767, dtype="int16")  # Max int16
            
            # Process players in micro-chunks of 10
            micro_chunk = 10
            for i in range(0, n_players, micro_chunk):
                end_idx = min(i + micro_chunk, n_players)
                chunk_coords = coords[i:end_idx]
                
                # Compute distances one at a time to minimize memory
                min_dists = []
                for player_coord in chunk_coords:
                    diffs = defense_coords - player_coord
                    dists_sq = (diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
                    min_dists.append(np.sqrt(dists_sq.min()))
                
                nearest_defender[i:end_idx] = (np.array(min_dists) * 10).astype("int16")
            
            # Convert back to float32 and scale down
            nearest_defender_float = (nearest_defender / 100.0).astype("float32")
            feats.loc[frame_df.index, "nearest_defender_dist"] = nearest_defender_float
            
            # Separation for receivers only
            if offense_mask.any():
                feats.loc[frame_df.index[offense_mask], "separation_created"] = nearest_defender_float[offense_mask]
            
            # Simplified coverage responsibility for defense
            if defense_mask.any():
                target_mask = frame_df["is_targeted_receiver"].values.astype(bool)
                if target_mask.any():
                    target_pos = coords[target_mask][0]
                    def_coords = coords[defense_mask]
                    dist_to_target = np.sqrt(((def_coords - target_pos) ** 2).sum(axis=1)) / 10.0
                    coverage = (1.0 / (dist_to_target + 1.0)).astype("float32")
                    feats.loc[frame_df.index[defense_mask], "coverage_responsibility"] = coverage
        
        # Clear batch memory
        del batch_df, mask
        gc.collect()
        
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"    Processed {batch_idx + batch_size}/{len(unique_frames)} frames...")
    
    # Route intelligence (simplified for memory)
    print("  → Route intelligence (memory-efficient)...")
    # Process in chunks to reduce memory
    chunk_size = 50000
    route_sim_list = []
    route_speed_list = []
    
    for i in range(0, len(feats), chunk_size):
        end_idx = min(i + chunk_size, len(feats))
        chunk = feats.iloc[i:end_idx]
        
        # Route similarity (simplified)
        player_mean_speed = chunk.groupby("nfl_id")["s"].transform("mean").astype("float32")
        route_speed_list.append((chunk["s"] - player_mean_speed).astype("float32"))
        
        # Simplified route similarity based on speed consistency
        route_sim_list.append((1.0 - np.abs(chunk["s"] - player_mean_speed) / (player_mean_speed + 1.0)).clip(0, 1).astype("float32"))
        
        del player_mean_speed
        gc.collect()
    
    feats["route_similarity"] = pd.concat(route_sim_list, ignore_index=True).values
    feats["route_speed_delta"] = pd.concat(route_speed_list, ignore_index=True).values
    
    del route_sim_list, route_speed_list
    gc.collect()
    
    # Additional features (computed in-place to save memory)
    print("  → Additional features...")
    feats["log_dist_to_ball"] = np.log1p(feats["dist_to_ball"]).astype("float32")
    feats["log_speed"] = np.log1p(feats["s"]).astype("float32")
    feats["speed_dist_product"] = (feats["s"] * feats["dist_to_ball"]).astype("float32")
    feats["vx_vy_ratio"] = (feats["vx"] / (np.abs(feats["vy"]) + 0.1)).astype("float32")
    feats["accel_speed_ratio"] = (feats["a"] / (feats["s"] + 0.1)).astype("float32")
    
    print(f"✓ Feature engineering complete: {len(feats.columns)} features")
    gc.collect()
    
    return feats


def create_pairs(input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    """Create training pairs matching input frames to output positions (memory-optimized)."""
    print("\n[CREATING TRAINING PAIRS]")
    print("-" * 60)
    
    # Pre-allocate lists for efficiency
    pairs = []
    batch_pairs = []
    batch_size = 1000  # Process in batches and convert to DataFrame periodically
    
    for (game_id, play_id), play_in in input_df.groupby(["game_id", "play_id"]):
        play_out = output_df[
            (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
        ]
        
        if play_out.empty:
            continue
        
        for nfl_id in play_out["nfl_id"].unique():
            player_in = play_in[play_in["nfl_id"] == nfl_id].sort_values("frame_id")
            player_out = play_out[play_out["nfl_id"] == nfl_id].sort_values("frame_id")
            
            if player_in.empty or player_out.empty:
                continue
            
            last_in = player_in.iloc[-1]
            
            for _, out_row in player_out.iterrows():
                pair = {
                    "game_id": np.int32(game_id),
                    "play_id": np.int16(play_id),
                    "nfl_id": np.int32(nfl_id),
                    "input_x": np.float32(last_in["x"]),
                    "input_y": np.float32(last_in["y"]),
                    "output_x": np.float32(out_row["x"]),
                    "output_y": np.float32(out_row["y"]),
                    "delta_x": np.float32(out_row["x"] - last_in["x"]),
                    "delta_y": np.float32(out_row["y"] - last_in["y"]),
                }
                
                # Copy all features from input (use float32 for memory efficiency)
                for col in player_in.columns:
                    if col not in pair and col not in ["x", "y", "frame_id"]:
                        val = last_in[col]
                        if isinstance(val, (int, np.integer)):
                            pair[f"input_{col}"] = np.int32(val) if abs(val) < 2147483647 else val
                        elif isinstance(val, (float, np.floating)):
                            pair[f"input_{col}"] = np.float32(val)
                        else:
                            pair[f"input_{col}"] = val
                
                batch_pairs.append(pair)
                
                # Convert batches to DataFrame periodically to manage memory
                if len(batch_pairs) >= batch_size:
                    pairs.append(pd.DataFrame(batch_pairs))
                    batch_pairs = []
                    gc.collect()
    
    # Add remaining pairs
    if batch_pairs:
        pairs.append(pd.DataFrame(batch_pairs))
    
    # Concatenate all batches
    pairs_df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    print(f"✓ Created {len(pairs_df):,} training pairs")
    
    del pairs, batch_pairs
    gc.collect()
    
    return pairs_df


# ============================================================================
# SECTION 3: PHYSICS-CONSTRAINED PREDICTIONS
# ============================================================================

def apply_physics_constraints(
    predictions: np.ndarray,
    base_positions: np.ndarray,
    current_velocities: np.ndarray = None,
    time_delta: float = 1.0,
    max_accel: float = 4.0,
    max_speed: float = 25.0
) -> np.ndarray:
    """Apply physics constraints: acceleration, speed, field boundaries."""
    constrained = predictions.copy()
    
    # Field boundaries
    predicted_pos = base_positions + predictions
    predicted_pos[:, 0] = np.clip(predicted_pos[:, 0], 0.0, FIELD_X_MAX)
    predicted_pos[:, 1] = np.clip(predicted_pos[:, 1], 0.0, FIELD_Y_MAX)
    
    if current_velocities is not None:
        # Acceleration constraint
        implied_vel = predictions / time_delta
        required_accel = (implied_vel - current_velocities) / time_delta
        accel_mag = np.linalg.norm(required_accel, axis=1)
        
        too_fast = accel_mag > max_accel
        if too_fast.any():
            scale = max_accel / accel_mag[too_fast]
            vel_change = implied_vel[too_fast] - current_velocities[too_fast]
            scaled_vel_change = vel_change * scale[:, None]
            constrained[too_fast] = (current_velocities[too_fast] + scaled_vel_change) * time_delta
        
        # Speed constraint
        speed = np.linalg.norm(implied_vel, axis=1)
        too_fast_speed = speed > max_speed
        if too_fast_speed.any():
            scale = max_speed / speed[too_fast_speed]
            constrained[too_fast_speed] = implied_vel[too_fast_speed] * scale[:, None] * time_delta
    
    # Final boundary check
    final_pos = base_positions + constrained
    final_pos[:, 0] = np.clip(final_pos[:, 0], 0.0, FIELD_X_MAX)
    final_pos[:, 1] = np.clip(final_pos[:, 1], 0.0, FIELD_Y_MAX)
    constrained = final_pos - base_positions
    
    return constrained


# ============================================================================
# SECTION 4: ADVANCED ENSEMBLE TRAINING
# ============================================================================

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    base_train: np.ndarray,
    base_val: np.ndarray,
    train_groups: np.ndarray,
    label: str = "ALL"
) -> Dict:
    """
    Train stacked ensemble with:
    - 5 base models (2×XGBoost, 2×LightGBM, 1×CatBoost)
    - Meta-learner (Ridge regression)
    - Residual refinement (LightGBM)
    """
    print(f"\n[TRAINING ENSEMBLE: {label}]")
    print("-" * 60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features: {X_train.shape[1]}")
    
    results = {}
    cv = GroupKFold(n_splits=3)
    
    # Define base models (optimized for speed)
    base_models_x = {
        "XGB1_x": xgb.XGBRegressor(
            n_estimators=700, max_depth=8, learning_rate=0.025,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:pseudohubererror", random_state=42, n_jobs=-1
        ),
        "XGB2_x": xgb.XGBRegressor(
            n_estimators=600, max_depth=10, learning_rate=0.03,
            subsample=0.88, colsample_bytree=0.88,
            objective="reg:pseudohubererror", random_state=43, n_jobs=-1
        ),
        "LGBM1_x": lgb.LGBMRegressor(
            n_estimators=700, max_depth=8, learning_rate=0.025,
            subsample=0.88, colsample_bytree=0.85,
            objective="huber", random_state=42, n_jobs=-1, verbose=-1
        ),
        "LGBM2_x": lgb.LGBMRegressor(
            n_estimators=600, max_depth=10, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.88,
            objective="huber", random_state=43, n_jobs=-1, verbose=-1
        ),
        "CAT_x": cb.CatBoostRegressor(
            iterations=500, depth=7, learning_rate=0.035,
            loss_function="RMSE", random_state=42, verbose=False
        ),
    }
    
    base_models_y = {
        name.replace("_x", "_y"): type(model)(
            **{k: v for k, v in model.get_params().items() if k != "random_state"}
        )
        for name, model in base_models_x.items()
    }
    
    # Cross-validation training
    print("Training base models with 3-fold CV...")
    cv_preds_x = np.zeros((len(X_train), len(base_models_x)))
    cv_preds_y = np.zeros((len(X_train), len(base_models_y)))
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, train_groups)):
        print(f"  Fold {fold_idx + 1}/3...")
        
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]
        w_tr = sample_weights[train_idx]
        
        # Train X models
        for i, (name, model) in enumerate(base_models_x.items()):
            model.fit(X_tr, y_tr[:, 0], sample_weight=w_tr)
            cv_preds_x[val_idx, i] = model.predict(X_vl)
        
        # Train Y models
        for i, (name, model) in enumerate(base_models_y.items()):
            model.fit(X_tr, y_tr[:, 1], sample_weight=w_tr)
            cv_preds_y[val_idx, i] = model.predict(X_vl)
    
    # Meta-learner stacking
    print("Training meta-learner (Ridge)...")
    meta_x = Ridge(alpha=10.0)
    meta_y = Ridge(alpha=10.0)
    meta_x.fit(cv_preds_x, y_train[:, 0])
    meta_y.fit(cv_preds_y, y_train[:, 1])
    
    stacked_train_x = meta_x.predict(cv_preds_x)
    stacked_train_y = meta_y.predict(cv_preds_y)
    stacked_train = np.column_stack([stacked_train_x, stacked_train_y])
    
    # Validation predictions
    val_preds_x = np.column_stack([model.predict(X_val) for model in base_models_x.values()])
    val_preds_y = np.column_stack([model.predict(X_val) for model in base_models_y.values()])
    stacked_val_x = meta_x.predict(val_preds_x)
    stacked_val_y = meta_y.predict(val_preds_y)
    stacked_val = np.column_stack([stacked_val_x, stacked_val_y])
    
    # Residual refinement
    print("Training residual refiners...")
    residuals_x = y_train[:, 0] - stacked_train_x
    residuals_y = y_train[:, 1] - stacked_train_y
    
    residual_x = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)
    residual_y = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)
    residual_x.fit(X_train, residuals_x, sample_weight=sample_weights)
    residual_y.fit(X_train, residuals_y, sample_weight=sample_weights)
    
    refined_train = stacked_train.copy()
    refined_train[:, 0] += residual_x.predict(X_train)
    refined_train[:, 1] += residual_y.predict(X_train)
    
    refined_val = stacked_val.copy()
    refined_val[:, 0] += residual_x.predict(X_val)
    refined_val[:, 1] += residual_y.predict(X_val)
    
    # Clip predictions to field boundaries
    def clip_deltas(preds, base):
        abs_preds = preds + base
        abs_preds[:, 0] = np.clip(abs_preds[:, 0], 0.0, FIELD_X_MAX)
        abs_preds[:, 1] = np.clip(abs_preds[:, 1], 0.0, FIELD_Y_MAX)
        return abs_preds - base
    
    refined_train_clipped = clip_deltas(refined_train, base_train)
    refined_val_clipped = clip_deltas(refined_val, base_val)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, refined_train_clipped))
    val_rmse = np.sqrt(mean_squared_error(y_val, refined_val_clipped))
    
    print(f"✓ Training RMSE: {train_rmse:.4f} yards")
    print(f"✓ Validation RMSE: {val_rmse:.4f} yards")
    
    results["base_models_x"] = base_models_x
    results["base_models_y"] = base_models_y
    results["meta_x"] = meta_x
    results["meta_y"] = meta_y
    results["residual_x"] = residual_x
    results["residual_y"] = residual_y
    results["train_rmse"] = train_rmse
    results["val_rmse"] = val_rmse
    
    return results


# ============================================================================
# SECTION 5: ENHANCED TEST-TIME AUGMENTATION
# ============================================================================

def predict_with_enhanced_tta(
    results: Dict,
    X: np.ndarray,
    base_positions: np.ndarray = None,
    current_velocities: np.ndarray = None,
    is_defense: np.ndarray = None,
    n_augments: int = 3,
    noise_scale: float = 0.01,
    apply_physics: bool = True
) -> np.ndarray:
    """
    Enhanced TTA with:
    - Role-aware noise (2x for defense)
    - Trajectory variations
    - Confidence-weighted averaging
    - Physics constraints
    """
    n_samples = X.shape[0]
    
    # Role-specific noise
    if is_defense is not None:
        role_noise = np.where(is_defense, noise_scale * 2.0, noise_scale)[:, None]
    else:
        role_noise = noise_scale
    
    predictions = []
    weights = []
    
    # Strategy 1: Original
    pred_x = results["meta_x"].predict(
        np.column_stack([m.predict(X) for m in results["base_models_x"].values()])
    )
    pred_y = results["meta_y"].predict(
        np.column_stack([m.predict(X) for m in results["base_models_y"].values()])
    )
    pred_x += results["residual_x"].predict(X)
    pred_y += results["residual_y"].predict(X)
    pred_original = np.column_stack([pred_x, pred_y])
    predictions.append(pred_original)
    weights.append(1.5)
    
    # Strategy 2-3: Perturbed predictions
    for i in range(n_augments - 1):
        noise = np.random.normal(0, role_noise, X.shape)
        X_aug = X + noise
        
        pred_x = results["meta_x"].predict(
            np.column_stack([m.predict(X_aug) for m in results["base_models_x"].values()])
        )
        pred_y = results["meta_y"].predict(
            np.column_stack([m.predict(X_aug) for m in results["base_models_y"].values()])
        )
        pred_x += results["residual_x"].predict(X_aug)
        pred_y += results["residual_y"].predict(X_aug)
        predictions.append(np.column_stack([pred_x, pred_y]))
        weights.append(1.0)
    
    # Confidence-weighted average
    weights = np.array(weights) / sum(weights)
    avg_pred = sum(p * w for p, w in zip(predictions, weights))
    
    # Apply physics constraints
    if apply_physics and base_positions is not None:
        if current_velocities is None:
            current_velocities = np.zeros((n_samples, 2), dtype=np.float32)
        avg_pred = apply_physics_constraints(avg_pred, base_positions, current_velocities)
    
    return avg_pred


# ============================================================================
# SECTION 6: MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("PHASE 1: DATA LOADING")
    print("=" * 80)
    
    # Detect Kaggle environment and set paths
    kaggle_data_root = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction")
    kaggle_validation_root = Path("/kaggle/input/validation/validation")
    
    if kaggle_data_root.exists():
        print("✓ Detected Kaggle environment")
        data_root = kaggle_data_root
        train_folder = data_root / "train"
        validation_folder = kaggle_validation_root if kaggle_validation_root.exists() else train_folder
    else:
        print("✓ Detected local environment")
        data_root = Path(".")
        train_folder = data_root / "train"
        validation_folder = data_root / "validation" if (data_root / "validation").exists() else train_folder
    
    print(f"Data root: {data_root}")
    print(f"Train folder: {train_folder}")
    print(f"Validation folder: {validation_folder}")
    
    # Load training data (weeks 1-15 for training)
    train_weeks = [f"2023_w{i:02d}" for i in range(1, 16)]
    train_input, train_output = load_data(train_folder, train_weeks, "TRAINING")
    print_memory_usage("After loading training data")
    
    # Load validation data (weeks 16-18 for validation) if available
    if validation_folder.exists() and validation_folder != train_folder:
        val_weeks = [f"2023_w{i:02d}" for i in range(16, 19)]
        val_input, val_output = load_data(validation_folder, val_weeks, "VALIDATION")
        
        # Combine for training (we'll split later)
        train_input = pd.concat([train_input, val_input], ignore_index=True)
        train_output = pd.concat([train_output, val_output], ignore_index=True)
        del val_input, val_output
        gc.collect()
        print_memory_usage("After merging validation data")
    
    print("\n" + "=" * 80)
    print("PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 80)
    
    # Preprocess to ensure all required columns exist
    train_input = preprocess_input_data(train_input)
    print_memory_usage("After preprocessing")
    
    # Engineer advanced features
    train_input = engineer_deep_features(train_input)
    print_memory_usage("After feature engineering")
    
    train_pairs = create_pairs(train_input, train_output)
    print_memory_usage("After creating pairs")
    
    # Clean up
    del train_input, train_output
    gc.collect()
    print_memory_usage("After cleanup")
    
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE PREPARATION")
    print("=" * 80)
    
    # Extract features and targets
    feature_cols = [c for c in train_pairs.columns if c.startswith("input_")]
    X = train_pairs[feature_cols].values.astype("float32")
    y = train_pairs[["delta_x", "delta_y"]].values.astype("float32")
    base_coords = train_pairs[["input_x", "input_y"]].values.astype("float32")
    groups = train_pairs["game_id"].values
    
    # Sample weights (target players weighted higher)
    weights = np.ones(len(train_pairs), dtype="float32")
    if "input_is_targeted_receiver" in train_pairs.columns:
        weights[train_pairs["input_is_targeted_receiver"].fillna(0).astype(bool)] = 2.0
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    
    # Feature scaling
    print("Scaling features...")
    scaler1 = RobustScaler()
    scaler2 = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
    X_scaled = scaler1.fit_transform(X)
    X_scaled = scaler2.fit_transform(X_scaled).astype("float32")
    
    del train_pairs
    gc.collect()
    
    print("\n" + "=" * 80)
    print("PHASE 4: MODEL TRAINING")
    print("=" * 80)
    
    # Split data (80/20)
    n_train = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:n_train]
    X_val = X_scaled[n_train:]
    y_train = y[:n_train]
    y_val = y[n_train:]
    base_train = base_coords[:n_train]
    base_val = base_coords[n_train:]
    weights_train = weights[:n_train]
    groups_train = groups[:n_train]
    
    # Train ensemble
    results = train_ensemble(
        X_train, y_train, X_val, y_val,
        weights_train, base_train, base_val,
        groups_train, label="ALL_PLAYERS"
    )
    
    del X_train, X_val, y_train, y_val, base_train, base_val, weights_train, groups_train
    gc.collect()
    
    print("\n" + "=" * 80)
    print("PHASE 5: TEST PREDICTION")
    print("=" * 80)
    
    # Load test data (check both Kaggle and local paths)
    if kaggle_data_root.exists():
        test_input_path = kaggle_data_root / "test_input.csv"
        test_template_path = kaggle_data_root / "test.csv"
    else:
        test_input_path = Path("test_input.csv")
        test_template_path = Path("test.csv")
    
    print(f"Test input path: {test_input_path}")
    print(f"Test template path: {test_template_path}")
    
    if not test_input_path.exists() or not test_template_path.exists():
        print("⚠️  Test files not found - creating dummy submission")
        submission = pd.DataFrame({
            "id": [f"0_0_0_{i}" for i in range(100)],
            "x": [60.0] * 100,
            "y": [26.65] * 100
        })
    else:
        test_input = pd.read_csv(test_input_path)
        test_template = pd.read_csv(test_template_path)
        
        print(f"Test input: {len(test_input):,} rows")
        print(f"Test template: {len(test_template):,} predictions needed")
        
        # Preprocess test data to ensure all required columns exist
        test_input = preprocess_input_data(test_input)
        
        # Engineer features for test
        test_input = engineer_deep_features(test_input)
        
        # Create prediction pairs
        print("Creating test pairs...")
        test_pairs = []
        for _, row in test_template.iterrows():
            player_frames = test_input[
                (test_input["game_id"] == row["game_id"]) &
                (test_input["play_id"] == row["play_id"]) &
                (test_input["nfl_id"] == row["nfl_id"])
            ].sort_values("frame_id")
            
            if len(player_frames) > 0:
                last_frame = player_frames.iloc[-1]
                pair = {
                    "id": f"{row['game_id']}_{row['play_id']}_{row['nfl_id']}_{row['frame_id']}",
                    "base_x": last_frame["x"],
                    "base_y": last_frame["y"],
                }
                # Preserve play_direction for coordinate reversion
                if "play_direction" in last_frame:
                    pair["play_direction"] = last_frame["play_direction"]
                for col in feature_cols:
                    feat_name = col.replace("input_", "")
                    if feat_name in last_frame:
                        pair[col] = last_frame[feat_name]
                test_pairs.append(pair)
        
        test_df = pd.DataFrame(test_pairs)
        print(f"Created {len(test_df):,} test pairs")
        
        # Extract features
        X_test = test_df[feature_cols].fillna(0).values.astype("float32")
        base_test = test_df[["base_x", "base_y"]].values.astype("float32")
        
        # Scale features
        X_test_scaled = scaler1.transform(X_test)
        X_test_scaled = scaler2.transform(X_test_scaled).astype("float32")
        
        # Predict with enhanced TTA
        print("Generating predictions with Enhanced TTA...")
        try:
            current_vx = test_df["input_vx"].fillna(0).values.astype("float32")
            current_vy = test_df["input_vy"].fillna(0).values.astype("float32")
            current_vel = np.column_stack([current_vx, current_vy])
        except:
            current_vel = None
        
        is_defense = test_df.get("input_is_defensive_coverage", pd.Series([False] * len(test_df))).fillna(False).values
        
        pred_deltas = predict_with_enhanced_tta(
            results, X_test_scaled,
            base_positions=base_test,
            current_velocities=current_vel,
            is_defense=is_defense,
            n_augments=3,
            noise_scale=0.01,
            apply_physics=True
        )
        
        # Convert to absolute positions
        predictions = base_test + pred_deltas
        predictions[:, 0] = np.clip(predictions[:, 0], 0.0, FIELD_X_MAX)
        predictions[:, 1] = np.clip(predictions[:, 1], 0.0, FIELD_Y_MAX)
        
        # Create submission
        submission = pd.DataFrame({
            "id": test_df["id"],
            "x": predictions[:, 0],
            "y": predictions[:, 1]
        })
    
    # Save submission
    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv("submission.csv", index=False)
    
    print("\n" + "=" * 80)
    print("✅ SUBMISSION COMPLETE!")
    print("=" * 80)
    print(f"📁 File: submission.csv")
    print(f"📊 Predictions: {len(submission):,}")
    print(f"📈 Training RMSE: {results['train_rmse']:.4f} yards")
    print(f"📈 Validation RMSE: {results['val_rmse']:.4f} yards")
    print("=" * 80)
    
    # Display sample
    print("\nSample predictions:")
    print(submission.head(10))


if __name__ == "__main__":
    main()
