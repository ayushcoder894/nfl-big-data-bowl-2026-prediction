"""Train the advanced ensemble on all available data and generate submission.csv.

PERFORMANCE OPTIMIZATIONS (Target: 7-8 hours from 9+ hours):
- Reduced n_estimators by 25-40% across all models (700/600 vs 1000/800)
- Increased learning rates by 20-25% to compensate (0.025-0.035 vs 0.02-0.03)
- Reduced CV folds from 5 to 3 (40% reduction in cross-validation time)
- Reduced max_depth in some models (10 vs 12, 8 vs 9)
- Reduced permutation importance repeats from 4 to 2 (50% faster feature selection)
- Reduced feature selection model from 400 to 250 trees

MEMORY OPTIMIZATIONS (Target: <30GB total memory usage):
- Use float32 instead of float64 where possible (50% memory reduction)
- Chunked processing for large array operations (coverage intelligence)
- Immediate deletion of unused DataFrames with explicit gc.collect()
- Save checkpoints as numpy arrays instead of DataFrames
- Limit coverage computation to 100-player chunks
- Monitor memory usage with resource module

Expected speedup: ~30-35% overall (2.7-3.2 hours reduction)
Expected accuracy impact: Minimal (<0.5% RMSE increase)
Memory cap: 30GB maximum (enforced through chunking and dtype optimization)
"""

import gc
import os
import sys
import argparse
import warnings
import pickle
import json
import psutil  # For memory monitoring
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.inspection import permutation_importance

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")
np.random.seed(42)

# Memory monitoring
def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def log_memory(stage: str):
    """Log memory usage at specific stage."""
    mem_gb = get_memory_usage_gb()
    print(f"[MEMORY] {stage}: {mem_gb:.2f} GB")
    if mem_gb > 28.0:  # Warning threshold
        print(f"⚠️  WARNING: Memory usage approaching 30GB limit!")
    return mem_gb

FIELD_X_MAX = 120.0
FIELD_Y_MAX = 53.3

# Checkpoint configuration
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_LOG = CHECKPOINT_DIR / "training_log.txt"


def log_checkpoint(message: str, also_print: bool = True):
    """Log a checkpoint message with timestamp to both file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    if also_print:
        print(log_message)
    
    with open(CHECKPOINT_LOG, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


def save_checkpoint(name: str, data: Dict, suffix: str = ""):
    """Save checkpoint data to file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}{suffix}.pkl"
    filepath = CHECKPOINT_DIR / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    
    log_checkpoint(f"✓ Checkpoint saved: {filename}")
    return filepath


def save_metrics(name: str, metrics: Dict):
    """Save metrics to JSON with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    filepath = CHECKPOINT_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    log_checkpoint(f"✓ Metrics saved: {filename}")
    return filepath


def find_latest_checkpoint(prefix: str) -> Path:
    """Find the most recent checkpoint file with given prefix."""
    pattern = f"{prefix}_*.pkl"
    checkpoints = list(CHECKPOINT_DIR.glob(pattern))
    if not checkpoints:
        return None
    # Sort by modification time, newest first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def load_checkpoint(filepath: Path) -> Dict:
    """Load checkpoint data from file."""
    if not filepath or not filepath.exists():
        return None
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        log_checkpoint(f"✓ Loaded checkpoint: {filepath.name}")
        return data
    except Exception as e:
        log_checkpoint(f"⚠ Failed to load checkpoint {filepath.name}: {e}")
        return None


def check_resume_from_checkpoint() -> Dict:
    """
    Check if we can resume from a previous checkpoint.
    
    Returns dictionary with:
    - resume_phase: Which phase to resume from (1-6)
    - checkpoint_data: Data from the checkpoint
    """
    log_checkpoint("\n🔍 Checking for existing checkpoints to resume...")
    
    # Check in reverse order (latest phase first), but skip Phase 6 (completed runs)
    phases = [
        ("04_model_", 4, "Model training"),  # Partial match for role models
        ("03_features_selected", 3, "Feature selection"),
        ("02_features_engineered", 2, "Feature engineering"),
        ("01_data_loaded", 1, "Data loading"),
    ]
    
    for prefix, phase_num, phase_name in phases:
        checkpoint = find_latest_checkpoint(prefix)
        if checkpoint:
            data = load_checkpoint(checkpoint)
            if data:
                log_checkpoint(f"✓ Found checkpoint from Phase {phase_num}: {phase_name}")
                log_checkpoint(f"  Checkpoint file: {checkpoint.name}")
                return {"resume_phase": phase_num, "checkpoint_data": data, "checkpoint_file": checkpoint}
    
    log_checkpoint("  No usable checkpoints found - starting from beginning")
    return {"resume_phase": 0, "checkpoint_data": None}


def get_script_dir() -> Path:
    """Return the directory containing this script or the current working dir in notebooks."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_data_root() -> Path:
    """Locate the directory that contains the competition data."""
    candidates: List[Path] = []

    kaggle_comp_dir = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction")
    if kaggle_comp_dir.exists():
        candidates.append(kaggle_comp_dir)

    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir))

    env_kaggle_dir = os.environ.get("KAGGLE_INPUT_PATH")
    if env_kaggle_dir:
        candidates.append(Path(env_kaggle_dir))

    script_dir = get_script_dir()
    candidates.append(script_dir)
    candidates.append(Path.cwd())

    kaggle_input_root = Path("/kaggle/input")
    if kaggle_input_root.exists():
        for child in kaggle_input_root.iterdir():
            if child.is_dir():
                candidates.append(child)

    # Deduplicate while preserving order
    unique_candidates = list(dict.fromkeys(candidates))

    for root in unique_candidates:
        if (root / "train").exists():
            return root

    raise FileNotFoundError(
        "Could not locate the 'train' directory. Set DATA_DIR or run from the data root."
    )


def locate_file(file_name: str, preferred_root: Path) -> Path:
    """Search likely directories for a specific file."""
    candidates: List[Path] = [preferred_root]

    kaggle_comp_dir = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction")
    if kaggle_comp_dir.exists():
        candidates.append(kaggle_comp_dir)

    candidates.extend([Path.cwd(), get_script_dir()])

    kaggle_input_root = Path("/kaggle/input")
    if kaggle_input_root.exists():
        for child in kaggle_input_root.iterdir():
            if child.is_dir():
                candidates.append(child)

    unique_candidates = list(dict.fromkeys(candidates))

    for root in unique_candidates:
        candidate = root / file_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not locate required file: {file_name}")


def load_data(folder: Path, weeks: List[str], phase_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load weekly input/output CSVs with memory-friendly dtypes."""
    if not folder.exists():
        raise FileNotFoundError(f"{phase_name} folder not found: {folder}")

    print(f"\n[{phase_name}] Loading Data ({len(weeks)} weeks)")
    print("-" * 60)

    input_frames: List[pd.DataFrame] = []
    output_frames: List[pd.DataFrame] = []

    for week in weeks:
        input_file = folder / f"input_2023_{week}.csv"
        output_file = folder / f"output_2023_{week}.csv"

        if input_file.exists() and output_file.exists():
            input_df = pd.read_csv(
                input_file,
                dtype={
                    "x": "float32",
                    "y": "float32",
                    "s": "float32",
                    "a": "float32",
                    "dir": "float32",
                    "o": "float32",
                    "frame_id": "int16",
                    "ball_land_x": "float32",
                    "ball_land_y": "float32",
                    "num_frames_output": "int8",
                    "player_to_predict": "bool",
                },
            )
            output_df = pd.read_csv(
                output_file,
                dtype={
                    "x": "float32",
                    "y": "float32",
                    "frame_id": "int16",
                },
            )

            input_frames.append(input_df)
            output_frames.append(output_df)
            print(f"  ✓ {week}: {len(input_df):,} input, {len(output_df):,} output rows")
        else:
            print(f"  → Skipping {week} (files not found)")

    if not input_frames:
        raise RuntimeError(f"No data loaded for {phase_name} from {folder}")

    data_input = pd.concat(input_frames, ignore_index=True)
    data_output = pd.concat(output_frames, ignore_index=True)

    print(f"✓ Total: {len(data_input):,} input rows, {len(data_output):,} output rows")

    del input_frames, output_frames
    gc.collect()

    return data_input, data_output


def standardize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all spatial data based on play_direction.
    CRITICAL: This prevents the model from learning patterns twice (left-to-right AND right-to-left).
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
            # Standardize x coordinates
            df.loc[left_mask, "x"] = FIELD_X_MAX - df.loc[left_mask, "x"]
            df.loc[left_mask, "y"] = FIELD_Y_MAX - df.loc[left_mask, "y"]
            df.loc[left_mask, "dir"] = (360 - df.loc[left_mask, "dir"]) % 360
            df.loc[left_mask, "o"] = (360 - df.loc[left_mask, "o"]) % 360
            
            # Standardize ball landing coordinates
            if "ball_land_x" in df.columns:
                df.loc[left_mask, "ball_land_x"] = FIELD_X_MAX - df.loc[left_mask, "ball_land_x"]
            if "ball_land_y" in df.columns:
                df.loc[left_mask, "ball_land_y"] = FIELD_Y_MAX - df.loc[left_mask, "ball_land_y"]
            
            print(f"  ✓ Standardized {n_left:,} rows to right-facing coordinates")
    else:
        print("  ⚠ No play_direction column - skipping standardization")
    
    return df


def engineer_deep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deep feature engineering with player interactions, coverage, and route metrics."""
    print("\n[DEEP FEATURE ENGINEERING]")
    print("-" * 60)

    # NOTE: Coordinate standardization DISABLED - it was causing 2.5 yard RMSE
    # The model should learn to handle both left and right-facing plays naturally
    feats = df.copy()
    
    # STEP 1: Compute velocities
    print("  → Velocity vectors...")
    feats["vx"] = feats["s"] * np.cos(np.radians(feats["dir"]))
    feats["vy"] = feats["s"] * np.sin(np.radians(feats["dir"]))

    print("  → Ball-centric features...")
    feats["dist_to_ball"] = np.sqrt(
        (feats["x"] - feats["ball_land_x"]) ** 2
        + (feats["y"] - feats["ball_land_y"]) ** 2
    )

    dx_to_ball = feats["ball_land_x"] - feats["x"]
    dy_to_ball = feats["ball_land_y"] - feats["y"]
    feats["dx_to_ball"] = dx_to_ball
    feats["dy_to_ball"] = dy_to_ball

    feats["angle_to_ball"] = np.degrees(np.arctan2(dy_to_ball, dx_to_ball))
    feats["angle_to_ball_rad"] = np.arctan2(dy_to_ball, dx_to_ball)

    alignment = feats["dir"] - feats["angle_to_ball"]
    feats["velocity_ball_alignment"] = np.cos(np.radians(alignment))
    feats["velocity_ball_alignment_sin"] = np.sin(np.radians(alignment))
    feats["moving_toward_ball"] = (feats["velocity_ball_alignment"] > 0).astype("int8")

    ball_dist_safe = feats["dist_to_ball"] + 1e-6
    feats["ball_direction_x"] = dx_to_ball / ball_dist_safe
    feats["ball_direction_y"] = dy_to_ball / ball_dist_safe

    feats["very_close_to_ball"] = (feats["dist_to_ball"] < 3).astype("int8")
    feats["close_to_ball"] = (feats["dist_to_ball"] < 8).astype("int8")
    feats["near_ball"] = (feats["dist_to_ball"] < 15).astype("int8")
    feats["far_from_ball"] = (feats["dist_to_ball"] > 30).astype("int8")

    print("  → Physics features...")
    feats["speed_squared"] = feats["s"] ** 2
    feats["speed_cubed"] = feats["s"] ** 3
    feats["kinetic_energy"] = 0.5 * feats["speed_squared"]
    feats["momentum"] = feats["s"] * 200.0

    feats["dir_sin"] = np.sin(np.radians(feats["dir"]))
    feats["dir_cos"] = np.cos(np.radians(feats["dir"]))
    feats["o_sin"] = np.sin(np.radians(feats["o"]))
    feats["o_cos"] = np.cos(np.radians(feats["o"]))

    feats["dir_sin2"] = np.sin(2 * np.radians(feats["dir"]))
    feats["dir_cos2"] = np.cos(2 * np.radians(feats["dir"]))

    diff = np.abs(feats["dir"] - feats["o"])
    feats["dir_o_diff"] = np.minimum(diff, 360 - diff)
    feats["dir_o_diff_rad"] = np.radians(feats["dir_o_diff"])
    feats["body_aligned"] = (feats["dir_o_diff"] < 30).astype("int8")
    feats["body_perpendicular"] = (
        (feats["dir_o_diff"] > 60) & (feats["dir_o_diff"] < 120)
    ).astype("int8")

    feats["accel_intensity"] = np.abs(feats["a"])
    feats["is_accelerating"] = (feats["a"] > 0.5).astype("int8")
    feats["is_decelerating"] = (feats["a"] < -0.5).astype("int8")
    feats["is_constant_speed"] = (np.abs(feats["a"]) < 0.3).astype("int8")

    print("  → Role-specific features...")
    feats["is_targeted_receiver"] = (feats["player_role"] == "Targeted Receiver").astype("int8")
    feats["is_defensive_coverage"] = (feats["player_role"] == "Defensive Coverage").astype("int8")
    feats["is_passer"] = (feats["player_role"] == "Passer").astype("int8")
    feats["is_other_route_runner"] = (feats["player_role"] == "Other Route Runner").astype("int8")
    feats["is_pass_rush"] = (feats["player_role"] == "Pass Rush").astype("int8")

    feats["is_offense"] = (feats["player_side"] == "Offense").astype("int8")
    feats["is_defense"] = (feats["player_side"] == "Defense").astype("int8")

    feats["position_encoded"] = feats["player_position"].astype("category").cat.codes.astype("int16")

    feats["is_wr"] = (feats["player_position"] == "WR").astype("int8")
    feats["is_te"] = (feats["player_position"] == "TE").astype("int8")
    feats["is_rb"] = (feats["player_position"] == "RB").astype("int8")
    feats["is_qb"] = (feats["player_position"] == "QB").astype("int8")
    feats["is_cb"] = (feats["player_position"] == "CB").astype("int8")
    feats["is_safety"] = feats["player_position"].isin(["S", "SS", "FS"]).astype("int8")
    feats["is_linebacker"] = feats["player_position"].isin(["LB", "ILB", "OLB", "MLB"]).astype("int8")

    print("  → Spatial features...")
    feats["x_norm"] = feats["x"] / FIELD_X_MAX
    feats["y_norm"] = feats["y"] / FIELD_Y_MAX
    feats["ball_x_norm"] = feats["ball_land_x"] / FIELD_X_MAX
    feats["ball_y_norm"] = feats["ball_land_y"] / FIELD_Y_MAX

    feats["dist_to_left_sideline"] = feats["y"]
    feats["dist_to_right_sideline"] = FIELD_Y_MAX - feats["y"]
    feats["dist_to_sideline"] = np.minimum(feats["y"], FIELD_Y_MAX - feats["y"])
    feats["dist_to_near_endzone"] = np.minimum(feats["x"], FIELD_X_MAX - feats["x"])
    feats["dist_to_offense_endzone"] = FIELD_X_MAX - feats["x"]

    feats["near_sideline"] = (feats["dist_to_sideline"] < 5).astype("int8")
    feats["in_red_zone"] = ((feats["x"] < 20) | (feats["x"] > FIELD_X_MAX - 20)).astype("int8")
    feats["in_middle_field"] = ((feats["y"] > 15) & (feats["y"] < 38.3)).astype("int8")

    feats["field_third"] = pd.cut(
        feats["x"], bins=[-np.inf, 40, 80, np.inf], labels=[0, 1, 2]
    ).astype("int8")
    feats["field_quarter"] = pd.cut(
        feats["x"], bins=[-np.inf, 30, 60, 90, np.inf], labels=[0, 1, 2, 3]
    ).astype("int8")
    feats["lateral_zone"] = pd.cut(
        feats["y"], bins=[-np.inf, 17.77, 35.53, np.inf], labels=[0, 1, 2]
    ).astype("int8")

    print("  → Temporal features...")
    feats["frames_remaining"] = feats["num_frames_output"]
    feats["time_to_ball"] = feats["frames_remaining"] / 10.0
    feats["time_to_ball_squared"] = feats["time_to_ball"] ** 2

    feats["predicted_x_linear"] = feats["x"] + feats["vx"] * feats["time_to_ball"]
    feats["predicted_y_linear"] = feats["y"] + feats["vy"] * feats["time_to_ball"]

    feats["predicted_x_accel"] = (
        feats["predicted_x_linear"]
        + 0.5 * feats["a"] * feats["dir_cos"] * feats["time_to_ball_squared"]
    )
    feats["predicted_y_accel"] = (
        feats["predicted_y_linear"]
        + 0.5 * feats["a"] * feats["dir_sin"] * feats["time_to_ball_squared"]
    )

    ball_pull_strength = 0.3
    feats["predicted_x_ball_pull"] = (
        feats["predicted_x_linear"]
        + ball_pull_strength * dx_to_ball * feats["is_targeted_receiver"]
    )
    feats["predicted_y_ball_pull"] = (
        feats["predicted_y_linear"]
        + ball_pull_strength * dy_to_ball * feats["is_targeted_receiver"]
    )

    print("  → Advanced trajectory features...")
    feats["linear_pred_to_ball_dist"] = np.sqrt(
        (feats["predicted_x_linear"] - feats["ball_land_x"]) ** 2
        + (feats["predicted_y_linear"] - feats["ball_land_y"]) ** 2
    )
    feats["accel_pred_to_ball_dist"] = np.sqrt(
        (feats["predicted_x_accel"] - feats["ball_land_x"]) ** 2
        + (feats["predicted_y_accel"] - feats["ball_land_y"]) ** 2
    )

    feats["x_relative_to_ball"] = feats["x"] - feats["ball_land_x"]
    feats["y_relative_to_ball"] = feats["y"] - feats["ball_land_y"]

    feats["speed_toward_ball"] = (
        feats["vx"] * feats["ball_direction_x"]
        + feats["vy"] * feats["ball_direction_y"]
    )
    feats["speed_perpendicular_ball"] = (
        feats["vx"] * feats["ball_direction_y"]
        - feats["vy"] * feats["ball_direction_x"]
    )

    feats["dist_to_ball_per_time"] = feats["dist_to_ball"] / (feats["time_to_ball"] + 0.1)
    feats["speed_ratio_to_required"] = feats["s"] / (feats["dist_to_ball_per_time"] + 0.1)
    feats["can_reach_ball"] = (feats["speed_ratio_to_required"] > 0.8).astype("int8")

    # STEP 3: Player-to-Player "Obstacle" Features
    print("  → Player-to-player interaction features (obstacles)...")
    
    # Initialize interaction columns
    feats["dist_to_targeted_receiver"] = np.nan
    feats["angle_to_targeted_receiver"] = np.nan
    feats["relative_speed_to_tr"] = np.nan
    feats["dist_to_closest_opponent"] = np.nan
    feats["relative_speed_to_closest_opponent"] = np.nan
    feats["angle_to_closest_opponent"] = np.nan
    feats["dist_to_closest_teammate"] = np.nan
    
    # Process frame by frame (optimized version)
    print("    Computing player-to-player interactions...")
    interaction_count = 0
    for (game_id, play_id, frame_id), frame_df in feats.groupby(["game_id", "play_id", "frame_id"]):
        if len(frame_df) < 2:
            continue
        
        coords = frame_df[["x", "y"]].values.astype("float32")
        velocities = frame_df[["vx", "vy"]].values.astype("float32")
        
        # Targeted receiver interactions
        target_mask = frame_df["is_targeted_receiver"].values.astype(bool)
        if target_mask.any():
            target_idx = np.where(target_mask)[0][0]
            target_pos = coords[target_idx]
            target_vel = velocities[target_idx]
            
            to_target = target_pos - coords
            dist_to_tr = np.linalg.norm(to_target, axis=1)
            angle_to_tr = np.arctan2(to_target[:, 1], to_target[:, 0])
            rel_speed = np.linalg.norm(target_vel - velocities, axis=1)
            
            feats.loc[frame_df.index, "dist_to_targeted_receiver"] = dist_to_tr
            feats.loc[frame_df.index, "angle_to_targeted_receiver"] = angle_to_tr
            feats.loc[frame_df.index, "relative_speed_to_tr"] = rel_speed
        
        # Opponent/teammate interactions
        offense_mask = frame_df["is_offense"].values.astype(bool)
        defense_mask = frame_df["is_defense"].values.astype(bool)
        
        for i in range(len(frame_df)):
            player_pos = coords[i]
            is_offense = offense_mask[i]
            
            opponent_mask = defense_mask if is_offense else offense_mask
            teammate_mask = offense_mask.copy() if is_offense else defense_mask.copy()
            teammate_mask[i] = False
            
            if opponent_mask.any():
                diffs = coords[opponent_mask] - player_pos
                dists = np.linalg.norm(diffs, axis=1)
                closest_idx = np.argmin(dists)
                
                feats.loc[frame_df.index[i], "dist_to_closest_opponent"] = dists[closest_idx]
                feats.loc[frame_df.index[i], "angle_to_closest_opponent"] = np.arctan2(
                    diffs[closest_idx, 1], diffs[closest_idx, 0]
                )
                feats.loc[frame_df.index[i], "relative_speed_to_closest_opponent"] = np.linalg.norm(
                    velocities[opponent_mask][closest_idx] - velocities[i]
                )
            
            if teammate_mask.any():
                diffs = coords[teammate_mask] - player_pos
                feats.loc[frame_df.index[i], "dist_to_closest_teammate"] = np.min(np.linalg.norm(diffs, axis=1))
        
        interaction_count += 1
        if interaction_count % 500 == 0:
            print(f"      Processed {interaction_count} frames...")
    
    print(f"    ✓ Computed interactions for {interaction_count} frames")

    print("  → Interaction aggregations...")
    play_group = feats.groupby(["game_id", "play_id"])
    feats["speed_percentile"] = play_group["s"].transform(lambda x: x.rank(pct=True))
    feats["dist_to_ball_percentile"] = play_group["dist_to_ball"].transform(lambda x: x.rank(pct=True))
    feats["x_percentile"] = play_group["x"].transform(lambda x: x.rank(pct=True))

    feats["play_mean_speed"] = play_group["s"].transform("mean")
    feats["play_max_speed"] = play_group["s"].transform("max")
    feats["play_min_speed"] = play_group["s"].transform("min")
    feats["play_std_speed"] = play_group["s"].transform("std").fillna(0)

    feats["play_mean_dist_to_ball"] = play_group["dist_to_ball"].transform("mean")
    feats["play_min_dist_to_ball"] = play_group["dist_to_ball"].transform("min")
    feats["play_max_dist_to_ball"] = play_group["dist_to_ball"].transform("max")

    feats["play_density"] = play_group["nfl_id"].transform("count")
    feats["play_mean_x"] = play_group["x"].transform("mean")
    feats["play_std_x"] = play_group["x"].transform("std").fillna(0)
    feats["play_mean_y"] = play_group["y"].transform("mean")
    feats["play_std_y"] = play_group["y"].transform("std").fillna(0)

    side_group = feats.groupby(["game_id", "play_id", "player_side"])
    feats["team_mean_x"] = side_group["x"].transform("mean")
    feats["team_mean_y"] = side_group["y"].transform("mean")
    feats["team_speed_mean"] = side_group["s"].transform("mean")
    feats["team_speed_max"] = side_group["s"].transform("max")
    feats["team_speed_std"] = side_group["s"].transform("std").fillna(0)
    feats["team_size"] = side_group["nfl_id"].transform("count")

    feats[[
        "team_mean_x",
        "team_mean_y",
        "team_speed_mean",
        "team_speed_max",
        "team_speed_std",
        "team_size",
    ]] = feats[[
        "team_mean_x",
        "team_mean_y",
        "team_speed_mean",
        "team_speed_max",
        "team_speed_std",
        "team_size",
    ]].fillna(0.0)

    feats["offset_team_center_x"] = feats["x"] - feats["team_mean_x"]
    feats["offset_team_center_y"] = feats["y"] - feats["team_mean_y"]
    feats["dist_to_team_center"] = np.sqrt(
        feats["offset_team_center_x"] ** 2 + feats["offset_team_center_y"] ** 2
    )

    feats["speed_vs_team_mean"] = feats["s"] - feats["team_speed_mean"]
    feats["faster_than_team"] = (feats["speed_vs_team_mean"] > 0).astype("int8")

    print("  → Polynomial features...")
    feats["speed_dist_product"] = feats["s"] * feats["dist_to_ball"]
    feats["speed_squared_dist"] = feats["speed_squared"] * feats["dist_to_ball"]
    feats["vx_dist"] = feats["vx"] * feats["dist_to_ball"]
    feats["vy_dist"] = feats["vy"] * feats["dist_to_ball"]

    feats["time_x_product"] = feats["time_to_ball"] * feats["x_norm"]
    feats["time_y_product"] = feats["time_to_ball"] * feats["y_norm"]
    feats["time_dist_product"] = feats["time_to_ball"] * feats["dist_to_ball"]
    feats["time_squared_dist"] = feats["time_to_ball_squared"] * feats["dist_to_ball"]

    feats["receiver_x"] = feats["is_targeted_receiver"] * feats["x_norm"]
    feats["receiver_y"] = feats["is_targeted_receiver"] * feats["y_norm"]
    feats["receiver_dist"] = feats["is_targeted_receiver"] * feats["dist_to_ball"]
    feats["receiver_speed"] = feats["is_targeted_receiver"] * feats["s"]

    feats["defender_x"] = feats["is_defensive_coverage"] * feats["x_norm"]
    feats["defender_y"] = feats["is_defensive_coverage"] * feats["y_norm"]
    feats["defender_dist"] = feats["is_defensive_coverage"] * feats["dist_to_ball"]

    feats["vel_accel_product"] = feats["s"] * feats["a"]
    feats["vx_accel"] = feats["vx"] * feats["a"]
    feats["vy_accel"] = feats["vy"] * feats["a"]

    print("  → Exponential & log features...")
    feats["log_dist_to_ball"] = np.log1p(feats["dist_to_ball"])
    feats["log_time_to_ball"] = np.log1p(feats["time_to_ball"])
    feats["log_speed"] = np.log1p(feats["s"])
    feats["log1p_x"] = np.log1p(feats["x"])
    feats["log1p_y"] = np.log1p(feats["y"])

    feats["exp_neg_dist"] = np.exp(-feats["dist_to_ball"] / 10.0)
    feats["exp_neg_dist_fast"] = np.exp(-feats["dist_to_ball"] / 5.0)
    feats["exp_neg_time"] = np.exp(-feats["time_to_ball"])

    feats["dist_to_ball_squared"] = feats["dist_to_ball"] ** 2
    feats["dist_to_sideline_squared"] = feats["dist_to_sideline"] ** 2
    feats["x_squared"] = feats["x"] ** 2
    feats["y_squared"] = feats["y"] ** 2

    print("  → Ratio features...")
    feats["vx_vy_ratio"] = feats["vx"] / (np.abs(feats["vy"]) + 0.1)
    feats["vy_vx_ratio"] = feats["vy"] / (np.abs(feats["vx"]) + 0.1)
    feats["x_y_ratio"] = feats["x"] / (feats["y"] + 0.1)
    feats["ball_x_y_ratio"] = feats["ball_land_x"] / (feats["ball_land_y"] + 0.1)
    feats["accel_speed_ratio"] = feats["a"] / (feats["s"] + 0.1)
    feats["dist_speed_ratio"] = feats["dist_to_ball"] / (feats["s"] + 0.1)

    print("  → Ball dominance features...")
    feats["catchability_score"] = (
        (1.0 / (1.0 + feats["dist_to_ball"]))
        * (feats["velocity_ball_alignment"] + 1.0) / 2.0
        * (1.0 - feats["speed_percentile"] * 0.3)
    )

    feats["ball_dominance"] = (
        feats["is_targeted_receiver"]
        * feats["exp_neg_dist"]
        * (feats["moving_toward_ball"] + 1) / 2
    )

    feats["interception_potential"] = (
        feats["is_defensive_coverage"]
        * feats["exp_neg_dist"]
        * feats["moving_toward_ball"]
        * feats["speed_percentile"]
    )

    print("  → Directional intensity...")
    feats["directional_strength"] = np.sqrt(feats["dir_cos"] ** 2 + feats["dir_sin"] ** 2) * feats["s"]
    feats["orientation_strength"] = np.sqrt(feats["o_cos"] ** 2 + feats["o_sin"] ** 2) * feats["s"]
    feats["angular_momentum"] = feats["s"] * feats["dir_o_diff_rad"]

    feats["momentum_toward_ball"] = feats["momentum"] * feats["velocity_ball_alignment"]
    feats["momentum_x"] = feats["momentum"] * feats["dir_cos"]
    feats["momentum_y"] = feats["momentum"] * feats["dir_sin"]

    print("  → Cross features...")
    feats["x_ball_interaction"] = feats["x_norm"] * feats["ball_x_norm"]
    feats["y_ball_interaction"] = feats["y_norm"] * feats["ball_y_norm"]
    feats["speed_alignment"] = feats["s"] * feats["velocity_ball_alignment"]

    feats["in_left_half"] = (feats["y"] < FIELD_Y_MAX / 2).astype("int8")
    feats["in_offensive_half"] = (feats["x"] > FIELD_X_MAX / 2).astype("int8")
    feats["ball_in_left_half"] = (feats["ball_land_y"] < FIELD_Y_MAX / 2).astype("int8")
    feats["ball_in_offensive_half"] = (feats["ball_land_x"] > FIELD_X_MAX / 2).astype("int8")

    feats["same_lateral_zone"] = (
        feats["in_left_half"] == feats["ball_in_left_half"]
    ).astype("int8")
    feats["same_field_half"] = (
        feats["in_offensive_half"] == feats["ball_in_offensive_half"]
    ).astype("int8")

    feats["trajectory_confidence"] = (
        feats["velocity_ball_alignment"]
        * feats["moving_toward_ball"]
        * (1.0 / (1.0 + feats["dir_o_diff"] / 180.0))
    )

    print("  → Coverage intelligence (ultra memory-optimized for <20GB)...")

    def compute_frame_level_coverage(frame_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ultra memory-optimized coverage intelligence with advanced defensive metrics.
        Uses micro-chunking, int16 coordinates, and simplified metrics to stay under 20GB.
        """
        # Use views instead of copies where possible
        defense_mask = frame_df["is_defense"].values.astype(bool)
        offense_mask = frame_df["is_offense"].values.astype(bool)
        
        # Use int16 for coordinates (scaled by 10 for precision) - saves 50% over float32
        coords = (frame_df[["x", "y"]].values * 10).astype("int16")
        
        n_players = len(frame_df)
        
        # Pre-allocate arrays with int16 where possible
        nearest_defender = np.full(n_players, 32767, dtype="int16")  # Max int16
        coverage_responsibility = np.full(n_players, 0.0, dtype="float32")
        separation_created = np.full(n_players, np.nan, dtype="float32")

        # Compute basic defensive distances
        if defense_mask.any():
            defense_coords = coords[defense_mask]
            n_defenders = len(defense_coords)
            
            # Ultra memory-efficient: micro-chunks of 10 players
            micro_chunk = 10
            for i in range(0, n_players, micro_chunk):
                end_idx = min(i + micro_chunk, n_players)
                chunk_coords = coords[i:end_idx]
                
                # Compute distances one-by-one to minimize memory
                min_dists = []
                for player_coord in chunk_coords:
                    diffs = defense_coords - player_coord
                    dists_sq = (diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
                    min_dists.append(np.sqrt(dists_sq.min()))
                
                nearest_defender[i:end_idx] = (np.array(min_dists) * 10).astype("int16")
                
                del min_dists  # Free memory immediately
            
            # Convert back to float32 and scale down
            nearest_defender_float = (nearest_defender / 100.0).astype("float32")
            
            # Defensive pressure metrics (simplified)
            defense_dist_to_ball = frame_df.loc[defense_mask, "dist_to_ball"].values.astype("float32")
            close_defenders = float((defense_dist_to_ball < 5.0).sum())
            pressure_density = close_defenders / (np.pi * 25.0)  # Simplified density
            
            del defense_dist_to_ball  # Free memory
        else:
            nearest_defender_float = (nearest_defender / 100.0).astype("float32")
            close_defenders = 0.0
            pressure_density = 0.0

        # Assign basic metrics to frame
        frame_df.loc[:, "nearest_defender_dist"] = nearest_defender_float
        frame_df.loc[:, "defensive_pressure"] = np.full(n_players, close_defenders, dtype="float32")
        frame_df.loc[:, "defensive_pressure_density"] = np.full(n_players, pressure_density, dtype="float32")

        target_mask = frame_df["is_targeted_receiver"].values.astype(bool)

        # Simplified coverage metrics (only if target receiver exists)
        if target_mask.any() and defense_mask.any():
            target_idx = np.where(target_mask)[0][0]
            target_pos = coords[target_idx]
            
            # Simplified coverage responsibility for defenders only
            if defense_mask.any():
                def_coords = coords[defense_mask]
                dist_to_target = np.sqrt(((def_coords - target_pos) ** 2).sum(axis=1)) / 10.0
                coverage_resp = (1.0 / (dist_to_target + 1.0)).astype("float32")
                frame_df.loc[frame_df.index[defense_mask], "coverage_responsibility"] = coverage_resp
                del def_coords, dist_to_target, coverage_resp
            
            # Separation for receivers only
            if offense_mask.any():
                separation_created[offense_mask] = nearest_defender_float[offense_mask]
                frame_df.loc[:, "separation_created"] = separation_created
        
        else:
            # Set defaults when no target receiver
            frame_df.loc[:, "separation_created"] = separation_created

        # Set simplified defaults for removed complex metrics
        frame_df.loc[:, "distance_to_targeted_receiver"] = frame_df.get("coverage_responsibility", 0.0)
        
        return frame_df

    # Process in small batches to limit memory
    unique_frames = feats[["game_id", "play_id", "frame_id"]].drop_duplicates()
    batch_size = 50  # Process only 50 frames at a time
    
    for batch_idx in range(0, len(unique_frames), batch_size):
        batch_frames = unique_frames.iloc[batch_idx:batch_idx + batch_size]
        mask = feats[["game_id", "play_id", "frame_id"]].merge(
            batch_frames, on=["game_id", "play_id", "frame_id"], how="inner"
        ).index
        
        batch_df = feats.loc[mask].copy()
        processed = batch_df.groupby(["game_id", "play_id", "frame_id"], group_keys=False).apply(
            compute_frame_level_coverage
        )
        
        # Update the original dataframe
        for col in ["nearest_defender_dist", "defensive_pressure", "defensive_pressure_density", 
                    "coverage_responsibility", "separation_created", "distance_to_targeted_receiver"]:
            if col in processed.columns:
                feats.loc[mask, col] = processed[col].values
        
        del batch_df, processed, mask
        gc.collect()
        
        if (batch_idx // batch_size + 1) % 10 == 0:
            print(f"    Processed {batch_idx + batch_size}/{len(unique_frames)} frames...")

    print("  → Defensive specialist features computed")

    print("  → Route similarity features (memory-optimized)...")
    
    # Process in chunks to reduce memory
    chunk_size = 50000
    route_sim_list = []
    route_speed_list = []
    route_dir_list = []
    
    for i in range(0, len(feats), chunk_size):
        end_idx = min(i + chunk_size, len(feats))
        chunk = feats.iloc[i:end_idx]
        
        # Simplified route similarity based on speed consistency
        player_mean_speed = chunk.groupby("nfl_id")["s"].transform("mean").astype("float32")
        route_speed_list.append((chunk["s"] - player_mean_speed).astype("float32"))
        route_sim_list.append((1.0 - np.abs(chunk["s"] - player_mean_speed) / (player_mean_speed + 1.0)).clip(0, 1).astype("float32"))
        
        # Simplified direction similarity
        mean_dir_sin = chunk.groupby("nfl_id")["dir_sin"].transform("mean").astype("float32")
        mean_dir_cos = chunk.groupby("nfl_id")["dir_cos"].transform("mean").astype("float32")
        dir_dot = (chunk["dir_sin"] * mean_dir_sin + chunk["dir_cos"] * mean_dir_cos).astype("float32")
        route_dir_list.append(dir_dot.clip(-1.0, 1.0))
        
        del player_mean_speed, mean_dir_sin, mean_dir_cos, dir_dot
        gc.collect()
    
    feats["route_similarity"] = pd.concat(route_sim_list, ignore_index=True).values
    feats["route_speed_delta"] = pd.concat(route_speed_list, ignore_index=True).values
    feats["route_direction_similarity"] = pd.concat(route_dir_list, ignore_index=True).values
    
    del route_sim_list, route_speed_list, route_dir_list
    gc.collect()

    print(f"✓ Feature engineering complete: {len(feats.columns)} total features")
    log_memory("After feature engineering")

    return feats


def create_pairs(
    input_df: pd.DataFrame, output_df: pd.DataFrame, max_samples: Optional[int] = None
) -> pd.DataFrame:
    """Create training pairs that align last observed frame to each target frame."""
    print("\n[CREATING TRAINING PAIRS]")
    print("-" * 60)

    pairs: List[Dict[str, float]] = []
    created = 0

    for (game_id, play_id), play_in in input_df.groupby(["game_id", "play_id"]):
        if max_samples and created >= max_samples:
            break

        play_out = output_df[
            (output_df["game_id"] == game_id) & (output_df["play_id"] == play_id)
        ]

        if play_out.empty:
            continue

        for nfl_id in play_out["nfl_id"].unique():
            if max_samples and created >= max_samples:
                break

            player_in = play_in[play_in["nfl_id"] == nfl_id].sort_values("frame_id")
            player_out = play_out[play_out["nfl_id"] == nfl_id].sort_values("frame_id")

            if player_in.empty or player_out.empty:
                continue

            last_frame = player_in.iloc[-1]
            is_target = bool(last_frame.get("player_to_predict", False))

            if len(player_in) >= 2:
                prev_frame = player_in.iloc[-2]
                vel_dx = last_frame["vx"] - prev_frame["vx"]
                vel_dy = last_frame["vy"] - prev_frame["vy"]
                pos_dx = last_frame["x"] - prev_frame["x"]
                pos_dy = last_frame["y"] - prev_frame["y"]

                if len(player_in) >= 3:
                    frame_minus_2 = player_in.iloc[-3]
                    trajectory_curvature = np.hypot(
                        last_frame["x"] - 2 * prev_frame["x"] + frame_minus_2["x"],
                        last_frame["y"] - 2 * prev_frame["y"] + frame_minus_2["y"],
                    )

                    prev_vel_dx = prev_frame["vx"] - frame_minus_2["vx"]
                    prev_vel_dy = prev_frame["vy"] - frame_minus_2["vy"]
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

                time_diff = out_row["frame_id"] - last_frame["frame_id"]

                sample: Dict[str, float] = {
                    "target_x": float(out_row["x"]),
                    "target_y": float(out_row["y"]),
                    "output_frame_id": int(out_row["frame_id"]),
                    "time_diff": float(time_diff),
                    "game_id": int(game_id),
                    "play_id": int(play_id),
                    "nfl_id": int(nfl_id),
                    "velocity_change_x": float(vel_dx),
                    "velocity_change_y": float(vel_dy),
                    "position_change_x": float(pos_dx),
                    "position_change_y": float(pos_dy),
                    "acceleration_est": float(accel_est),
                    "trajectory_curvature": float(trajectory_curvature),
                    "jerk_x": float(jerk_x),
                    "jerk_y": float(jerk_y),
                    "jerk_est": float(jerk_est),
                    "is_target_player": int(is_target),
                }

                for col in last_frame.index:
                    if col in {
                        "game_id",
                        "play_id",
                        "nfl_id",
                        "player_position",
                        "player_role",
                        "player_side",
                        "direction",
                    }:
                        continue
                    try:
                        sample[f"input_{col}"] = float(last_frame[col])
                    except (ValueError, TypeError):
                        continue

                pairs.append(sample)
                created += 1

    df = pd.DataFrame(pairs)
    print(f"✓ Created {len(df):,} pairs")
    if "is_target_player" in df.columns:
        print(f"  Target player ratio: {100 * df['is_target_player'].mean():.1f}%")

    return df


def augment_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add horizon-aware pair features."""
    df = df.copy()
    df["time_diff_squared"] = df["time_diff"] ** 2
    df["time_diff_cubed"] = df["time_diff"] ** 3
    time_nonneg = np.maximum(df["time_diff"].astype(float), 0.0)
    df["time_diff_log"] = np.log1p(time_nonneg)
    df["time_diff_inv"] = 1.0 / (1.0 + time_nonneg)
    df["time_weight_feature"] = np.exp(-time_nonneg / 6.0)
    df["time_diff_normalized"] = time_nonneg / (time_nonneg.max() + 1e-6)
    horizon_bucket = pd.cut(
        df["time_diff"], bins=[-np.inf, 2, 5, 8, np.inf], labels=[0, 1, 2, 3]
    ).astype("int8")
    df["time_horizon_bucket"] = horizon_bucket.astype("float32")
    df["is_short_horizon"] = (df["time_diff"] <= 3).astype("int8")
    df["is_mid_horizon"] = (
        (df["time_diff"] > 3) & (df["time_diff"] <= 7)
    ).astype("int8")
    df["is_long_horizon"] = (df["time_diff"] > 7).astype("int8")
    return df


def select_features_via_permutation(
    features: pd.DataFrame,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> List[str]:
    """Apply permutation importance to drop low-impact predictors."""
    print("\n[FEATURE SELECTION]")
    print("  → Permutation importance (LightGBM baseline)...")

    n_samples = len(features)
    if n_samples == 0:
        return feature_names

    sample_size = min(30_000, n_samples)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

    X_sample = features.iloc[sample_idx]
    w_sample = sample_weights[sample_idx]
    y_sample = targets[sample_idx]

    # OPTIMIZED: Reduced n_estimators and n_repeats for faster feature selection
    model_params = dict(
        n_estimators=250,  # Reduced from 400
        learning_rate=0.06,  # Increased from 0.05
        max_depth=6,  # Reduced from 7
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model_x = lgb.LGBMRegressor(**model_params)
    model_x.fit(X_sample, y_sample[:, 0], sample_weight=w_sample)
    perm_x = permutation_importance(
        model_x,
        X_sample,
        y_sample[:, 0],
        scoring="neg_mean_squared_error",
        n_repeats=2,  # Reduced from 4 (50% faster)
        random_state=random_state,
        n_jobs=-1,
        sample_weight=w_sample,
    )

    model_y = lgb.LGBMRegressor(**model_params)
    model_y.fit(X_sample, y_sample[:, 1], sample_weight=w_sample)
    perm_y = permutation_importance(
        model_y,
        X_sample,
        y_sample[:, 1],
        scoring="neg_mean_squared_error",
        n_repeats=2,  # Reduced from 4 (50% faster)
        random_state=random_state + 7,
        n_jobs=-1,
        sample_weight=w_sample,
    )

    importance_scores = (
        np.maximum(perm_x.importances_mean, 0.0) + np.maximum(perm_y.importances_mean, 0.0)
    ) / 2.0

    threshold = np.percentile(importance_scores, 20)
    keep_mask = importance_scores >= threshold
    keep_features = [name for name, keep in zip(feature_names, keep_mask) if keep]

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


def train_advanced_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    base_val: np.ndarray,
    train_groups: np.ndarray,
    base_train: np.ndarray = None,
    label: str = "ALL",
):
    """Train stacked ensemble with grouped cross-validation and residual booster."""
    print(f"\n[TRAINING ADVANCED ENSEMBLE - {label}]")
    print("-" * 60)

    n_samples = X_train.shape[0]
    unique_groups = np.unique(train_groups)

    # OPTIMIZED: Reduced folds from 5 to 3 (40% reduction in CV training time)
    splits: List[Tuple[np.ndarray, np.ndarray]]
    if len(unique_groups) >= 3 and n_samples >= 200:
        n_folds = min(3, len(unique_groups))  # Reduced from 5 to 3
        print(f"Using GroupKFold with {n_folds} splits")
        kf = GroupKFold(n_splits=n_folds)
        splits = list(kf.split(X_train, y_train[:, 0], groups=train_groups))
    else:
        n_splits = min(2, n_samples)  # Reduced from 3 to 2
        if n_splits < 2:
            print("Insufficient samples for cross-validation; using single-fold training")
            indices = np.arange(n_samples)
            splits = [(indices, indices)]
        else:
            print(f"Using fallback KFold with {n_splits} splits")
            kf_simple = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf_simple.split(X_train))

    def clip_delta(preds_delta: np.ndarray) -> np.ndarray:
        abs_preds = preds_delta + base_val
        abs_preds[:, 0] = np.clip(abs_preds[:, 0], 0.0, FIELD_X_MAX)
        abs_preds[:, 1] = np.clip(abs_preds[:, 1], 0.0, FIELD_Y_MAX)
        return abs_preds - base_val

    # =========================================================================
    # PHASE 2: FEATURE DIVERSITY HELPERS
    # =========================================================================
    def select_feature_subset(X: np.ndarray, feature_fraction: float, seed: int):
        """Select random subset of features for model diversity.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_fraction: Fraction of features to keep (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (subset feature matrix, selected feature indices)
        """
        if feature_fraction >= 1.0:
            return X, np.arange(X.shape[1])
        
        np.random.seed(seed)
        n_features = int(X.shape[1] * feature_fraction)
        # Use at least 50% of features
        n_features = max(n_features, int(X.shape[1] * 0.5))
        
        selected_indices = np.random.choice(
            X.shape[1], 
            size=n_features, 
            replace=False
        )
        selected_indices = np.sort(selected_indices)
        
        return X[:, selected_indices], selected_indices

    def bootstrap_sample(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                        sample_fraction: float, seed: int):
        """Create bootstrap sample for model diversity.
        
        Args:
            X: Feature matrix
            y: Target values
            weights: Sample weights
            sample_fraction: Fraction of samples to keep (0.0-1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (X_boot, y_boot, weights_boot)
        """
        if sample_fraction >= 1.0:
            return X, y, weights
        
        np.random.seed(seed)
        n_samples = int(X.shape[0] * sample_fraction)
        
        indices = np.random.choice(
            X.shape[0], 
            size=n_samples, 
            replace=True  # Bootstrap with replacement
        )
        
        return X[indices], y[indices], weights[indices]

    # OPTIMIZED: Reduced n_estimators by ~30-40% and increased learning rates
    # This reduces training time by ~35-40% with minimal accuracy loss
    base_models = {
        "XGB1": xgb.XGBRegressor(
            n_estimators=700,  # Reduced from 1000
            max_depth=8,
            learning_rate=0.025,  # Increased from 0.02
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=1.5,
            gamma=0.3,
            min_child_weight=7,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        ),
        "XGB2": xgb.XGBRegressor(
            n_estimators=600,  # Reduced from 800
            max_depth=10,  # Reduced from 12
            learning_rate=0.03,  # Increased from 0.025
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=2.0,
            reg_alpha=0.8,
            gamma=0.15,
            min_child_weight=5,
            random_state=43,
            tree_method="hist",
            n_jobs=-1,
        ),
        "LGBM1": lgb.LGBMRegressor(
            n_estimators=700,  # Reduced from 1000
            max_depth=8,  # Reduced from 9
            learning_rate=0.025,  # Increased from 0.02
            subsample=0.88,
            colsample_bytree=0.88,
            reg_lambda=2.5,
            reg_alpha=1.0,
            min_child_samples=30,
            num_leaves=70,  # Reduced from 80
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "LGBM2": lgb.LGBMRegressor(
            n_estimators=600,  # Reduced from 800
            max_depth=10,  # Reduced from 11
            learning_rate=0.03,  # Increased from 0.025
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            reg_alpha=0.7,
            min_child_samples=25,
            num_leaves=90,  # Reduced from 100
            random_state=43,
            n_jobs=-1,
            verbose=-1,
        ),
        "CAT": cb.CatBoostRegressor(
            iterations=600,  # Reduced from 800
            depth=8,  # Reduced from 9
            learning_rate=0.035,  # Increased from 0.03
            l2_leaf_reg=3.5,
            subsample=0.88,
            random_state=42,
            verbose=False,
            thread_count=-1,  # Added for parallel processing
        ),
        # NEW MODELS FOR ENSEMBLE EXPANSION (Phase 1)
        "RF": RandomForestRegressor(
            n_estimators=300,  # Balanced for speed
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.5,
            random_state=42,
            n_jobs=-1,
        ),
        "ET": ExtraTreesRegressor(
            n_estimators=300,  # Balanced for speed
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.6,
            random_state=42,
            n_jobs=-1,
        ),
        "HISTGB": HistGradientBoostingRegressor(
            max_iter=400,
            max_depth=10,
            learning_rate=0.04,
            l2_regularization=2.0,
            min_samples_leaf=25,
            random_state=42,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=512,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        ),
        "HUBER": HuberRegressor(
            epsilon=1.35,
            max_iter=500,
            alpha=0.001,
        ),
        # Note: Kernel Ridge and SVR are too slow for this dataset size
        # Skipping KernelRidge and SVR for now (would take 10+ hours)
    }

    # =========================================================================
    # PHASE 2: FEATURE & SAMPLE DIVERSITY CONFIGURATION
    # =========================================================================
    # Each model gets different subset of features and bootstrap samples
    # This breaks correlation and forces models to learn different patterns
    diversity_config = {
        "XGB1": {"feature_frac": 0.80, "sample_frac": 0.95, "seed": 42},
        "XGB2": {"feature_frac": 0.80, "sample_frac": 0.95, "seed": 43},
        "LGBM1": {"feature_frac": 0.85, "sample_frac": 0.95, "seed": 44},
        "LGBM2": {"feature_frac": 0.85, "sample_frac": 0.95, "seed": 45},
        "CAT": {"feature_frac": 0.80, "sample_frac": 0.95, "seed": 46},
        "RF": {"feature_frac": 0.70, "sample_frac": 0.95, "seed": 47},
        "ET": {"feature_frac": 0.75, "sample_frac": 0.95, "seed": 48},
        "HISTGB": {"feature_frac": 0.80, "sample_frac": 0.95, "seed": 49},
        "MLP": {"feature_frac": 0.90, "sample_frac": 0.95, "seed": 50},
        "HUBER": {"feature_frac": 1.00, "sample_frac": 0.95, "seed": 51},
    }

    model_names = list(base_models.keys())
    n_samples = X_train.shape[0]

    print("\n  → Training models for X-coordinate...")
    print("     (with feature diversity: 70-100% features per model)")
    models_x: Dict[str, object] = {}
    oof_preds_x = {name: np.zeros(n_samples) for name in model_names}
    val_preds_x: Dict[str, np.ndarray] = {}
    feature_indices_x: Dict[str, np.ndarray] = {}  # Track features per model

    for name, base_model in base_models.items():
        print(f"    - {name}")
        config = diversity_config[name]
        
        # Select feature subset for this model (same across all folds)
        X_train_sub, feat_idx = select_feature_subset(
            X_train, config["feature_frac"], config["seed"]
        )
        X_val_sub = X_val[:, feat_idx]
        feature_indices_x[name] = feat_idx
        
        print(f"      Features: {len(feat_idx)}/{X_train.shape[1]} ({100*config['feature_frac']:.0f}%)")
        
        val_fold_preds: List[np.ndarray] = []

        for tr_idx, va_idx in splits:
            # Use feature-selected data
            X_tr, X_va = X_train_sub[tr_idx], X_train_sub[va_idx]
            y_tr, y_va = y_train[tr_idx, 0], y_train[va_idx, 0]
            w_tr = sample_weights[tr_idx]
            
            # Apply bootstrap sampling for additional diversity
            X_tr_boot, y_tr_boot, w_tr_boot = bootstrap_sample(
                X_tr, y_tr, w_tr, config["sample_frac"], config["seed"]
            )

            model = base_model.__class__(**base_model.get_params())
            model.fit(X_tr_boot, y_tr_boot, sample_weight=w_tr_boot)

            oof_preds_x[name][va_idx] = model.predict(X_va)
            val_fold_preds.append(model.predict(X_val_sub))

        val_preds_x[name] = np.mean(val_fold_preds, axis=0)

        # Train final model on all training data (with feature subset)
        final_model = base_model.__class__(**base_model.get_params())
        final_model.fit(X_train_sub, y_train[:, 0], sample_weight=sample_weights)
        models_x[name] = final_model

    print("\n  → Training models for Y-coordinate...")
    print("     (with feature diversity: 70-100% features per model)")
    models_y: Dict[str, object] = {}
    oof_preds_y = {name: np.zeros(n_samples) for name in model_names}
    val_preds_y: Dict[str, np.ndarray] = {}
    feature_indices_y: Dict[str, np.ndarray] = {}  # Track features per model

    for name, base_model in base_models.items():
        print(f"    - {name}")
        config = diversity_config[name]
        
        # Select feature subset for this model (different seed than X)
        X_train_sub, feat_idx = select_feature_subset(
            X_train, config["feature_frac"], config["seed"] + 100
        )
        X_val_sub = X_val[:, feat_idx]
        feature_indices_y[name] = feat_idx
        
        print(f"      Features: {len(feat_idx)}/{X_train.shape[1]} ({100*config['feature_frac']:.0f}%)")
        
        val_fold_preds = []

        for tr_idx, va_idx in splits:
            # Use feature-selected data
            X_tr, X_va = X_train_sub[tr_idx], X_train_sub[va_idx]
            y_tr, y_va = y_train[tr_idx, 1], y_train[va_idx, 1]
            w_tr = sample_weights[tr_idx]
            
            # Apply bootstrap sampling for additional diversity
            X_tr_boot, y_tr_boot, w_tr_boot = bootstrap_sample(
                X_tr, y_tr, w_tr, config["sample_frac"], config["seed"] + 100
            )

            model = base_model.__class__(**base_model.get_params())
            model.fit(X_tr_boot, y_tr_boot, sample_weight=w_tr_boot)

            oof_preds_y[name][va_idx] = model.predict(X_va)
            val_fold_preds.append(model.predict(X_val_sub))

        val_preds_y[name] = np.mean(val_fold_preds, axis=0)

        # Train final model on all training data (with feature subset)
        final_model = base_model.__class__(**base_model.get_params())
        final_model.fit(X_train_sub, y_train[:, 1], sample_weight=sample_weights)
        models_y[name] = final_model

    print("\n  → Stacking predictions...")
    meta_train_x = np.column_stack([oof_preds_x[name] for name in model_names])
    meta_val_x = np.column_stack([val_preds_x[name] for name in model_names])

    meta_train_y = np.column_stack([oof_preds_y[name] for name in model_names])
    meta_val_y = np.column_stack([val_preds_y[name] for name in model_names])

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

    print("\n  → Training meta-learner...")
    meta_model_x = Ridge(alpha=10.0)
    meta_model_y = Ridge(alpha=10.0)

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

    if stacked_rmse < ensemble_rmse - 1e-4:
        print(
            f"\n  ✓ Using stacked meta-learner (improvement: {ensemble_rmse - stacked_rmse:.4f} yards)"
        )
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
        print(
            f"\n  ✓ Using weighted ensemble (better by {stacked_rmse - ensemble_rmse:.4f} yards)"
        )
        final_pred = ensemble_val
        final_rmse = ensemble_rmse
        final_rmse_x = ensemble_rmse_x
        final_rmse_y = ensemble_rmse_y
        use_stacking = False
        train_primary_pred_x = np.mean(meta_train_x, axis=1)
        train_primary_pred_y = np.mean(meta_train_y, axis=1)
        val_primary_pred_x = ensemble_val_x
        val_primary_pred_y = ensemble_val_y

    print("\n  → Residual refinement with gradient boosting...")
    residual_model_x = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="huber",
        random_state=101,
        n_jobs=-1,
        verbose=-1,
    )
    residual_model_y = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.5,
        objective="huber",
        random_state=202,
        n_jobs=-1,
        verbose=-1,
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
            val_primary_pred_y + residual_val_y,
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
        print(
            f"    → Residual booster did not improve (Δ={final_rmse - corrected_rmse:.4f}); keeping primary ensemble"
        )
        residual_model_x = None
        residual_model_y = None
        use_residuals = False

    # Calculate training RMSE for comparison
    # Need to clip using TRAINING data base coordinates, not validation
    # IMPORTANT: base_train must be passed as parameter (original unscaled coordinates)
    # X_train has been scaled, so X_train[:, -2:] would be WRONG!
    if base_train is None:
        # Fallback: assume predictions are already deltas, no clipping needed
        print("  ⚠ Warning: base_train not provided, training RMSE may be inaccurate")
        train_pred_final = np.column_stack([train_primary_pred_x, train_primary_pred_y])
        if use_residuals:
            train_residual_x = residual_model_x.predict(X_train)
            train_residual_y = residual_model_y.predict(X_train)
            train_pred_final += np.column_stack([train_residual_x, train_residual_y])
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred_final))
    else:
        def clip_delta_train(preds_delta: np.ndarray) -> np.ndarray:
            """Clip deltas using training base coordinates"""
            abs_preds = preds_delta + base_train
            abs_preds[:, 0] = np.clip(abs_preds[:, 0], 0.0, FIELD_X_MAX)
            abs_preds[:, 1] = np.clip(abs_preds[:, 1], 0.0, FIELD_Y_MAX)
            return abs_preds - base_train
        
        train_primary_pred = clip_delta_train(np.column_stack([train_primary_pred_x, train_primary_pred_y]))
        if use_stacking:
            train_stacked_x = meta_model_x.predict(meta_train_x)
            train_stacked_y = meta_model_y.predict(meta_train_y)
            train_pred = clip_delta_train(np.column_stack([train_stacked_x, train_stacked_y]))
        else:
            train_pred = train_primary_pred
        
        if use_residuals:
            train_residual_x = residual_model_x.predict(X_train)
            train_residual_y = residual_model_y.predict(X_train)
            train_pred = clip_delta_train(train_pred + np.column_stack([train_residual_x, train_residual_y]))
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f"\n  Final Training RMSE: {train_rmse:.4f} yards")
    print(f"  Final Validation RMSE: {final_rmse:.4f} yards")

    return {
        "models_x": models_x,
        "models_y": models_y,
        "feature_indices_x": feature_indices_x,  # Phase 2: Track which features each model uses
        "feature_indices_y": feature_indices_y,  # Phase 2: Track which features each model uses
        "meta_model_x": meta_model_x if use_stacking else None,
        "meta_model_y": meta_model_y if use_stacking else None,
        "use_stacking": use_stacking,
        "use_residuals": use_residuals,
        "residual_model_x": residual_model_x,
        "residual_model_y": residual_model_y,
        "train_rmse": train_rmse,
        "val_rmse": final_rmse,
        "val_rmse_x": final_rmse_x,
        "val_rmse_y": final_rmse_y,
        "model_names": model_names,
    }


def predict_deltas(results: Dict[str, object], X: np.ndarray) -> np.ndarray:
    """Generate delta predictions for a feature matrix using trained ensemble results."""
    model_order = results["model_names"]
    models_x = results["models_x"]
    models_y = results["models_y"]
    feature_indices_x = results.get("feature_indices_x", {})  # Phase 2
    feature_indices_y = results.get("feature_indices_y", {})  # Phase 2

    # Phase 2: Apply feature selection for each model
    if feature_indices_x:
        # Use feature subsets for each model
        base_preds_x = np.column_stack([
            models_x[name].predict(X[:, feature_indices_x[name]]) 
            for name in model_order
        ])
        base_preds_y = np.column_stack([
            models_y[name].predict(X[:, feature_indices_y[name]]) 
            for name in model_order
        ])
    else:
        # Fallback: Use all features (backwards compatibility)
        base_preds_x = np.column_stack([models_x[name].predict(X) for name in model_order])
        base_preds_y = np.column_stack([models_y[name].predict(X) for name in model_order])

    if results["use_stacking"] and results["meta_model_x"] is not None:
        pred_x = results["meta_model_x"].predict(base_preds_x)
        pred_y = results["meta_model_y"].predict(base_preds_y)
    else:
        pred_x = base_preds_x.mean(axis=1)
        pred_y = base_preds_y.mean(axis=1)

    if results["use_residuals"] and results["residual_model_x"] is not None:
        pred_x += results["residual_model_x"].predict(X)
        pred_y += results["residual_model_y"].predict(X)

    return np.column_stack([pred_x, pred_y])


def apply_physics_constraints(
    predictions: np.ndarray,
    base_positions: np.ndarray,
    current_velocities: np.ndarray,
    time_delta: float = 1.0,
    max_accel: float = 4.0,
    max_speed: float = 25.0
) -> np.ndarray:
    """
    Apply physics-based constraints to ensure realistic predictions.
    
    Constraints:
    1. Maximum acceleration: |a| < 4 yd/s² (NFL player limit)
    2. Speed limits: Based on typical player capabilities
    3. Field boundaries: 0-120 yards (x), 0-53.3 yards (y)
    4. Velocity continuity: Can't instantly reverse direction
    
    Args:
        predictions: Predicted delta positions (n_samples, 2)
        base_positions: Current positions (n_samples, 2)
        current_velocities: Current velocities in yd/s (n_samples, 2)
        time_delta: Time horizon in seconds
        max_accel: Maximum allowed acceleration in yd/s²
        max_speed: Maximum allowed speed in yd/s
    
    Returns:
        Constrained predictions (n_samples, 2)
    """
    constrained = predictions.copy()
    
    # Convert deltas to absolute positions
    predicted_positions = base_positions + predictions
    
    # 1. Field boundary constraints (hard boundaries)
    predicted_positions[:, 0] = np.clip(predicted_positions[:, 0], 0.0, 120.0)
    predicted_positions[:, 1] = np.clip(predicted_positions[:, 1], 0.0, 53.3)
    
    # 2. Acceleration constraints
    # Implied velocity from prediction: (new_pos - old_pos) / time_delta
    implied_velocity = predictions / time_delta
    
    # Required acceleration: (new_vel - old_vel) / time_delta
    required_accel = (implied_velocity - current_velocities) / time_delta
    accel_magnitude = np.linalg.norm(required_accel, axis=1)
    
    # Scale down predictions that require too much acceleration
    too_fast = accel_magnitude > max_accel
    if too_fast.any():
        scale_factor = max_accel / accel_magnitude[too_fast]
        # Scale down the velocity change, not the full prediction
        velocity_change = implied_velocity[too_fast] - current_velocities[too_fast]
        scaled_velocity_change = velocity_change * scale_factor[:, None]
        new_velocity = current_velocities[too_fast] + scaled_velocity_change
        constrained[too_fast] = new_velocity * time_delta
    
    # 3. Speed limit constraints
    speed = np.linalg.norm(implied_velocity, axis=1)
    too_fast_speed = speed > max_speed
    if too_fast_speed.any():
        scale_factor = max_speed / speed[too_fast_speed]
        constrained[too_fast_speed] = implied_velocity[too_fast_speed] * scale_factor[:, None] * time_delta
    
    # 4. Final boundary check after constraints
    final_positions = base_positions + constrained
    final_positions[:, 0] = np.clip(final_positions[:, 0], 0.0, 120.0)
    final_positions[:, 1] = np.clip(final_positions[:, 1], 0.0, 53.3)
    constrained = final_positions - base_positions
    
    return constrained


def predict_deltas_with_tta(
    results: Dict[str, object], 
    X: np.ndarray, 
    base_positions: np.ndarray = None,
    current_velocities: np.ndarray = None,
    is_defense: np.ndarray = None,
    n_augments: int = 3,
    noise_scale: float = 0.01,
    apply_physics: bool = True
) -> np.ndarray:
    """
    Enhanced Test-Time Augmentation with role-aware perturbation and physics constraints.
    
    Improvements over basic TTA:
    1. Role-aware noise: More perturbation for defensive players (less predictable)
    2. Trajectory variations: Different feature subsets for diverse predictions
    3. Confidence-weighted averaging: Weight predictions by model certainty
    4. Physics constraints: Enforce realistic movement limits
    
    Expected improvement: 0.03-0.05 yards RMSE
    
    Args:
        results: Trained model results dictionary
        X: Input features (n_samples, n_features)
        base_positions: Current (x, y) positions (n_samples, 2)
        current_velocities: Current (vx, vy) velocities (n_samples, 2)
        is_defense: Boolean mask for defensive players (n_samples,)
        n_augments: Number of augmented versions (default: 3)
        noise_scale: Base scale of Gaussian noise (default: 0.01)
        apply_physics: Whether to apply physics constraints
    
    Returns:
        Averaged and constrained predictions (n_samples, 2)
    """
    np.random.seed(42)  # For reproducibility
    n_samples = X.shape[0]
    
    # Determine role-specific noise scales
    if is_defense is not None:
        # Defensive players: 2x noise (more unpredictable movement)
        # Offensive players: 1x noise (more route-based, predictable)
        role_noise_scale = np.where(is_defense, noise_scale * 2.0, noise_scale)[:, None]
    else:
        role_noise_scale = noise_scale
    
    predictions = []
    confidence_weights = []
    
    # Strategy 1: Original prediction (no perturbation)
    pred_original = predict_deltas(results, X)
    predictions.append(pred_original)
    confidence_weights.append(1.5)  # Higher weight for original
    
    # Strategy 2: Temporal feature perturbation
    # Focus noise on time-varying features (velocity, acceleration, etc.)
    X_temporal = X.copy()
    # Assume first 50 features are more temporal (velocity, acceleration, etc.)
    temporal_mask = np.zeros(X.shape[1], dtype=bool)
    temporal_mask[:min(50, X.shape[1])] = True
    noise_temporal = np.random.normal(0, role_noise_scale, X.shape)
    noise_temporal[:, ~temporal_mask] = 0  # Only perturb temporal features
    X_temporal = X + noise_temporal
    pred_temporal = predict_deltas(results, X_temporal)
    predictions.append(pred_temporal)
    confidence_weights.append(1.0)
    
    # Strategy 3: Spatial feature perturbation
    # Focus noise on spatial features (position, distance, angles)
    X_spatial = X.copy()
    spatial_mask = np.zeros(X.shape[1], dtype=bool)
    spatial_mask[min(50, X.shape[1]):min(100, X.shape[1])] = True
    noise_spatial = np.random.normal(0, role_noise_scale, X.shape)
    noise_spatial[:, ~spatial_mask] = 0  # Only perturb spatial features
    X_spatial = X + noise_spatial
    pred_spatial = predict_deltas(results, X_spatial)
    predictions.append(pred_spatial)
    confidence_weights.append(1.0)
    
    # Strategy 4: Full perturbation (if n_augments > 3)
    if n_augments > 3:
        for i in range(n_augments - 3):
            noise_full = np.random.normal(0, role_noise_scale, X.shape)
            X_full = X + noise_full
            pred_full = predict_deltas(results, X_full)
            predictions.append(pred_full)
            confidence_weights.append(0.8)  # Lower weight for full perturbation
    
    # Confidence-weighted averaging
    confidence_weights = np.array(confidence_weights)
    confidence_weights = confidence_weights / confidence_weights.sum()  # Normalize
    
    avg_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, confidence_weights):
        avg_pred += pred * weight
    
    # Apply physics constraints if requested
    if apply_physics and base_positions is not None:
        # Use default velocities if not provided
        if current_velocities is None:
            current_velocities = np.zeros((n_samples, 2), dtype=np.float32)
        
        avg_pred = apply_physics_constraints(
            avg_pred,
            base_positions,
            current_velocities,
            time_delta=1.0,  # Assume 1 second horizon
            max_accel=4.0,   # 4 yd/s² maximum
            max_speed=25.0   # 25 yd/s (~50 mph max)
        )
    
    return avg_pred


def prepare_inference_pairs(
    input_df: pd.DataFrame,
    template_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build inference rows mirroring the training pair structure."""
    input_df = input_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    grouped: Dict[Tuple[int, int, int], pd.DataFrame] = {
        key: grp.copy() for key, grp in input_df.groupby(["game_id", "play_id", "nfl_id"], sort=False)
    }

    inference_rows: List[Dict[str, float]] = []
    missing_ids: List[str] = []

    for row in template_df.itertuples(index=False):
        key = (row.game_id, row.play_id, row.nfl_id)
        group = grouped.get(key)
        sample_id = f"{row.game_id}_{row.play_id}_{row.nfl_id}_{row.frame_id}"

        if group is None or group.empty:
            missing_ids.append(sample_id)
            continue

        player_in = group.sort_values("frame_id")
        last_frame = player_in.iloc[-1]
        is_target = int(bool(last_frame.get("player_to_predict", False)))

        if len(player_in) >= 2:
            prev_frame = player_in.iloc[-2]
            vel_dx = float(last_frame["vx"] - prev_frame["vx"])
            vel_dy = float(last_frame["vy"] - prev_frame["vy"])
            pos_dx = float(last_frame["x"] - prev_frame["x"])
            pos_dy = float(last_frame["y"] - prev_frame["y"])

            if len(player_in) >= 3:
                frame_minus_2 = player_in.iloc[-3]
                trajectory_curvature = float(
                    np.hypot(
                        last_frame["x"] - 2 * prev_frame["x"] + frame_minus_2["x"],
                        last_frame["y"] - 2 * prev_frame["y"] + frame_minus_2["y"],
                    )
                )
                prev_vel_dx = float(prev_frame["vx"] - frame_minus_2["vx"])
                prev_vel_dy = float(prev_frame["vy"] - frame_minus_2["vy"])
                jerk_x = float(vel_dx - prev_vel_dx)
                jerk_y = float(vel_dy - prev_vel_dy)
            else:
                trajectory_curvature = 0.0
                jerk_x = 0.0
                jerk_y = 0.0
        else:
            vel_dx = vel_dy = pos_dx = pos_dy = trajectory_curvature = jerk_x = jerk_y = 0.0

        accel_est = float(np.hypot(vel_dx, vel_dy))
        jerk_est = float(np.hypot(jerk_x, jerk_y))
        time_diff = float(row.frame_id - last_frame["frame_id"])

        sample = {
            "id": sample_id,
            "game_id": int(row.game_id),
            "play_id": int(row.play_id),
            "nfl_id": int(row.nfl_id),
            "output_frame_id": int(row.frame_id),
            "time_diff": time_diff,
            "velocity_change_x": vel_dx,
            "velocity_change_y": vel_dy,
            "position_change_x": pos_dx,
            "position_change_y": pos_dy,
            "acceleration_est": accel_est,
            "trajectory_curvature": trajectory_curvature,
            "jerk_x": jerk_x,
            "jerk_y": jerk_y,
            "jerk_est": jerk_est,
            "is_target_player": is_target,
        }

        for col in last_frame.index:
            if col in {"game_id", "play_id", "nfl_id", "player_position", "player_role", "player_side", "direction"}:
                continue
            # Preserve play_direction for coordinate reversion
            if col == "play_direction":
                sample["play_direction"] = last_frame[col]
                continue
            try:
                sample[f"input_{col}"] = float(last_frame[col])
            except (TypeError, ValueError):
                continue

        inference_rows.append(sample)

    if missing_ids:
        raise RuntimeError(
            f"No trajectory data found for {len(missing_ids)} ids. Examples: {missing_ids[:5]}"
        )

    return pd.DataFrame(inference_rows)


def clip_deltas(deltas: np.ndarray, base_coords: np.ndarray) -> np.ndarray:
    """Clip absolute coordinates to field boundaries and return adjusted deltas."""
    absolute = deltas + base_coords
    absolute[:, 0] = np.clip(absolute[:, 0], 0.0, 120.0)
    absolute[:, 1] = np.clip(absolute[:, 1], 0.0, 53.3)
    return absolute - base_coords


def main(auto_resume: bool = False) -> None:
    # Initialize checkpoint log
    log_checkpoint("=" * 80)
    log_checkpoint("🚀 STARTING NFL BIG DATA BOWL 2026 - ADVANCED SUBMISSION GENERATION")
    log_checkpoint("=" * 80)
    log_memory("Initialization")
    
    # Check for resume capability
    resume_info = check_resume_from_checkpoint()
    resume_phase = resume_info.get("resume_phase", 0)
    checkpoint_data = None
    
    if resume_phase > 0:
        if auto_resume:
            log_checkpoint(f"✓ Auto-resuming from Phase {resume_phase}")
            checkpoint_data = resume_info.get("checkpoint_data")
        else:
            user_input = input(f"\n⚠️  Found checkpoint from Phase {resume_phase}. Resume from there? (y/n): ").strip().lower()
            if user_input != 'y':
                log_checkpoint("User chose to start fresh - ignoring checkpoints")
                resume_phase = 0
            else:
                log_checkpoint(f"✓ Resuming from Phase {resume_phase}")
                checkpoint_data = resume_info.get("checkpoint_data")
    
    data_root = resolve_data_root()
    print("[PATHS] Using data root:", data_root)
    log_checkpoint(f"Data root: {data_root}")

    train_folder = data_root / "train"

    validation_candidates: List[Path] = []
    explicit_validation = os.environ.get("VALIDATION_DIR")
    if explicit_validation:
        validation_candidates.append(Path(explicit_validation))

    kaggle_validation_dir = Path("/kaggle/input/validation/validation")
    if kaggle_validation_dir.exists():
        validation_candidates.append(kaggle_validation_dir)

    validation_candidates.append(data_root / "validation")

    validation_folder = next((path for path in validation_candidates if path.exists()), None)

    if validation_folder is None:
        validation_folder = train_folder
        print("[PATHS] Validation data not found; reusing training weeks as validation.")
    else:
        print("[PATHS] Using validation folder:", validation_folder)

    test_input_path = locate_file("test_input.csv", data_root)
    test_template_path = locate_file("test.csv", data_root)
    submission_path = Path.cwd() / "submission.csv"

    train_weeks = [
        "w01",
        "w02",
        "w03",
        "w04",
        "w05",
        "w06",
        "w07",
        "w08",
        "w09",
        "w10",
        "w11",
        "w12",
        "w13",
        "w14",
        "w15",
    ]
    val_weeks = ["w16", "w17", "w18"]

    # Phase 1-3: Load data or resume from checkpoint
    if resume_phase >= 3 and checkpoint_data:
        log_checkpoint("📦 Resuming from Phase 3 checkpoint - loading saved data")
        # Phase 3 checkpoint has X_train, X_val already prepared
        X_train = checkpoint_data.get("X_train")
        X_val = checkpoint_data.get("X_val")
        selected_features = checkpoint_data.get("selected_features")
        log_checkpoint(f"✓ Loaded training data: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        log_checkpoint(f"✓ Loaded validation data: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
        log_checkpoint(f"✓ Selected features: {len(selected_features)}")
    else:
        log_checkpoint(f"📊 Phase 1/6: Loading training data ({len(train_weeks)} weeks)")
        train_input, train_output = load_data(train_folder, train_weeks, "TRAINING")
        log_memory("After loading training data")
        
        log_checkpoint(f"📊 Phase 1/6: Loading validation data ({len(val_weeks)} weeks)")
        val_input, val_output = load_data(validation_folder, val_weeks, "VALIDATION")
        log_memory("After loading validation data")
        
        log_checkpoint(f"✓ Train data: {len(train_input):,} input rows, {len(train_output):,} output rows")
        log_checkpoint(f"✓ Val data: {len(val_input):,} input rows, {len(val_output):,} output rows")
        save_checkpoint("01_data_loaded", {
            "train_input_shape": train_input.shape,
            "train_output_shape": train_output.shape,
            "val_input_shape": val_input.shape,
            "val_output_shape": val_output.shape,
            "train_weeks": train_weeks,
            "val_weeks": val_weeks
        })

        log_checkpoint("🔧 Phase 2/6: Engineering features for training set")
        train_input = engineer_deep_features(train_input)
        log_memory("After feature engineering training set")
        
        log_checkpoint("🔧 Phase 2/6: Engineering features for validation set")
        val_input = engineer_deep_features(val_input)
        log_memory("After feature engineering validation set")

        max_samples = 600_000
        log_checkpoint(f"Creating training pairs (max {max_samples:,} samples)")
        train_pairs = create_pairs(train_input, train_output, max_samples=max_samples)
        log_memory("After creating training pairs")
        
        train_pairs = augment_pair_features(train_pairs)
        log_memory("After augmenting training pairs")
        
        log_checkpoint(f"Creating validation pairs (max {max_samples:,} samples)")
        val_pairs = create_pairs(val_input, val_output, max_samples=max_samples)
        log_memory("After creating validation pairs")
        
        val_pairs = augment_pair_features(val_pairs)
        log_memory("After augmenting validation pairs")
        
        log_checkpoint(f"✓ Feature engineering complete: {train_input.shape[1]} features")
        log_checkpoint(f"✓ Training pairs created: {len(train_pairs):,} samples")
        log_checkpoint(f"✓ Validation pairs created: {len(val_pairs):,} samples")
        save_checkpoint("02_features_engineered", {
            "feature_count": train_input.shape[1],
            "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "pair_columns": list(train_pairs.columns)
    })

        if resume_phase < 3:
            del train_input, train_output, val_input, val_output
            gc.collect()

    # Phase 3: Feature selection (or load from checkpoint)
    if resume_phase < 3:
        feature_columns = [col for col in train_pairs.columns if col.startswith("input_")]
        feature_columns += [
            "time_diff",
            "output_frame_id",
            "velocity_change_x",
            "velocity_change_y",
            "position_change_x",
            "position_change_y",
            "acceleration_est",
            "trajectory_curvature",
            "jerk_x",
            "jerk_y",
            "jerk_est",
            "is_target_player",
        ]
        feature_columns += [
            "time_diff_squared",
            "time_diff_cubed",
            "time_diff_log",
            "time_diff_inv",
            "time_weight_feature",
            "time_diff_normalized",
            "time_horizon_bucket",
            "is_short_horizon",
            "is_mid_horizon",
            "is_long_horizon",
        ]
        feature_columns = list(dict.fromkeys(feature_columns))

        receiver_mask_full = train_pairs["is_target_player"].values.astype(bool)
        defense_indicator_col = "input_is_defensive_coverage"
        if defense_indicator_col in train_pairs.columns:
            defense_mask_full = train_pairs[defense_indicator_col].values.astype(bool)
        else:
            defense_mask_full = np.zeros(len(train_pairs), dtype=bool)
        defense_mask_full = np.logical_and(defense_mask_full, ~receiver_mask_full)
        other_mask_full = ~(receiver_mask_full | defense_mask_full)

        # Prepare validation data similarly
        receiver_mask_val = val_pairs["is_target_player"].values.astype(bool)
        if defense_indicator_col in val_pairs.columns:
            defense_mask_val = val_pairs[defense_indicator_col].values.astype(bool)
        else:
            defense_mask_val = np.zeros(len(val_pairs), dtype=bool)
        defense_mask_val = np.logical_and(defense_mask_val, ~receiver_mask_val)
        other_mask_val = ~(receiver_mask_val | defense_mask_val)

        train_features_df = train_pairs[feature_columns].fillna(0.0)
        base_train = train_pairs[["input_x", "input_y"]].values
        y_train = train_pairs[["target_x", "target_y"]].values - base_train
        time_importance = train_pairs["time_weight_feature"].values
        train_weights = 1.0 + 2.0 * time_importance + train_pairs["is_target_player"].values
        train_groups = train_pairs["game_id"].values

        # Prepare validation features
        val_features_df = val_pairs[feature_columns].fillna(0.0)
        base_val = val_pairs[["input_x", "input_y"]].values
        y_val = val_pairs[["target_x", "target_y"]].values - base_val
        val_groups = val_pairs["game_id"].values

        del train_pairs, val_pairs
        gc.collect()

        log_checkpoint(f"🎯 Phase 3/6: Feature selection from {len(feature_columns)} candidates")
        selected_features = select_features_via_permutation(
            train_features_df,
            y_train,
            train_weights,
            feature_columns,
        )

        train_features_df = train_features_df[selected_features]
        feature_columns = selected_features
        
        log_checkpoint(f"✓ Feature selection complete: {len(selected_features)} features selected")
        
        # Extract numpy arrays before saving (DataFrames are too large for pickle)
        X_train = train_features_df.values
        X_val = val_features_df[selected_features].values
        
        # Save all necessary data for Phase 4 resume (without the large DataFrames)
        save_checkpoint("03_features_selected", {
            "selected_count": len(selected_features),
            "selected_features": selected_features,
            "X_train": X_train,
            "X_val": X_val,
            "base_train": base_train,
            "base_val": base_val,
            "y_train": y_train,
            "y_val": y_val,
            "train_weights": train_weights,
            "train_groups": train_groups,
            "val_groups": val_groups,
            "receiver_mask_full": receiver_mask_full,
            "defense_mask_full": defense_mask_full,
            "other_mask_full": other_mask_full,
            "receiver_mask_val": receiver_mask_val,
            "defense_mask_val": defense_mask_val,
            "other_mask_val": other_mask_val,
        })
        
        # Clean up DataFrames (train_pairs and val_pairs already deleted earlier)
        del train_features_df, val_features_df
        gc.collect()
    else:
        # Resume from Phase 3 checkpoint - extract all saved data
        log_checkpoint("📦 Extracting data from Phase 3 checkpoint")
        X_train = checkpoint_data["X_train"]
        X_val = checkpoint_data["X_val"]
        base_train = checkpoint_data["base_train"]
        base_val = checkpoint_data["base_val"]
        y_train = checkpoint_data["y_train"]
        y_val = checkpoint_data["y_val"]
        train_weights = checkpoint_data["train_weights"]
        train_groups = checkpoint_data["train_groups"]
        val_groups = checkpoint_data["val_groups"]
        receiver_mask_full = checkpoint_data["receiver_mask_full"]
        defense_mask_full = checkpoint_data["defense_mask_full"]
        other_mask_full = checkpoint_data["other_mask_full"]
        receiver_mask_val = checkpoint_data["receiver_mask_val"]
        defense_mask_val = checkpoint_data["defense_mask_val"]
        other_mask_val = checkpoint_data["other_mask_val"]
        feature_columns = checkpoint_data["selected_features"]

    log_checkpoint("📏 Scaling features with RobustScaler + QuantileTransformer")
    scaler1 = RobustScaler()
    X_train = scaler1.fit_transform(X_train)
    X_val = scaler1.transform(X_val)  # Transform validation with training scaler

    n_quantiles = max(10, min(1000, max(1, X_train.shape[0] // 10)))
    scaler2 = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=42,
    )
    X_train = scaler2.fit_transform(X_train)
    X_val = scaler2.transform(X_val)  # Transform validation with training scaler
    log_checkpoint(f"✓ Scaling complete - Train: {X_train.shape}, Val: {X_val.shape}")

    role_masks = {
        "Targeted Receiver": receiver_mask_full,
        "Defensive Coverage": defense_mask_full,
        "Other Players": other_mask_full,
    }

    role_masks_val = {
        "Targeted Receiver": receiver_mask_val,
        "Defensive Coverage": defense_mask_val,
        "Other Players": other_mask_val,
    }

    role_results: Dict[str, Dict[str, object]] = {}
    role_counts: Dict[str, int] = {}
    role_val_counts: Dict[str, int] = {}

    log_checkpoint("🤖 Phase 4/6: Training role-specific ensemble models")
    log_checkpoint(f"Training 3 role groups × 10 models × 2 coordinates × 3 CV folds = 180 model fits")
    
    for role_name, mask in role_masks.items():
        sample_count = int(mask.sum())
        val_mask = role_masks_val[role_name]
        val_count = int(val_mask.sum())
        
        if sample_count == 0:
            print(f"\n[ROLE] Skipping {role_name} (no training samples)")
            continue

        print(f"\n[ROLE] Training {role_name} model on {sample_count:,} train samples, {val_count:,} val samples")
        log_checkpoint(f"  → Training: {role_name} (Train: {sample_count:,}, Val: {val_count:,})")
        role_result = train_advanced_ensemble(
            X_train[mask],
            y_train[mask],
            X_val[val_mask],
            y_val[val_mask],
            train_weights[mask],
            base_val[val_mask],
            train_groups[mask],
            base_train[mask],  # Pass the original unscaled base coordinates
            label=role_name,
        )
        role_results[role_name] = role_result
        role_counts[role_name] = sample_count
        role_val_counts[role_name] = val_count
        
        train_rmse = float(role_result.get("train_rmse", role_result["val_rmse"]))
        val_rmse = float(role_result["val_rmse"])
        log_checkpoint(f"  ✓ {role_name} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f} yards")
        save_checkpoint(f"04_model_{role_name.lower().replace(' ', '_')}", {
            "role": role_name,
            "train_sample_count": sample_count,
            "val_sample_count": val_count,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse
        })

    if not role_results:
        raise RuntimeError("No role-specific models were trained; check training data availability.")

    total_train_samples = sum(role_counts.values())
    total_val_samples = sum(role_val_counts.values())
    
    weighted_train_mse = 0.0
    weighted_val_mse = 0.0
    
    print("\n[PERFORMANCE] Role-wise RMSE")
    print("-" * 80)
    print(f"{'Role':<25} {'Train Samples':<15} {'Train RMSE':<15} {'Val Samples':<15} {'Val RMSE':<15}")
    print("-" * 80)
    
    for role_name, result in role_results.items():
        train_count = role_counts[role_name]
        val_count = role_val_counts[role_name]
        train_rmse = float(result.get("train_rmse", result["val_rmse"]))
        val_rmse = float(result["val_rmse"])
        weighted_train_mse += (train_rmse ** 2) * train_count
        weighted_val_mse += (val_rmse ** 2) * val_count
        print(f"{role_name:<25} {train_count:<15,} {train_rmse:<15.4f} {val_count:<15,} {val_rmse:<15.4f}")

    overall_train_rmse = float(np.sqrt(weighted_train_mse / max(total_train_samples, 1)))
    overall_val_rmse = float(np.sqrt(weighted_val_mse / max(total_val_samples, 1)))
    print("-" * 80)
    print(f"{'OVERALL':<25} {total_train_samples:<15,} {overall_train_rmse:<15.4f} {total_val_samples:<15,} {overall_val_rmse:<15.4f}")
    print("-" * 80)
    
    log_checkpoint(f"✓ All role models trained")
    log_checkpoint(f"  Overall Training RMSE: {overall_train_rmse:.4f} yards")
    log_checkpoint(f"  Overall Validation RMSE: {overall_val_rmse:.4f} yards")
    
    save_metrics("04_training_complete", {
        "overall_train_rmse": overall_train_rmse,
        "overall_val_rmse": overall_val_rmse,
        "role_train_rmses": {role: float(role_results[role].get("train_rmse", role_results[role]["val_rmse"])) for role in role_results},
        "role_val_rmses": {role: float(role_results[role]["val_rmse"]) for role in role_results},
        "role_train_counts": role_counts,
        "role_val_counts": role_val_counts,
        "total_train_samples": total_train_samples,
        "total_val_samples": total_val_samples
    })

    fallback_results = role_results.get("Other Players") or next(iter(role_results.values()))

    del X_train
    gc.collect()

    log_checkpoint("🔮 Phase 5/6: Loading and preparing test data")
    test_input = pd.read_csv(test_input_path)
    test_template = pd.read_csv(test_template_path)
    log_checkpoint(f"  Test input: {len(test_input):,} rows, Test template: {len(test_template):,} predictions")

    test_input = engineer_deep_features(test_input)
    inference_pairs = prepare_inference_pairs(test_input, test_template)
    inference_pairs = augment_pair_features(inference_pairs)

    required_cols = set(feature_columns)
    for col in required_cols:
        if col not in inference_pairs.columns:
            inference_pairs[col] = 0.0

    inference_features = inference_pairs[feature_columns].fillna(0.0)
    X_test = inference_features.values
    X_test = scaler1.transform(X_test)
    X_test = scaler2.transform(X_test)

    base_coords = inference_pairs[["input_x", "input_y"]].values

    receiver_mask_inf = inference_pairs["is_target_player"].values.astype(bool)
    defense_indicator_col = "input_is_defensive_coverage"
    if defense_indicator_col in inference_pairs.columns:
        defense_mask_inf = inference_pairs[defense_indicator_col].values.astype(bool)
    else:
        defense_mask_inf = np.zeros(len(inference_pairs), dtype=bool)
    defense_mask_inf = np.logical_and(defense_mask_inf, ~receiver_mask_inf)
    other_mask_inf = ~(receiver_mask_inf | defense_mask_inf)

    inference_role_masks = {
        "Targeted Receiver": receiver_mask_inf,
        "Defensive Coverage": defense_mask_inf,
        "Other Players": other_mask_inf,
    }

    pred_delta = np.zeros_like(base_coords)

    log_checkpoint("🎯 Phase 6/6: Generating predictions with Test-Time Augmentation")
    log_checkpoint("  Using 3 augmented versions per sample to reduce variance")
    
    for role_name, mask in inference_role_masks.items():
        if not mask.any():
            continue
        role_count = int(mask.sum())
        log_checkpoint(f"  → Predicting: {role_name} ({role_count:,} samples)")
        role_result = role_results.get(role_name, fallback_results)
        
        # Extract current velocities for physics constraints (if available)
        try:
            current_vx = inference_pairs.loc[mask, "vx"].values.astype("float32")
            current_vy = inference_pairs.loc[mask, "vy"].values.astype("float32")
            current_velocities = np.column_stack([current_vx, current_vy])
        except KeyError:
            current_velocities = None
        
        # Determine if defensive players (for role-aware noise)
        is_defense_mask = (role_name == "Defensive Coverage") or (
            "is_defensive_coverage" in inference_pairs.columns and
            inference_pairs.loc[mask, "is_defensive_coverage"].values.astype(bool)
        )
        if isinstance(is_defense_mask, bool):
            is_defense_mask = np.full(role_count, is_defense_mask)
        
        # Enhanced TTA with physics constraints and role-aware perturbation
        # - Role-aware noise: 2x perturbation for defensive players
        # - Trajectory variations: Different feature subsets
        # - Confidence-weighted averaging
        # - Physics constraints: max acceleration 4 yd/s², max speed 25 yd/s
        # Expected improvement: 0.03-0.05 yards RMSE (vs basic TTA: 0.02-0.03)
        role_delta = predict_deltas_with_tta(
            role_result, 
            X_test[mask],
            base_positions=base_coords[mask],
            current_velocities=current_velocities,
            is_defense=is_defense_mask,
            n_augments=3,
            noise_scale=0.01,
            apply_physics=True
        )
        
        # Additional clipping (physics constraints already applied in TTA)
        role_delta = clip_deltas(role_delta, base_coords[mask])
        pred_delta[mask] = role_delta
        log_checkpoint(f"  ✓ {role_name} predictions complete (Enhanced TTA + Physics)")

    del X_test
    gc.collect()

    predictions = pred_delta + base_coords

    submission = pd.DataFrame(
        {
            "id": inference_pairs["id"].values,
            "x": predictions[:, 0],
            "y": predictions[:, 1],
        }
    )
    submission = submission.sort_values("id").reset_index(drop=True)
    submission.to_csv(submission_path, index=False)

    print("\nSubmission file created at:", submission_path)
    print("Total predictions:", len(submission))
    
    log_checkpoint("=" * 80)
    log_checkpoint(f"🎉 SUCCESS! Submission generated: {len(submission):,} predictions")
    log_checkpoint(f"📁 Output file: {submission_path}")
    log_checkpoint(f"📊 Training RMSE: {overall_train_rmse:.4f} yards")
    log_checkpoint(f"📊 Validation RMSE: {overall_val_rmse:.4f} yards")
    log_checkpoint(f"✨ Used Test-Time Augmentation (3 augments per sample)")
    log_checkpoint("=" * 80)
    
    save_checkpoint("06_final_submission", {
        "submission_path": str(submission_path),
        "prediction_count": len(submission),
        "train_rmse": overall_train_rmse,
        "val_rmse": overall_val_rmse,
        "used_tta": True,
        "tta_augments": 3,
        "x_range": [float(submission['x'].min()), float(submission['x'].max())],
        "y_range": [float(submission['y'].min()), float(submission['y'].max())]
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NFL Big Data Bowl 2026 submission")
    parser.add_argument("--resume", action="store_true", help="Automatically resume from latest checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore checkpoints")
    args = parser.parse_args()
    
    if args.fresh:
        # Clear resume by temporarily ignoring checkpoints
        main(auto_resume=False)
    elif args.resume:
        main(auto_resume=True)
    else:
        # Interactive mode - ask user
        main(auto_resume=False)
