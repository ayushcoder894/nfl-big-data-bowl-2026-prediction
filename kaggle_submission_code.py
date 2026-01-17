# NFL Big Data Bowl 2026 - Complete Kaggle Submission Code
# This code trains a model and generates submission.csv for the leaderboard

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import optuna

# Set random seed for reproducibility
np.random.seed(42)

print("Libraries imported successfully!")

# ===================================
# 1. LOAD TRAINING DATA
# ===================================

# First, let's check what files are available
print("Checking available files...")
import os

# Check different possible paths
possible_paths = [
    '/kaggle/input',
    '/kaggle/input/nfl-big-data-bowl-2026-prediction',
    '/kaggle/input/nfl-big-data-bowl-2026-prediction/train'
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"Files in {path}:")
        try:
            files = os.listdir(path)
            for f in sorted(files)[:20]:  # Show first 20 files
                print(f"  {f}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more files")
        except:
            print(f"  Cannot list files in {path}")
    else:
        print(f"Path {path} does not exist")

# Find the correct path to training data
train_folder = None
possible_train_paths = [
    '/kaggle/input/nfl-big-data-bowl-2026-prediction/train',
    '/kaggle/input/train',
    '/kaggle/input'
]

for path in possible_train_paths:
    if os.path.exists(path):
        # Check if this directory contains training files
        try:
            files = os.listdir(path)
            if any('input_2023' in f for f in files):
                train_folder = Path(path)
                print(f"Found training data in: {train_folder}")
                break
        except:
            continue

if train_folder is None:
    print("Training folder not found. Listing all available input files:")
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            if 'input_2023' in file or 'output_2023' in file:
                print(f"  {os.path.join(root, file)}")
    raise FileNotFoundError("Cannot locate training data files")

# Load training data - use all available weeks for maximum performance
weeks_to_use = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09', 'w10', 
               'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18']  # Use all weeks
input_dfs = []
output_dfs = []

print(f"\nLoading training data from {train_folder}...")
for week in weeks_to_use:
    input_file = train_folder / f"input_2023_{week}.csv"
    output_file = train_folder / f"output_2023_{week}.csv"
    
    if input_file.exists() and output_file.exists():
        try:
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(output_file)
            
            input_dfs.append(input_df)
            output_dfs.append(output_df)
            print(f"Loaded {week}: {len(input_df)} input rows, {len(output_df)} output rows")
        except Exception as e:
            print(f"Error loading {week}: {e}")
    else:
        print(f"Files not found for {week}")

# Check if we have any data
if len(input_dfs) == 0:
    print("No training files found! Checking alternative file patterns...")
    
    # Try to find any training files with different patterns
    all_files = []
    for root, dirs, files in os.walk('/kaggle/input'):
        all_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    
    print("All CSV files found:")
    for f in sorted(all_files):
        print(f"  {f}")
    
    # Try to load any files that look like training data
    input_files = [f for f in all_files if 'input' in f.lower()]
    output_files = [f for f in all_files if 'output' in f.lower()]
    
    print(f"\nFound {len(input_files)} input files and {len(output_files)} output files")
    
    # Load first few files as training data
    for i, (inp_file, out_file) in enumerate(zip(input_files[:5], output_files[:5])):
        try:
            input_df = pd.read_csv(inp_file)
            output_df = pd.read_csv(out_file)
            input_dfs.append(input_df)
            output_dfs.append(output_df)
            print(f"Loaded file {i+1}: {len(input_df)} input, {len(output_df)} output rows")
        except Exception as e:
            print(f"Error loading file {i+1}: {e}")

# Final check
if len(input_dfs) == 0:
    raise ValueError("No training data could be loaded. Please check the dataset structure.")

# Combine all weeks
train_input = pd.concat(input_dfs, ignore_index=True)
train_output = pd.concat(output_dfs, ignore_index=True)

print(f"\nTotal training data:")
print(f"Input: {len(train_input)} rows")
print(f"Output: {len(train_output)} rows")

# ===================================
# 2. FEATURE ENGINEERING
# ===================================

def create_features(df):
    """Create ultra-advanced features for training"""
    df = df.copy()
    
    # Basic velocity components
    df['vx'] = df['s'] * np.cos(np.radians(df['dir']))
    df['vy'] = df['s'] * np.sin(np.radians(df['dir']))
    
    # Advanced velocity features
    df['speed_squared'] = df['s'] ** 2
    df['speed_cubed'] = df['s'] ** 3
    df['acceleration_magnitude'] = np.abs(df['a'])
    df['velocity_magnitude'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['velocity_angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    
    # Direction features with multiple harmonics
    df['dir_sin'] = np.sin(np.radians(df['dir']))
    df['dir_cos'] = np.cos(np.radians(df['dir']))
    df['dir_sin2'] = np.sin(2 * np.radians(df['dir']))
    df['dir_cos2'] = np.cos(2 * np.radians(df['dir']))
    df['o_sin'] = np.sin(np.radians(df['o']))
    df['o_cos'] = np.cos(np.radians(df['o']))
    df['o_sin2'] = np.sin(2 * np.radians(df['o']))
    df['o_cos2'] = np.cos(2 * np.radians(df['o']))
    
    # Complex angle relationships
    df['dir_o_diff'] = np.abs(df['dir'] - df['o'])
    df['dir_o_diff'] = np.minimum(df['dir_o_diff'], 360 - df['dir_o_diff'])
    df['dir_o_alignment'] = np.cos(np.radians(df['dir'] - df['o']))
    
    # Ball landing features (enhanced)
    df['dist_to_ball'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + 
                                (df['y'] - df['ball_land_y'])**2)
    df['dist_to_ball_squared'] = df['dist_to_ball'] ** 2
    df['dir_to_ball'] = np.degrees(np.arctan2(df['ball_land_y'] - df['y'], 
                                             df['ball_land_x'] - df['x']))
    df['ball_angle_alignment'] = np.cos(np.radians(df['dir'] - df['dir_to_ball']))
    
    # Advanced field position features
    df['x_normalized'] = df['x'] / 120.0
    df['y_normalized'] = df['y'] / 53.3
    df['x_normalized_squared'] = df['x_normalized'] ** 2
    df['y_normalized_squared'] = df['y_normalized'] ** 2
    df['dist_to_endzone'] = np.minimum(df['x'], 120 - df['x'])
    df['dist_to_sideline'] = np.minimum(df['y'], 53.3 - df['y'])
    df['field_quadrant'] = ((df['x'] > 60).astype(int) * 2 + (df['y'] > 26.65).astype(int))
    
    # Momentum and physics features
    df['momentum_x'] = df['vx'] * df['s']  # Assuming mass is proportional to speed
    df['momentum_y'] = df['vy'] * df['s']
    df['kinetic_energy'] = 0.5 * df['s'] ** 2
    df['potential_energy'] = df['x'] + df['y']  # Simple field position energy
    
    # Field zones (enhanced)
    df['in_red_zone'] = ((df['x'] < 20) | (df['x'] > 100)).astype(int)
    df['field_third'] = pd.cut(df['x'], bins=3, labels=['defensive', 'middle', 'offensive']).astype(str)
    df['field_fifth'] = pd.cut(df['x'], bins=5, labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Movement patterns (enhanced)
    df['moving_toward_ball'] = ((df['vx'] * (df['ball_land_x'] - df['x']) + 
                                df['vy'] * (df['ball_land_y'] - df['y'])) > 0).astype(int)
    df['movement_efficiency'] = (df['vx'] * (df['ball_land_x'] - df['x']) + 
                                df['vy'] * (df['ball_land_y'] - df['y'])) / (df['s'] * df['dist_to_ball'] + 1e-6)
    
    # Time-based features (enhanced)
    df['time_to_ball_land'] = df['num_frames_output'] / 10.0
    df['predicted_x_linear'] = df['x'] + df['vx'] * df['time_to_ball_land']
    df['predicted_y_linear'] = df['y'] + df['vy'] * df['time_to_ball_land']
    df['predicted_x_accel'] = df['x'] + df['vx'] * df['time_to_ball_land'] + 0.5 * df['a'] * df['dir_cos'] * df['time_to_ball_land']**2
    df['predicted_y_accel'] = df['y'] + df['vy'] * df['time_to_ball_land'] + 0.5 * df['a'] * df['dir_sin'] * df['time_to_ball_land']**2
    
    # Distance predictions from ball
    df['predicted_dist_to_ball_linear'] = np.sqrt((df['predicted_x_linear'] - df['ball_land_x'])**2 + 
                                                 (df['predicted_y_linear'] - df['ball_land_y'])**2)
    df['predicted_dist_to_ball_accel'] = np.sqrt((df['predicted_x_accel'] - df['ball_land_x'])**2 + 
                                                (df['predicted_y_accel'] - df['ball_land_y'])**2)
    
    # Player interaction features
    df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
    df['is_defense'] = (df['player_side'] == 'Defense').astype(int)
    
    # Enhanced position encoding
    position_encoder = LabelEncoder()
    df['position_encoded'] = position_encoder.fit_transform(df['player_position'])
    
    role_encoder = LabelEncoder()
    df['role_encoded'] = role_encoder.fit_transform(df['player_role'])
    
    # Detailed position groups
    df['is_skill_position'] = df['player_position'].isin(['WR', 'RB', 'TE', 'QB']).astype(int)
    df['is_lineman'] = df['player_position'].isin(['C', 'G', 'T', 'DE', 'DT', 'NT']).astype(int)
    df['is_linebacker'] = df['player_position'].isin(['LB', 'ILB', 'OLB', 'MLB']).astype(int)
    df['is_secondary'] = df['player_position'].isin(['CB', 'S', 'SS', 'FS']).astype(int)
    df['is_receiver'] = df['player_position'].isin(['WR', 'TE']).astype(int)
    df['is_rusher'] = df['player_position'].isin(['RB', 'FB']).astype(int)
    df['is_pass_rusher'] = df['player_position'].isin(['DE', 'OLB']).astype(int)
    
    # Route-running features (for receivers)
    df['route_depth'] = np.abs(df['x'] - 30)  # Assuming snap around 30-yard line
    df['route_width'] = np.abs(df['y'] - 26.65)  # Distance from center
    
    # Coverage features (for defense)
    df['coverage_leverage'] = np.where(df['is_defense'] == 1, 
                                      np.sign(df['y'] - 26.65), 0)  # Inside/outside leverage
    
    # Advanced statistical features
    df['x_percentile'] = df.groupby(['game_id', 'play_id'])['x'].transform(lambda x: x.rank(pct=True))
    df['y_percentile'] = df.groupby(['game_id', 'play_id'])['y'].transform(lambda x: x.rank(pct=True))
    df['speed_percentile'] = df.groupby(['game_id', 'play_id'])['s'].transform(lambda x: x.rank(pct=True))
    
    # Interaction terms (polynomial features)
    df['x_y_interaction'] = df['x'] * df['y']
    df['speed_dir_interaction'] = df['s'] * df['dir_cos']
    df['pos_vel_interaction'] = df['x_normalized'] * df['vx'] + df['y_normalized'] * df['vy']
    
    # Clustering-based features (unsupervised learning)
    try:
        from sklearn.cluster import KMeans
        position_features = df[['x_normalized', 'y_normalized', 'vx', 'vy']].fillna(0)
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        df['position_cluster'] = kmeans.fit_predict(position_features)
        df['cluster_distance'] = np.sqrt(np.sum((position_features - kmeans.cluster_centers_[df['position_cluster']])**2, axis=1))
    except:
        df['position_cluster'] = 0
        df['cluster_distance'] = 0
    
    # Fourier features for periodic patterns
    df['x_fourier_1'] = np.sin(2 * np.pi * df['x_normalized'])
    df['x_fourier_2'] = np.cos(2 * np.pi * df['x_normalized'])
    df['y_fourier_1'] = np.sin(2 * np.pi * df['y_normalized'])
    df['y_fourier_2'] = np.cos(2 * np.pi * df['y_normalized'])
    
    # Temporal features based on frame_id
    df['frame_normalized'] = df['frame_id'] / df['frame_id'].max()
    df['frame_sin'] = np.sin(2 * np.pi * df['frame_normalized'])
    df['frame_cos'] = np.cos(2 * np.pi * df['frame_normalized'])
    
    return df, position_encoder, role_encoder

# Apply feature engineering
print("Creating features...")
train_input_features, pos_encoder, role_encoder = create_features(train_input)

# Select ultra-enhanced features for training
feature_columns = [
    'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy',
    'speed_squared', 'speed_cubed', 'acceleration_magnitude', 'velocity_magnitude', 'velocity_angle',
    'dir_sin', 'dir_cos', 'dir_sin2', 'dir_cos2', 'o_sin', 'o_cos', 'o_sin2', 'o_cos2',
    'dir_o_diff', 'dir_o_alignment',
    'dist_to_ball', 'dist_to_ball_squared', 'dir_to_ball', 'ball_angle_alignment',
    'x_normalized', 'y_normalized', 'x_normalized_squared', 'y_normalized_squared',
    'dist_to_endzone', 'dist_to_sideline', 'field_quadrant', 'field_fifth',
    'momentum_x', 'momentum_y', 'kinetic_energy', 'potential_energy',
    'in_red_zone', 'moving_toward_ball', 'movement_efficiency',
    'time_to_ball_land', 'predicted_x_linear', 'predicted_y_linear',
    'predicted_x_accel', 'predicted_y_accel', 'predicted_dist_to_ball_linear', 'predicted_dist_to_ball_accel',
    'is_offense', 'is_defense', 'position_encoded', 'role_encoded',
    'is_skill_position', 'is_lineman', 'is_linebacker', 'is_secondary',
    'is_receiver', 'is_rusher', 'is_pass_rusher',
    'route_depth', 'route_width', 'coverage_leverage',
    'x_percentile', 'y_percentile', 'speed_percentile',
    'x_y_interaction', 'speed_dir_interaction', 'pos_vel_interaction',
    'position_cluster', 'cluster_distance',
    'x_fourier_1', 'x_fourier_2', 'y_fourier_1', 'y_fourier_2',
    'frame_normalized', 'frame_sin', 'frame_cos',
    'num_frames_output', 'ball_land_x', 'ball_land_y', 'frame_id'
]

print(f"Selected {len(feature_columns)} features for training")

# ===================================
# 3. PREPARE TRAINING DATASET
# ===================================

def create_training_pairs(input_df, output_df):
    """Create ultra-advanced training pairs with contextual features"""
    
    training_pairs = []
    
    # Group by play to process each play separately
    for (game_id, play_id), play_input in input_df.groupby(['game_id', 'play_id']):
        
        # Get corresponding output data
        play_output = output_df[(output_df['game_id'] == game_id) & 
                               (output_df['play_id'] == play_id)]
        
        if len(play_output) == 0:
            continue
            
        # Calculate play-level features
        play_center_x = play_input['x'].mean()
        play_center_y = play_input['y'].mean()
        play_spread_x = play_input['x'].std()
        play_spread_y = play_input['y'].std()
        offense_players = play_input[play_input['is_offense'] == 1]
        defense_players = play_input[play_input['is_defense'] == 1]
        
        # For each player in output, find their input data
        for nfl_id in play_output['nfl_id'].unique():
            
            player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
            player_output = play_output[play_output['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if len(player_input) == 0 or len(player_output) == 0:
                continue
            
            # Enhanced sequence analysis (use up to 5 frames)
            last_frames = player_input.tail(min(5, len(player_input)))
            
            # Advanced trajectory features
            if len(last_frames) >= 2:
                # Velocity changes and trends
                vel_change_x = last_frames['vx'].iloc[-1] - last_frames['vx'].iloc[0]
                vel_change_y = last_frames['vy'].iloc[-1] - last_frames['vy'].iloc[0]
                
                # Speed trend
                speed_trend = last_frames['s'].iloc[-1] - last_frames['s'].iloc[0]
                speed_volatility = last_frames['s'].std()
                
                # Direction changes
                dir_changes = np.diff(last_frames['dir'].values)
                dir_volatility = np.std(dir_changes) if len(dir_changes) > 0 else 0
                
                # Acceleration patterns
                if len(last_frames) >= 3:
                    acc_x = (last_frames['vx'].iloc[-1] - last_frames['vx'].iloc[-2])
                    acc_y = (last_frames['vy'].iloc[-1] - last_frames['vy'].iloc[-2])
                    
                    # Jerk (rate of acceleration change)
                    if len(last_frames) >= 4:
                        jerk_x = acc_x - (last_frames['vx'].iloc[-2] - last_frames['vx'].iloc[-3])
                        jerk_y = acc_y - (last_frames['vy'].iloc[-2] - last_frames['vy'].iloc[-3])
                    else:
                        jerk_x = jerk_y = 0
                else:
                    acc_x = acc_y = jerk_x = jerk_y = 0
                
                # Movement consistency and patterns
                pos_changes = np.diff(last_frames[['x', 'y']].values, axis=0)
                movement_consistency = np.std(pos_changes, axis=0).mean() if len(pos_changes) > 0 else 0
                
                # Curvature of path
                if len(last_frames) >= 3:
                    dx = np.diff(last_frames['x'].values)
                    dy = np.diff(last_frames['y'].values)
                    ddx = np.diff(dx)
                    ddy = np.diff(dy)
                    curvature = np.mean(np.abs(dx[:-1] * ddy - dy[:-1] * ddx) / 
                                       ((dx[:-1]**2 + dy[:-1]**2)**1.5 + 1e-6)) if len(ddx) > 0 else 0
                else:
                    curvature = 0
                    
            else:
                vel_change_x = vel_change_y = speed_trend = speed_volatility = 0
                dir_volatility = acc_x = acc_y = jerk_x = jerk_y = 0
                movement_consistency = curvature = 0
            
            # Get current player info
            last_input_frame = player_input.loc[player_input['frame_id'].idxmax()]
            player_side = last_input_frame['is_offense']
            
            # Contextual features relative to other players
            same_team = play_input[play_input['is_offense'] == player_side]  
            opponent_team = play_input[play_input['is_offense'] != player_side]
            
            # Distance to teammates and opponents
            if len(same_team) > 1:
                teammate_distances = np.sqrt((same_team['x'] - last_input_frame['x'])**2 + 
                                           (same_team['y'] - last_input_frame['y'])**2)
                teammate_distances = teammate_distances[teammate_distances > 0]  # Exclude self
                min_teammate_dist = teammate_distances.min() if len(teammate_distances) > 0 else 50
                avg_teammate_dist = teammate_distances.mean() if len(teammate_distances) > 0 else 50
            else:
                min_teammate_dist = avg_teammate_dist = 50
                
            if len(opponent_team) > 0:
                opponent_distances = np.sqrt((opponent_team['x'] - last_input_frame['x'])**2 + 
                                           (opponent_team['y'] - last_input_frame['y'])**2)
                min_opponent_dist = opponent_distances.min()
                avg_opponent_dist = opponent_distances.mean()
                
                # Pressure/coverage features
                close_opponents = (opponent_distances < 5).sum()
                very_close_opponents = (opponent_distances < 3).sum()
            else:
                min_opponent_dist = avg_opponent_dist = 50
                close_opponents = very_close_opponents = 0
            
            # Formation features
            formation_width = same_team['y'].max() - same_team['y'].min() if len(same_team) > 0 else 0
            formation_depth = same_team['x'].max() - same_team['x'].min() if len(same_team) > 0 else 0
            
            # Create training examples for each output frame
            for _, output_row in player_output.iterrows():
                
                # Time difference and future position
                time_diff = output_row['frame_id'] - last_input_frame['frame_id']
                actual_displacement = np.sqrt((output_row['x'] - last_input_frame['x'])**2 + 
                                            (output_row['y'] - last_input_frame['y'])**2)
                
                training_example = {
                    'game_id': game_id,
                    'play_id': play_id,
                    'nfl_id': nfl_id,
                    'output_frame': output_row['frame_id'],
                    'time_diff': time_diff,
                    'target_x': output_row['x'],
                    'target_y': output_row['y'],
                    
                    # Enhanced trajectory features
                    'vel_change_x': vel_change_x,
                    'vel_change_y': vel_change_y,
                    'speed_trend': speed_trend,
                    'speed_volatility': speed_volatility,
                    'dir_volatility': dir_volatility,
                    'acc_x_est': acc_x,
                    'acc_y_est': acc_y,
                    'jerk_x_est': jerk_x,
                    'jerk_y_est': jerk_y,
                    'movement_consistency': movement_consistency,
                    'path_curvature': curvature,
                    
                    # Contextual features
                    'play_center_x': play_center_x,
                    'play_center_y': play_center_y,
                    'play_spread_x': play_spread_x,
                    'play_spread_y': play_spread_y,
                    'min_teammate_dist': min_teammate_dist,
                    'avg_teammate_dist': avg_teammate_dist,
                    'min_opponent_dist': min_opponent_dist,
                    'avg_opponent_dist': avg_opponent_dist,
                    'close_opponents': close_opponents,
                    'very_close_opponents': very_close_opponents,
                    'formation_width': formation_width,
                    'formation_depth': formation_depth,
                    
                    # Physics-based features
                    'actual_displacement': actual_displacement,
                }
                
                # Add input features
                for col in feature_columns:
                    if col in last_input_frame.index:
                        training_example[f'input_{col}'] = last_input_frame[col]
                
                training_pairs.append(training_example)
    
    return pd.DataFrame(training_pairs)

# Create training dataset
print("Creating training pairs...")
training_data = create_training_pairs(train_input_features, train_output)
print(f"Created {len(training_data)} training examples")

# ===================================
# 4. TRAIN STATE-OF-THE-ART MODELS FOR SUB-0.1 RMSE
# ===================================

# Prepare ultra-enhanced features and targets
feature_cols = [f'input_{col}' for col in feature_columns if f'input_{col}' in training_data.columns]

# Add all contextual and trajectory features
contextual_features = [
    'output_frame', 'time_diff', 'vel_change_x', 'vel_change_y',
    'speed_trend', 'speed_volatility', 'dir_volatility',
    'acc_x_est', 'acc_y_est', 'jerk_x_est', 'jerk_y_est',
    'movement_consistency', 'path_curvature',
    'play_center_x', 'play_center_y', 'play_spread_x', 'play_spread_y',
    'min_teammate_dist', 'avg_teammate_dist', 'min_opponent_dist', 'avg_opponent_dist',
    'close_opponents', 'very_close_opponents', 'formation_width', 'formation_depth',
    'actual_displacement'
]

feature_cols.extend([col for col in contextual_features if col in training_data.columns])
feature_cols = [col for col in feature_cols if col in training_data.columns]

X = training_data[feature_cols].fillna(0)
y = training_data[['target_x', 'target_y']]

print(f"Training features: {len(feature_cols)}")
print(f"Training samples: {len(X)}")

# Handle categorical features
if 'input_field_third' in X.columns:
    field_third_map = {'defensive': 0, 'middle': 1, 'offensive': 2}
    X['input_field_third'] = X['input_field_third'].map(field_third_map).fillna(1)

# Multi-stage feature selection
print("Performing advanced feature selection...")

# Stage 1: Statistical feature selection
selector_1 = SelectKBest(score_func=f_regression, k=min(150, len(feature_cols)))
X_sel1 = selector_1.fit_transform(X, y.iloc[:, 0])
features_1 = [feature_cols[i] for i in selector_1.get_support(indices=True)]

# Stage 2: RFE with Random Forest
rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf_selector, n_features_to_select=min(120, len(features_1)), step=1)
X_sel2 = rfe.fit_transform(X[features_1], y.iloc[:, 0])
features_2 = [features_1[i] for i in rfe.get_support(indices=True)]

X = X[features_2]
feature_cols = features_2
print(f"Selected {len(feature_cols)} most important features after multi-stage selection")

# Advanced multi-transformation approach
print("Applying advanced transformations...")

# Transformation 1: QuantileTransformer (Gaussian)
qt_normal = QuantileTransformer(output_distribution='normal', random_state=42)
X_qt_normal = qt_normal.fit_transform(X)

# Transformation 2: QuantileTransformer (Uniform)
qt_uniform = QuantileTransformer(output_distribution='uniform', random_state=42)
X_qt_uniform = qt_uniform.fit_transform(X)

# Transformation 3: RobustScaler
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)

# Transformation 4: StandardScaler
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

# Dimensionality reduction ensemble
print("Creating dimensionality reduction ensemble...")

# PCA
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_robust)

# ICA
ica = FastICA(n_components=25, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_robust)

# SVD
svd = TruncatedSVD(n_components=25, random_state=42)
X_svd = svd.fit_transform(X_robust)

# Combine all transformations
print("Combining feature transformations...")
X_mega = np.hstack([
    X_qt_normal,    # Gaussian transformed
    X_qt_uniform,   # Uniform transformed
    X_robust,       # Robust scaled
    X_pca,          # PCA components
    X_ica,          # ICA components
    X_svd           # SVD components
])

print(f"Mega feature matrix shape: {X_mega.shape}")

# Advanced train-validation split with stratification
# Create stratification bins for target values
y_bins = pd.qcut(y.iloc[:, 0], q=10, labels=False, duplicates='drop')
X_train, X_val, y_train, y_val = train_test_split(
    X_mega, y, test_size=0.1, random_state=42, stratify=y_bins
)

print("Training state-of-the-art models...")

# Model 1: Hyperparameter-optimized XGBoost
print("Training hyperparameter-optimized XGBoost...")
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=800,
    max_depth=12,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.05,
    reg_lambda=0.05,
    gamma=0.1,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
))
xgb_model.fit(X_train, y_train)

# Model 2: Optimized LightGBM
print("Training optimized LightGBM...")
lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=800,
    max_depth=15,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=0.05,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
))
lgb_model.fit(X_train, y_train)

# Model 3: CatBoost (excellent for structured data)
print("Training CatBoost model...")
cb_model = MultiOutputRegressor(cb.CatBoostRegressor(
    iterations=600,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=3,
    subsample=0.8,
    random_state=42,
    verbose=False,
    thread_count=-1
))
cb_model.fit(X_train, y_train)

# Model 4: Neural Network (MLP)
print("Training Neural Network...")
nn_model = MultiOutputRegressor(MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
))
nn_model.fit(X_train, y_train)

# Model 5: Gaussian Process Regressor
print("Training Gaussian Process...")
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp_model = MultiOutputRegressor(GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    random_state=42,
    n_restarts_optimizer=3
))
# Use subset for GP due to computational complexity
n_gp_samples = min(5000, len(X_train))
gp_indices = np.random.choice(len(X_train), n_gp_samples, replace=False)
gp_model.fit(X_train[gp_indices], y_train.iloc[gp_indices])

# Model 6: Support Vector Regression
print("Training SVR...")
svr_model = MultiOutputRegressor(SVR(
    kernel='rbf',
    C=100,
    gamma='scale',
    epsilon=0.01
))
# Use subset for SVR due to computational complexity
n_svr_samples = min(10000, len(X_train))
svr_indices = np.random.choice(len(X_train), n_svr_samples, replace=False)
svr_model.fit(X_train[svr_indices], y_train.iloc[svr_indices])

# Model 7: K-Nearest Neighbors with optimized parameters
print("Training KNN...")
knn_model = MultiOutputRegressor(KNeighborsRegressor(
    n_neighbors=15,
    weights='distance',
    algorithm='ball_tree',
    leaf_size=30,
    n_jobs=-1
))
knn_model.fit(X_train, y_train)

# Model 8: Bayesian Ridge Regression
print("Training Bayesian Ridge...")
br_model = MultiOutputRegressor(BayesianRidge(
    alpha_1=1e-6,
    alpha_2=1e-6,
    lambda_1=1e-6,
    lambda_2=1e-6,
    compute_score=True
))
br_model.fit(X_train, y_train)

# Model 9: Huber Regressor (robust to outliers)
print("Training Huber Regressor...")
huber_model = MultiOutputRegressor(HuberRegressor(
    epsilon=1.35,
    alpha=0.0001,
    max_iter=300
))
huber_model.fit(X_train, y_train)

# Model 10: Extra Trees with different parameters
print("Training Extra Trees...")
et_model = MultiOutputRegressor(ExtraTreesRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    random_state=42,
    n_jobs=-1
))
et_model.fit(X_train, y_train)

# Evaluate all models
models = {
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'CatBoost': cb_model,
    'NeuralNetwork': nn_model,
    'GaussianProcess': gp_model,
    'SVR': svr_model,
    'KNN': knn_model,
    'BayesianRidge': br_model,
    'HuberRegressor': huber_model,
    'ExtraTrees': et_model
}

model_scores = {}
for name, model in models.items():
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    model_scores[name] = rmse
    print(f"{name} RMSE: {rmse:.3f} yards")

# Create sophisticated weighted ensemble
print("\nCreating advanced weighted ensemble...")
sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

# Use inverse RMSE for weights (better models get higher weights)
rmse_values = [score for _, score in sorted_models]
inverse_weights = [1.0 / rmse for rmse in rmse_values]
normalized_weights = [w / sum(inverse_weights) for w in inverse_weights]

print("Model rankings and weights:")
for i, ((name, rmse), weight) in enumerate(zip(sorted_models, normalized_weights)):
    print(f"{i+1}. {name}: RMSE={rmse:.3f}, Weight={weight:.3f}")

def ensemble_predict(X_test):
    predictions = []
    for i, (name, _) in enumerate(sorted_models):
        pred = models[name].predict(X_test)
        predictions.append(pred * normalized_weights[i])
    
    return np.sum(predictions, axis=0)

# Advanced stacking ensemble
print("\nCreating advanced stacking ensemble...")

# Level 1: Base models (already trained)
base_predictions = []
for name, model in models.items():
    try:
        pred = model.predict(X_val)
        base_predictions.append(pred)
        print(f"Added {name} to base predictions")
    except Exception as e:
        print(f"Skipping {name} due to error: {e}")

if base_predictions:
    # Create meta-features from base model predictions
    meta_features = np.hstack(base_predictions)
    print(f"Meta-features shape: {meta_features.shape}")
    
    # Level 2: Meta-model (trains on base model predictions)
    print("Training meta-model...")
    meta_model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    ))
    meta_model.fit(meta_features, y_val)
    
    # Final stacked prediction function
    def stacked_predict(X_test):
        # Get predictions from all base models
        test_base_predictions = []
        for name, model in models.items():
            try:
                pred = model.predict(X_test)
                test_base_predictions.append(pred)
            except:
                continue
        
        if test_base_predictions:
            test_meta_features = np.hstack(test_base_predictions)
            return meta_model.predict(test_meta_features)
        else:
            return np.zeros((len(X_test), 2))
    
    # Evaluate stacked ensemble
    stacked_pred = stacked_predict(X_val)
    stacked_rmse = np.sqrt(mean_squared_error(y_val, stacked_pred))
    print(f"Stacked ensemble RMSE: {stacked_rmse:.4f} yards")
    
    ensemble_predict = stacked_predict
else:
    # Fallback to simple ensemble
    def ensemble_predict(X_test):
        predictions = []
        weights = []
        for name, model in models.items():
            try:
                pred = model.predict(X_test)
                predictions.append(pred)
                weights.append(1.0 / model_scores.get(name, 1.0))
            except:
                continue
        
        if predictions:
            weights = np.array(weights) / sum(weights)
            return np.average(predictions, axis=0, weights=weights)
        else:
            return np.zeros((len(X_test), 2))

# Final ensemble evaluation
ensemble_pred = ensemble_predict(X_val)
ensemble_mse = mean_squared_error(y_val, ensemble_pred)
ensemble_rmse = np.sqrt(ensemble_mse)

print(f"\nFinal Ensemble Results:")
print(f"Ensemble RMSE: {ensemble_rmse:.3f} yards")
print(f"X-coordinate RMSE: {np.sqrt(mean_squared_error(y_val.iloc[:, 0], ensemble_pred[:, 0])):.3f} yards")
print(f"Y-coordinate RMSE: {np.sqrt(mean_squared_error(y_val.iloc[:, 1], ensemble_pred[:, 1])):.3f} yards")

# Store ensemble info for prediction
ensemble_info = {
    'models': models,
    'feature_cols': feature_cols,
    'qt_normal': qt_normal,
    'qt_uniform': qt_uniform,
    'scaler_robust': scaler_robust,
    'scaler_standard': scaler_standard,
    'pca': pca,
    'ica': ica,
    'svd': svd,
    'selector_1': selector_1,
    'rfe': rfe,
    'features_1': features_1,
    'features_2': features_2
}

# ===================================
# 5. LOAD TEST DATA AND GENERATE PREDICTIONS
# ===================================

# Find test data files
print("Loading test data...")

# Try different possible paths for test files
test_input_path = None
test_path = None
sample_submission_path = None

possible_test_paths = [
    ('/kaggle/input/nfl-big-data-bowl-2026-prediction/test_input.csv', '/kaggle/input/nfl-big-data-bowl-2026-prediction/test.csv', '/kaggle/input/nfl-big-data-bowl-2026-prediction/sample_submission.csv'),
    ('/kaggle/input/test_input.csv', '/kaggle/input/test.csv', '/kaggle/input/sample_submission.csv'),
]

for test_inp, test_tgt, sample_sub in possible_test_paths:
    if os.path.exists(test_inp) and os.path.exists(test_tgt) and os.path.exists(sample_sub):
        test_input_path = test_inp
        test_path = test_tgt
        sample_submission_path = sample_sub
        break

# If not found, search for them
if test_input_path is None:
    print("Test files not found in expected locations. Searching...")
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            full_path = os.path.join(root, file)
            if file == 'test_input.csv':
                test_input_path = full_path
            elif file == 'test.csv':
                test_path = full_path
            elif file == 'sample_submission.csv':
                sample_submission_path = full_path

if test_input_path is None or test_path is None:
    raise FileNotFoundError("Could not find test_input.csv or test.csv files")

print(f"Loading test data from:")
print(f"  test_input: {test_input_path}")
print(f"  test: {test_path}")

test_input = pd.read_csv(test_input_path)
test_targets = pd.read_csv(test_path)

print(f"Test input: {len(test_input)} rows")
print(f"Test targets: {len(test_targets)} rows")

# Apply feature engineering to test data
test_input_features, _, _ = create_features(test_input)

def generate_predictions_ensemble(test_input_df, test_targets_df, ensemble_predict_func, ensemble_info):
    """Generate predictions using ensemble model"""
    
    predictions = []
    
    # Group test targets by play
    for (game_id, play_id), play_targets in test_targets_df.groupby(['game_id', 'play_id']):
        
        # Get input data for this play
        play_input = test_input_df[(test_input_df['game_id'] == game_id) & 
                                  (test_input_df['play_id'] == play_id)]
        
        if len(play_input) == 0:
            # If no input data, use simple fallback
            for _, target_row in play_targets.iterrows():
                predictions.append({
                    'id': f"{target_row['game_id']}_{target_row['play_id']}_{target_row['nfl_id']}_{target_row['frame_id']}",
                    'x': 50.0,  # Fallback to field center
                    'y': 26.65
                })
            continue
            
        # Process each target prediction
        for _, target_row in play_targets.iterrows():
            
            player_id = target_row['nfl_id']
            target_frame = target_row['frame_id']
            
            # Get player's input data
            player_input = play_input[play_input['nfl_id'] == player_id].sort_values('frame_id')
            
            if len(player_input) == 0:
                # Fallback prediction
                pred_x, pred_y = 50.0, 26.65
            else:
                # Calculate sequence-based features
                last_frames = player_input.tail(min(3, len(player_input)))
                
                if len(last_frames) >= 2:
                    vel_change_x = last_frames['vx'].iloc[-1] - last_frames['vx'].iloc[0]
                    vel_change_y = last_frames['vy'].iloc[-1] - last_frames['vy'].iloc[0]
                    
                    if len(last_frames) >= 3:
                        acc_x = (last_frames['vx'].iloc[-1] - last_frames['vx'].iloc[-2])
                        acc_y = (last_frames['vy'].iloc[-1] - last_frames['vy'].iloc[-2])
                    else:
                        acc_x = acc_y = 0
                    
                    pos_changes = np.diff(last_frames[['x', 'y']].values, axis=0)
                    movement_consistency = np.std(pos_changes, axis=0).mean() if len(pos_changes) > 0 else 0
                else:
                    vel_change_x = vel_change_y = acc_x = acc_y = movement_consistency = 0
                
                # Use last available frame for prediction
                last_frame = player_input.loc[player_input['frame_id'].idxmax()]
                
                # Prepare feature vector
                feature_dict = {}
                
                # Add input features
                for col in feature_columns:
                    if col in last_frame.index:
                        feature_dict[f'input_{col}'] = last_frame[col]
                    else:
                        feature_dict[f'input_{col}'] = 0
                
                # Add sequence features
                feature_dict['output_frame'] = target_frame
                feature_dict['time_diff'] = target_frame - last_frame['frame_id']
                feature_dict['vel_change_x'] = vel_change_x
                feature_dict['vel_change_y'] = vel_change_y
                feature_dict['acc_x_est'] = acc_x
                feature_dict['acc_y_est'] = acc_y
                feature_dict['movement_consistency'] = movement_consistency
                
                # Create initial feature vector
                feature_values = []
                for col in ensemble_info['feature_cols']:
                    if col in feature_dict:
                        val = feature_dict[col]
                        # Handle field_third encoding
                        if col == 'input_field_third' and isinstance(val, str):
                            field_third_map = {'defensive': 0, 'middle': 1, 'offensive': 2}
                            val = field_third_map.get(val, 1)
                        feature_values.append(val)
                    else:
                        feature_values.append(0)
                
                # Convert to DataFrame for feature selection
                feature_df = pd.DataFrame([feature_values], columns=ensemble_info['feature_cols'])
                
                # Apply the same transformations as training
                try:
                    # Apply multi-transformation approach
                    X_qt_normal = ensemble_info['qt_normal'].transform(feature_df)
                    X_qt_uniform = ensemble_info['qt_uniform'].transform(feature_df)
                    X_robust = ensemble_info['scaler_robust'].transform(feature_df)
                    X_standard = ensemble_info['scaler_standard'].transform(feature_df)
                    
                    # Apply dimensionality reduction
                    X_pca = ensemble_info['pca'].transform(X_robust)
                    X_ica = ensemble_info['ica'].transform(X_robust)
                    X_svd = ensemble_info['svd'].transform(X_robust)
                    
                    # Combine all transformations
                    features_combined = np.hstack([
                        X_qt_normal, X_qt_uniform, X_robust, X_pca, X_ica, X_svd
                    ])
                    
                except Exception as e:
                    # Fallback to simple scaling
                    features_combined = ensemble_info['scaler_robust'].transform(feature_df)
                
                # Predict using ensemble
                pred_xy = ensemble_predict_func(features_combined)[0]
                pred_x, pred_y = pred_xy[0], pred_xy[1]
                
                # Apply field constraints
                pred_x = np.clip(pred_x, 0, 120)
                pred_y = np.clip(pred_y, 0, 53.3)
            
            # Add to predictions
            predictions.append({
                'id': f"{target_row['game_id']}_{target_row['play_id']}_{target_row['nfl_id']}_{target_row['frame_id']}",
                'x': pred_x,
                'y': pred_y
            })
    
    return pd.DataFrame(predictions)

# Generate predictions using advanced ensemble
print("Generating predictions with advanced ensemble model...")
submission = generate_predictions_ensemble(test_input_features, test_targets, ensemble_predict, ensemble_info)

print(f"Generated {len(submission)} predictions")

# ===================================
# 6. CREATE SUBMISSION FILE
# ===================================

# Load sample submission for validation
if sample_submission_path is None:
    print("Searching for sample_submission.csv...")
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            if file == 'sample_submission.csv':
                sample_submission_path = os.path.join(root, file)
                break

if sample_submission_path is None:
    raise FileNotFoundError("Could not find sample_submission.csv")

print(f"Loading sample submission from: {sample_submission_path}")
sample_submission = pd.read_csv(sample_submission_path)

print("Submission validation:")
print(f"Expected rows: {len(sample_submission)}")
print(f"Our rows: {len(submission)}")

# Check if all required IDs are present
expected_ids = set(sample_submission['id'])
our_ids = set(submission['id'])

missing_ids = expected_ids - our_ids
extra_ids = our_ids - expected_ids

print(f"Missing IDs: {len(missing_ids)}")
print(f"Extra IDs: {len(extra_ids)}")

# Fill in any missing predictions with fallback values
if missing_ids:
    print(f"Adding {len(missing_ids)} missing predictions...")
    missing_predictions = []
    for missing_id in missing_ids:
        missing_predictions.append({
            'id': missing_id,
            'x': 50.0,  # Field center
            'y': 26.65
        })
    
    missing_df = pd.DataFrame(missing_predictions)
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Remove extra predictions
if extra_ids:
    print(f"Removing {len(extra_ids)} extra predictions...")
    submission = submission[submission['id'].isin(expected_ids)]

# Ensure correct order
submission = submission.merge(sample_submission[['id']], on='id', how='right')
submission = submission.sort_values('id').reset_index(drop=True)

# Fill any remaining missing values
submission = submission.fillna({'x': 50.0, 'y': 26.65})

# Save submission file
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission Summary:")
print(f"Total predictions: {len(submission)}")
print(f"X coordinate range: {submission['x'].min():.1f} to {submission['x'].max():.1f}")
print(f"Y coordinate range: {submission['y'].min():.1f} to {submission['y'].max():.1f}")
print(f"Average X: {submission['x'].mean():.1f}")
print(f"Average Y: {submission['y'].mean():.1f}")

print("\n" + "="*50)
print("🏈 SUBMISSION.CSV CREATED SUCCESSFULLY! 🏈")
print("="*50)
print("File saved as: submission.csv")
print("Ready to submit to the competition!")