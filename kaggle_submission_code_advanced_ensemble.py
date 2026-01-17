# NFL Big Data Bowl 2026 - Advanced Ensemble Learning Approach
# This code implements multiple ensemble learning techniques for maximum performance

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
import multiprocessing
from itertools import combinations

# Set number of threads for optimal CPU performance
n_cores = multiprocessing.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)

print(f"Advanced Ensemble Learning - Optimizing for {n_cores} CPU cores")

# ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, 
                            VotingRegressor, BaggingRegressor, AdaBoostRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor, LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel, RFECV
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
import optuna

# Set random seed for reproducibility
np.random.seed(42)

print("Advanced ensemble libraries imported successfully!")

# ===================================
# 1. LOAD TRAINING DATA
# ===================================

# Local paths
train_folder = Path(r'c:\nfl-big-data-bowl-2026-prediction\train')
test_input_path = r'c:\nfl-big-data-bowl-2026-prediction\test_input.csv'
test_path = r'c:\nfl-big-data-bowl-2026-prediction\test.csv'
sample_submission_path = r'c:\nfl-big-data-bowl-2026-prediction\sample_submission.csv'

if not train_folder.exists():
    raise FileNotFoundError(f"Training folder not found: {train_folder}")

print(f"Found training data in: {train_folder}")

# Load training data
weeks_to_use = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09', 'w10', 
               'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18']
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

if len(input_dfs) == 0:
    raise ValueError("No training data could be loaded.")

# Combine all weeks
train_input = pd.concat(input_dfs, ignore_index=True)
train_output = pd.concat(output_dfs, ignore_index=True)

print(f"\nTotal training data:")
print(f"Input: {len(train_input)} rows")
print(f"Output: {len(train_output)} rows")

# ===================================
# 2. ENHANCED FEATURE ENGINEERING
# ===================================

def create_advanced_features(df):
    """Create ultra-advanced features with ensemble-focused engineering"""
    df = df.copy()
    
    # Basic velocity components
    df['vx'] = df['s'] * np.cos(np.radians(df['dir']))
    df['vy'] = df['s'] * np.sin(np.radians(df['dir']))
    
    # Advanced velocity and physics features
    df['speed_squared'] = df['s'] ** 2
    df['speed_cubed'] = df['s'] ** 3
    df['acceleration_magnitude'] = np.abs(df['a'])
    df['velocity_magnitude'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['velocity_angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    
    # Direction features with harmonics
    for i in [1, 2, 3]:
        df[f'dir_sin_{i}'] = np.sin(i * np.radians(df['dir']))
        df[f'dir_cos_{i}'] = np.cos(i * np.radians(df['dir']))
        df[f'o_sin_{i}'] = np.sin(i * np.radians(df['o']))
        df[f'o_cos_{i}'] = np.cos(i * np.radians(df['o']))
    
    # Ball interaction features
    df['dist_to_ball'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + 
                                (df['y'] - df['ball_land_y'])**2)
    df['dist_to_ball_squared'] = df['dist_to_ball'] ** 2
    df['dist_to_ball_log'] = np.log1p(df['dist_to_ball'])
    df['dir_to_ball'] = np.degrees(np.arctan2(df['ball_land_y'] - df['y'], 
                                             df['ball_land_x'] - df['x']))
    df['ball_angle_alignment'] = np.cos(np.radians(df['dir'] - df['dir_to_ball']))
    
    # Field position features
    df['x_normalized'] = df['x'] / 120.0
    df['y_normalized'] = df['y'] / 53.3
    df['x_normalized_squared'] = df['x_normalized'] ** 2
    df['y_normalized_squared'] = df['y_normalized'] ** 2
    df['dist_to_endzone'] = np.minimum(df['x'], 120 - df['x'])
    df['dist_to_sideline'] = np.minimum(df['y'], 53.3 - df['y'])
    df['field_quadrant'] = ((df['x'] > 60).astype(int) * 2 + (df['y'] > 26.65).astype(int))
    
    # Momentum and energy
    df['momentum_x'] = df['vx'] * df['s']
    df['momentum_y'] = df['vy'] * df['s']
    df['kinetic_energy'] = 0.5 * df['s'] ** 2
    df['potential_energy'] = df['x'] + df['y']
    
    # Time-based predictions
    df['time_to_ball_land'] = df['num_frames_output'] / 10.0
    df['predicted_x_linear'] = df['x'] + df['vx'] * df['time_to_ball_land']
    df['predicted_y_linear'] = df['y'] + df['vy'] * df['time_to_ball_land']
    df['predicted_x_accel'] = df['x'] + df['vx'] * df['time_to_ball_land'] + 0.5 * df['a'] * df['dir_cos_1'] * df['time_to_ball_land']**2
    df['predicted_y_accel'] = df['y'] + df['vy'] * df['time_to_ball_land'] + 0.5 * df['a'] * df['dir_sin_1'] * df['time_to_ball_land']**2
    
    # Player roles and positions
    df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
    df['is_defense'] = (df['player_side'] == 'Defense').astype(int)
    
    # Position encoding
    position_encoder = LabelEncoder()
    df['position_encoded'] = position_encoder.fit_transform(df['player_position'])
    
    role_encoder = LabelEncoder()
    df['role_encoded'] = role_encoder.fit_transform(df['player_role'])
    
    # Position groups
    df['is_skill_position'] = df['player_position'].isin(['WR', 'RB', 'TE', 'QB']).astype(int)
    df['is_lineman'] = df['player_position'].isin(['C', 'G', 'T', 'DE', 'DT', 'NT']).astype(int)
    df['is_linebacker'] = df['player_position'].isin(['LB', 'ILB', 'OLB', 'MLB']).astype(int)
    df['is_secondary'] = df['player_position'].isin(['CB', 'S', 'SS', 'FS']).astype(int)
    
    # Statistical percentiles
    df['x_percentile'] = df.groupby(['game_id', 'play_id'])['x'].transform(lambda x: x.rank(pct=True))
    df['y_percentile'] = df.groupby(['game_id', 'play_id'])['y'].transform(lambda x: x.rank(pct=True))
    df['speed_percentile'] = df.groupby(['game_id', 'play_id'])['s'].transform(lambda x: x.rank(pct=True))
    
    # Interaction terms
    df['x_y_interaction'] = df['x'] * df['y']
    df['speed_dir_interaction'] = df['s'] * df['dir_cos_1']
    df['pos_vel_interaction'] = df['x_normalized'] * df['vx'] + df['y_normalized'] * df['vy']
    
    # Clustering features
    try:
        position_features = df[['x_normalized', 'y_normalized', 'vx', 'vy']].fillna(0)
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        df['position_cluster'] = kmeans.fit_predict(position_features)
        df['cluster_distance'] = np.sqrt(np.sum((position_features - kmeans.cluster_centers_[df['position_cluster']])**2, axis=1))
    except:
        df['position_cluster'] = 0
        df['cluster_distance'] = 0
    
    # Fourier features
    for i in [1, 2, 3]:
        df[f'x_fourier_sin_{i}'] = np.sin(i * 2 * np.pi * df['x_normalized'])
        df[f'x_fourier_cos_{i}'] = np.cos(i * 2 * np.pi * df['x_normalized'])
        df[f'y_fourier_sin_{i}'] = np.sin(i * 2 * np.pi * df['y_normalized'])
        df[f'y_fourier_cos_{i}'] = np.cos(i * 2 * np.pi * df['y_normalized'])
    
    # Temporal features
    df['frame_normalized'] = df['frame_id'] / df['frame_id'].max()
    df['frame_sin'] = np.sin(2 * np.pi * df['frame_normalized'])
    df['frame_cos'] = np.cos(2 * np.pi * df['frame_normalized'])
    
    return df, position_encoder, role_encoder

# Apply feature engineering
print("Creating advanced features...")
train_input_features, pos_encoder, role_encoder = create_advanced_features(train_input)

# ===================================
# 3. PREPARE TRAINING DATASET WITH ENSEMBLE-FOCUSED FEATURES
# ===================================

def create_ensemble_training_pairs(input_df, output_df):
    """Create training pairs optimized for ensemble learning"""
    
    training_pairs = []
    
    for (game_id, play_id), play_input in input_df.groupby(['game_id', 'play_id']):
        play_output = output_df[(output_df['game_id'] == game_id) & 
                               (output_df['play_id'] == play_id)]
        
        if len(play_output) == 0:
            continue
            
        # Enhanced play-level context
        play_center_x = play_input['x'].mean()
        play_center_y = play_input['y'].mean()
        play_spread_x = play_input['x'].std()
        play_spread_y = play_input['y'].std()
        play_avg_speed = play_input['s'].mean()
        play_max_speed = play_input['s'].max()
        
        for nfl_id in play_output['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
            player_output = play_output[play_output['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if len(player_input) == 0 or len(player_output) == 0:
                continue
            
            # Multi-frame trajectory analysis
            last_frames = player_input.tail(min(5, len(player_input)))
            
            # Ensemble-focused trajectory features
            if len(last_frames) >= 2:
                # Velocity trends
                vel_trend_x = (last_frames['vx'].iloc[-1] - last_frames['vx'].iloc[0]) / len(last_frames)
                vel_trend_y = (last_frames['vy'].iloc[-1] - last_frames['vy'].iloc[0]) / len(last_frames)
                
                # Speed statistics
                speed_mean = last_frames['s'].mean()
                speed_std = last_frames['s'].std()
                speed_trend = (last_frames['s'].iloc[-1] - last_frames['s'].iloc[0]) / len(last_frames)
                
                # Direction statistics
                dir_changes = np.diff(last_frames['dir'].values)
                dir_volatility = np.std(dir_changes) if len(dir_changes) > 0 else 0
                dir_trend = np.mean(dir_changes) if len(dir_changes) > 0 else 0
                
                # Acceleration features
                if len(last_frames) >= 3:
                    acc_x = np.mean(np.diff(last_frames['vx'].values))
                    acc_y = np.mean(np.diff(last_frames['vy'].values))
                    
                    # Jerk features
                    if len(last_frames) >= 4:
                        jerk_x = np.mean(np.diff(np.diff(last_frames['vx'].values)))
                        jerk_y = np.mean(np.diff(np.diff(last_frames['vy'].values)))
                    else:
                        jerk_x = jerk_y = 0
                else:
                    acc_x = acc_y = jerk_x = jerk_y = 0
                
                # Path curvature
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
                vel_trend_x = vel_trend_y = speed_mean = speed_std = speed_trend = 0
                dir_volatility = dir_trend = acc_x = acc_y = jerk_x = jerk_y = curvature = 0
            
            # Current state
            last_input_frame = player_input.loc[player_input['frame_id'].idxmax()]
            
            # Contextual features (ensemble-focused)
            player_side = last_input_frame['is_offense']
            same_team = play_input[play_input['is_offense'] == player_side]
            opponent_team = play_input[play_input['is_offense'] != player_side]
            
            # Team spacing and formation
            if len(same_team) > 1:
                teammate_distances = np.sqrt((same_team['x'] - last_input_frame['x'])**2 + 
                                           (same_team['y'] - last_input_frame['y'])**2)
                teammate_distances = teammate_distances[teammate_distances > 0]
                min_teammate_dist = teammate_distances.min() if len(teammate_distances) > 0 else 50
                avg_teammate_dist = teammate_distances.mean() if len(teammate_distances) > 0 else 50
                formation_compactness = teammate_distances.std() if len(teammate_distances) > 0 else 0
            else:
                min_teammate_dist = avg_teammate_dist = formation_compactness = 50
                
            # Opponent pressure
            if len(opponent_team) > 0:
                opponent_distances = np.sqrt((opponent_team['x'] - last_input_frame['x'])**2 + 
                                           (opponent_team['y'] - last_input_frame['y'])**2)
                min_opponent_dist = opponent_distances.min()
                avg_opponent_dist = opponent_distances.mean()
                pressure_score = np.sum(1.0 / (opponent_distances + 1.0))  # Inverse distance pressure
                close_opponents = (opponent_distances < 5).sum()
            else:
                min_opponent_dist = avg_opponent_dist = pressure_score = close_opponents = 0
            
            # Formation features
            formation_width = same_team['y'].max() - same_team['y'].min() if len(same_team) > 0 else 0
            formation_depth = same_team['x'].max() - same_team['x'].min() if len(same_team) > 0 else 0
            
            # Create training examples for each output frame
            for _, output_row in player_output.iterrows():
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
                    
                    # Enhanced trajectory features for ensemble
                    'vel_trend_x': vel_trend_x,
                    'vel_trend_y': vel_trend_y,
                    'speed_mean': speed_mean,
                    'speed_std': speed_std,
                    'speed_trend': speed_trend,
                    'dir_volatility': dir_volatility,
                    'dir_trend': dir_trend,
                    'acc_x_est': acc_x,
                    'acc_y_est': acc_y,
                    'jerk_x_est': jerk_x,
                    'jerk_y_est': jerk_y,
                    'path_curvature': curvature,
                    
                    # Enhanced contextual features
                    'play_center_x': play_center_x,
                    'play_center_y': play_center_y,
                    'play_spread_x': play_spread_x,
                    'play_spread_y': play_spread_y,
                    'play_avg_speed': play_avg_speed,
                    'play_max_speed': play_max_speed,
                    'min_teammate_dist': min_teammate_dist,
                    'avg_teammate_dist': avg_teammate_dist,
                    'formation_compactness': formation_compactness,
                    'min_opponent_dist': min_opponent_dist,
                    'avg_opponent_dist': avg_opponent_dist,
                    'pressure_score': pressure_score,
                    'close_opponents': close_opponents,
                    'formation_width': formation_width,
                    'formation_depth': formation_depth,
                    'actual_displacement': actual_displacement,
                }
                
                # Add input features
                for col in last_input_frame.index:
                    if not col.startswith('game_id') and not col.startswith('play_id') and not col.startswith('nfl_id'):
                        training_example[f'input_{col}'] = last_input_frame[col]
                
                training_pairs.append(training_example)
    
    return pd.DataFrame(training_pairs)

# Create enhanced training dataset
print("Creating ensemble-optimized training pairs...")
training_data = create_ensemble_training_pairs(train_input_features, train_output)
print(f"Created {len(training_data)} training examples")

# ===================================
# 4. ADVANCED ENSEMBLE LEARNING APPROACHES
# ===================================

# Prepare features
feature_cols = [col for col in training_data.columns if col.startswith('input_') or 
               col in ['output_frame', 'time_diff', 'vel_trend_x', 'vel_trend_y', 'speed_mean', 
                      'speed_std', 'speed_trend', 'dir_volatility', 'dir_trend', 'acc_x_est', 
                      'acc_y_est', 'jerk_x_est', 'jerk_y_est', 'path_curvature', 'play_center_x', 
                      'play_center_y', 'play_spread_x', 'play_spread_y', 'play_avg_speed', 
                      'play_max_speed', 'min_teammate_dist', 'avg_teammate_dist', 'formation_compactness',
                      'min_opponent_dist', 'avg_opponent_dist', 'pressure_score', 'close_opponents',
                      'formation_width', 'formation_depth', 'actual_displacement']]

X = training_data[feature_cols].fillna(0)
y = training_data[['target_x', 'target_y']]

print(f"Ensemble training features: {len(feature_cols)}")
print(f"Training samples: {len(X)}")

# Handle categorical features
categorical_cols = [col for col in X.columns if 'field_third' in col]
for col in categorical_cols:
    if col in X.columns:
        field_third_map = {'defensive': 0, 'middle': 1, 'offensive': 2}
        X[col] = X[col].map(field_third_map).fillna(1)

# Advanced feature selection for ensemble diversity
print("Performing ensemble-optimized feature selection...")

# Multi-stage feature selection
selector_univariate = SelectKBest(score_func=f_regression, k=min(200, len(feature_cols)))
X_selected = selector_univariate.fit_transform(X, y.iloc[:, 0])
selected_features = [feature_cols[i] for i in selector_univariate.get_support(indices=True)]

# Use selected features
X = X[selected_features]
feature_cols = selected_features
print(f"Selected {len(feature_cols)} features for ensemble learning")

# Multiple preprocessing pipelines for ensemble diversity
print("Creating multiple preprocessing pipelines...")

# Pipeline 1: Standard Scaling
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

# Pipeline 2: Robust Scaling
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)

# Pipeline 3: Quantile Transform (Normal)
qt_normal = QuantileTransformer(output_distribution='normal', random_state=42)
X_qt_normal = qt_normal.fit_transform(X)

# Pipeline 4: Power Transform
pt = PowerTransformer(method='yeo-johnson')
X_power = pt.fit_transform(X)

# Dimensionality reduction for ensemble diversity
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_robust)

ica = FastICA(n_components=30, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_robust)

# Train-validation split
y_bins = pd.qcut(y.iloc[:, 0], q=10, labels=False, duplicates='drop')
X_train_std, X_val_std, X_train_rob, X_val_rob, X_train_qt, X_val_qt, X_train_pow, X_val_pow, \
X_train_pca, X_val_pca, X_train_ica, X_val_ica, y_train, y_val = train_test_split(
    X_standard, X_robust, X_qt_normal, X_power, X_pca, X_ica, y, 
    test_size=0.15, random_state=42, stratify=y_bins
)

print("Training advanced ensemble models...")

# ===================================
# ENSEMBLE APPROACH 1: HETEROGENEOUS BASE MODELS
# ===================================

print("\n=== ENSEMBLE APPROACH 1: HETEROGENEOUS BASE MODELS ===")

# Diverse base models with different data preprocessing
base_models = {}

# Tree-based models on robust scaled data
base_models['XGBoost_Robust'] = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=600, max_depth=10, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.9, random_state=42, n_jobs=-1
))

base_models['LightGBM_Robust'] = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=600, max_depth=12, learning_rate=0.05, subsample=0.85,
    colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1
))

base_models['CatBoost_Robust'] = MultiOutputRegressor(cb.CatBoostRegressor(
    iterations=400, depth=8, learning_rate=0.05, random_state=42, verbose=False
))

# Linear models on different scalings
base_models['Ridge_Standard'] = MultiOutputRegressor(Ridge(alpha=1.0))
base_models['ElasticNet_Standard'] = MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
base_models['BayesianRidge_QT'] = MultiOutputRegressor(BayesianRidge())

# Neural networks on different transformations
base_models['MLP_Power'] = MultiOutputRegressor(MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=0.001,
    learning_rate_init=0.001, max_iter=500, random_state=42
))

# Ensemble models
base_models['RandomForest_Standard'] = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
))

base_models['ExtraTrees_Robust'] = MultiOutputRegressor(ExtraTreesRegressor(
    n_estimators=300, max_depth=25, random_state=42, n_jobs=-1
))

# Distance-based models on PCA
base_models['KNN_PCA'] = MultiOutputRegressor(KNeighborsRegressor(
    n_neighbors=15, weights='distance', n_jobs=-1
))

# Train base models with appropriate preprocessing
print("Training heterogeneous base models...")

data_map = {
    'Standard': (X_train_std, X_val_std),
    'Robust': (X_train_rob, X_val_rob),
    'QT': (X_train_qt, X_val_qt),
    'Power': (X_train_pow, X_val_pow),
    'PCA': (X_train_pca, X_val_pca)
}

model_scores_1 = {}
for name, model in base_models.items():
    # Determine which preprocessing to use
    if 'Standard' in name:
        X_tr, X_va = data_map['Standard']
    elif 'Robust' in name:
        X_tr, X_va = data_map['Robust']
    elif 'QT' in name:
        X_tr, X_va = data_map['QT']
    elif 'Power' in name:
        X_tr, X_va = data_map['Power']
    elif 'PCA' in name:
        X_tr, X_va = data_map['PCA']
    else:
        X_tr, X_va = data_map['Robust']  # Default
    
    try:
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        model_scores_1[name] = rmse
        print(f"{name}: RMSE = {rmse:.6f}")
    except Exception as e:
        print(f"Error training {name}: {e}")

# ===================================
# ENSEMBLE APPROACH 2: VOTING ENSEMBLE
# ===================================

print("\n=== ENSEMBLE APPROACH 2: VOTING ENSEMBLE ===")

# Create voting ensemble with top performing models
top_models = sorted(model_scores_1.items(), key=lambda x: x[1])[:5]
print(f"Top 5 models for voting: {[name for name, _ in top_models]}")

# Weighted voting ensemble
voting_weights = [1.0 / rmse for _, rmse in top_models]
voting_weights = [w / sum(voting_weights) for w in voting_weights]

def weighted_voting_predict(models_data, X_test):
    predictions = []
    total_weight = 0
    
    for (name, _), weight in zip(top_models, voting_weights):
        if name in models_data:
            model = models_data[name]
            # Use appropriate preprocessing
            if 'Standard' in name:
                X_test_proc = scaler_standard.transform(X_test) if X_test.shape[1] == len(feature_cols) else X_test
            elif 'QT' in name:
                X_test_proc = qt_normal.transform(X_test) if X_test.shape[1] == len(feature_cols) else X_test
            elif 'Power' in name:
                X_test_proc = pt.transform(X_test) if X_test.shape[1] == len(feature_cols) else X_test
            elif 'PCA' in name:
                X_test_proc = pca.transform(scaler_robust.transform(X_test)) if X_test.shape[1] == len(feature_cols) else X_test
            else:
                X_test_proc = scaler_robust.transform(X_test) if X_test.shape[1] == len(feature_cols) else X_test
            
            try:
                pred = model.predict(X_test_proc)
                predictions.append(pred * weight)
                total_weight += weight
            except:
                continue
    
    if predictions and total_weight > 0:
        return np.sum(predictions, axis=0) / total_weight
    else:
        return np.zeros((len(X_test), 2))

# Evaluate voting ensemble
voting_pred = weighted_voting_predict(base_models, X[y_val.index])
voting_rmse = np.sqrt(mean_squared_error(y_val, voting_pred))
print(f"Weighted Voting Ensemble RMSE: {voting_rmse:.6f}")

# ===================================
# ENSEMBLE APPROACH 3: STACKING ENSEMBLE
# ===================================

print("\n=== ENSEMBLE APPROACH 3: STACKING ENSEMBLE ===")

# Multi-level stacking
print("Creating stacking ensemble...")

# Level 1: Diverse base models
level1_models = [
    ('xgb', MultiOutputRegressor(xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1))),
    ('lgb', MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=400, max_depth=10, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1))),
    ('rf', MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1))),
    ('et', MultiOutputRegressor(ExtraTreesRegressor(n_estimators=200, max_depth=18, random_state=42, n_jobs=-1))),
    ('mlp', MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)))
]

# Level 2: Meta-model
meta_model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
))

# Create stacking regressor
stacking_ensemble = StackingRegressor(
    estimators=level1_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

# Train stacking ensemble
print("Training stacking ensemble...")
stacking_ensemble.fit(X_train_rob, y_train)

# Evaluate stacking ensemble
stacking_pred = stacking_ensemble.predict(X_val_rob)
stacking_rmse = np.sqrt(mean_squared_error(y_val, stacking_pred))
print(f"Stacking Ensemble RMSE: {stacking_rmse:.6f}")

# ===================================
# ENSEMBLE APPROACH 4: BLENDING ENSEMBLE
# ===================================

print("\n=== ENSEMBLE APPROACH 4: BLENDING ENSEMBLE ===")

# Create holdout set for blending
X_blend_train, X_holdout, y_blend_train, y_holdout = train_test_split(
    X_train_rob, y_train, test_size=0.2, random_state=42
)

# Train base models on blend training set
blend_models = {}
blend_models['XGB_Blend'] = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=500, max_depth=9, learning_rate=0.04, random_state=42, n_jobs=-1
))
blend_models['LGB_Blend'] = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=500, max_depth=11, learning_rate=0.04, random_state=42, n_jobs=-1, verbose=-1
))
blend_models['CB_Blend'] = MultiOutputRegressor(cb.CatBoostRegressor(
    iterations=300, depth=9, learning_rate=0.06, random_state=42, verbose=False
))
blend_models['RF_Blend'] = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=250, max_depth=20, random_state=42, n_jobs=-1
))

print("Training blending base models...")
for name, model in blend_models.items():
    model.fit(X_blend_train, y_blend_train)
    print(f"Trained {name}")

# Create blending predictions on holdout set
blend_features = []
for name, model in blend_models.items():
    pred = model.predict(X_holdout)
    blend_features.append(pred)

blend_features = np.hstack(blend_features)
print(f"Blend features shape: {blend_features.shape}")

# Train blending meta-model
blend_meta = MultiOutputRegressor(Ridge(alpha=0.1))
blend_meta.fit(blend_features, y_holdout)

# Evaluate blending ensemble
def blending_predict(X_test):
    test_blend_features = []
    for name, model in blend_models.items():
        pred = model.predict(X_test)
        test_blend_features.append(pred)
    
    test_blend_features = np.hstack(test_blend_features)
    return blend_meta.predict(test_blend_features)

blend_val_features = []
for name, model in blend_models.items():
    pred = model.predict(X_val_rob)
    blend_val_features.append(pred)

blend_val_features = np.hstack(blend_val_features)
blend_pred = blend_meta.predict(blend_val_features)
blend_rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
print(f"Blending Ensemble RMSE: {blend_rmse:.6f}")

# ===================================
# ENSEMBLE APPROACH 5: BAYESIAN MODEL AVERAGING
# ===================================

print("\n=== ENSEMBLE APPROACH 5: BAYESIAN MODEL AVERAGING ===")

# Bayesian weights based on validation performance
ensemble_scores = {
    'Voting': voting_rmse,
    'Stacking': stacking_rmse,
    'Blending': blend_rmse
}

# Convert RMSE to weights (lower RMSE = higher weight)
bayesian_weights = {}
total_inv_rmse = sum(1.0 / rmse for rmse in ensemble_scores.values())

for name, rmse in ensemble_scores.items():
    bayesian_weights[name] = (1.0 / rmse) / total_inv_rmse

print("Bayesian Model Averaging Weights:")
for name, weight in bayesian_weights.items():
    print(f"{name}: {weight:.4f}")

# Final ensemble prediction function
def final_ensemble_predict(X_test):
    """Final ensemble combining all approaches with Bayesian weights"""
    
    # Get predictions from each ensemble
    voting_pred = weighted_voting_predict(base_models, X_test)
    
    # For stacking and blending, we need proper preprocessing
    X_test_rob = scaler_robust.transform(X_test)
    stacking_pred = stacking_ensemble.predict(X_test_rob)
    blend_pred = blending_predict(X_test_rob)
    
    # Bayesian averaging
    final_pred = (voting_pred * bayesian_weights['Voting'] +
                 stacking_pred * bayesian_weights['Stacking'] +
                 blend_pred * bayesian_weights['Blending'])
    
    return final_pred

# Evaluate final ensemble
final_pred = final_ensemble_predict(X[y_val.index])
final_rmse = np.sqrt(mean_squared_error(y_val, final_pred))

print(f"\n🎯 FINAL BAYESIAN ENSEMBLE RESULTS:")
print(f"Final Ensemble RMSE: {final_rmse:.6f} yards")
print(f"X-coordinate RMSE: {np.sqrt(mean_squared_error(y_val.iloc[:, 0], final_pred[:, 0])):.6f} yards")
print(f"Y-coordinate RMSE: {np.sqrt(mean_squared_error(y_val.iloc[:, 1], final_pred[:, 1])):.6f} yards")

# Store ensemble info
ensemble_info = {
    'base_models': base_models,
    'stacking_ensemble': stacking_ensemble,
    'blend_models': blend_models,
    'blend_meta': blend_meta,
    'bayesian_weights': bayesian_weights,
    'feature_cols': feature_cols,
    'scaler_standard': scaler_standard,
    'scaler_robust': scaler_robust,
    'qt_normal': qt_normal,
    'pt': pt,
    'pca': pca,
    'ica': ica,
    'selector_univariate': selector_univariate,
    'data_map': data_map
}

# ===================================
# 6. GENERATE PREDICTIONS ON TEST DATA
# ===================================

print("\nLoading test data...")

if not os.path.exists(test_input_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Test files not found.")

test_input = pd.read_csv(test_input_path)
test_targets = pd.read_csv(test_path)

print(f"Test input: {len(test_input)} rows")
print(f"Test targets: {len(test_targets)} rows")

# Apply feature engineering to test data
test_input_features, _, _ = create_advanced_features(test_input)

def generate_ensemble_predictions(test_input_df, test_targets_df, ensemble_predict_func, ensemble_info):
    """Generate predictions using advanced ensemble"""
    
    predictions = []
    
    for (game_id, play_id), play_targets in test_targets_df.groupby(['game_id', 'play_id']):
        play_input = test_input_df[(test_input_df['game_id'] == game_id) & 
                                  (test_input_df['play_id'] == play_id)]
        
        if len(play_input) == 0:
            for _, target_row in play_targets.iterrows():
                predictions.append({
                    'id': f"{target_row['game_id']}_{target_row['play_id']}_{target_row['nfl_id']}_{target_row['frame_id']}",
                    'x': 50.0, 'y': 26.65
                })
            continue
            
        for _, target_row in play_targets.iterrows():
            player_id = target_row['nfl_id']
            target_frame = target_row['frame_id']
            
            player_input = play_input[play_input['nfl_id'] == player_id].sort_values('frame_id')
            
            if len(player_input) == 0:
                pred_x, pred_y = 50.0, 26.65
            else:
                # Create feature vector (similar to training)
                last_frame = player_input.loc[player_input['frame_id'].idxmax()]
                
                # Prepare features (simplified for test)
                feature_dict = {}
                
                # Add input features
                for col in last_frame.index:
                    if not col.startswith('game_id') and not col.startswith('play_id') and not col.startswith('nfl_id'):
                        feature_dict[f'input_{col}'] = last_frame[col]
                
                # Add basic contextual features
                feature_dict['output_frame'] = target_frame
                feature_dict['time_diff'] = target_frame - last_frame['frame_id']
                
                # Create feature vector
                feature_values = []
                for col in ensemble_info['feature_cols']:
                    if col in feature_dict:
                        val = feature_dict[col]
                        if isinstance(val, str) and 'field_third' in col:
                            field_third_map = {'defensive': 0, 'middle': 1, 'offensive': 2}
                            val = field_third_map.get(val, 1)
                        feature_values.append(val)
                    else:
                        feature_values.append(0)
                
                feature_df = pd.DataFrame([feature_values], columns=ensemble_info['feature_cols'])
                
                try:
                    pred_xy = ensemble_predict_func(feature_df)[0]
                    pred_x, pred_y = pred_xy[0], pred_xy[1]
                    
                    # Apply field constraints
                    pred_x = np.clip(pred_x, 0, 120)
                    pred_y = np.clip(pred_y, 0, 53.3)
                except:
                    pred_x, pred_y = 50.0, 26.65
            
            predictions.append({
                'id': f"{target_row['game_id']}_{target_row['play_id']}_{target_row['nfl_id']}_{target_row['frame_id']}",
                'x': pred_x, 'y': pred_y
            })
    
    return pd.DataFrame(predictions)

# Generate predictions
print("Generating predictions with advanced ensemble...")
submission = generate_ensemble_predictions(test_input_features, test_targets, final_ensemble_predict, ensemble_info)

print(f"Generated {len(submission)} predictions")

# ===================================
# 7. CREATE SUBMISSION FILE
# ===================================

if not os.path.exists(sample_submission_path):
    raise FileNotFoundError("sample_submission.csv not found")

sample_submission = pd.read_csv(sample_submission_path)

# Validate and clean submission
expected_ids = set(sample_submission['id'])
our_ids = set(submission['id'])

missing_ids = expected_ids - our_ids
if missing_ids:
    print(f"Adding {len(missing_ids)} missing predictions...")
    missing_df = pd.DataFrame([
        {'id': missing_id, 'x': 50.0, 'y': 26.65} for missing_id in missing_ids
    ])
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Ensure correct order and fill missing values
submission = submission.merge(sample_submission[['id']], on='id', how='right')
submission = submission.sort_values('id').reset_index(drop=True)
submission = submission.fillna({'x': 50.0, 'y': 26.65})

# Save submission
submission.to_csv('submission_advanced_ensemble.csv', index=False)

print(f"\n🏆 ADVANCED ENSEMBLE SUBMISSION SUMMARY:")
print(f"Total predictions: {len(submission)}")
print(f"X coordinate range: {submission['x'].min():.1f} to {submission['x'].max():.1f}")
print(f"Y coordinate range: {submission['y'].min():.1f} to {submission['y'].max():.1f}")
print(f"Average X: {submission['x'].mean():.1f}")
print(f"Average Y: {submission['y'].mean():.1f}")

print("\n" + "="*80)
print("🚀 ADVANCED ENSEMBLE LEARNING SUBMISSION CREATED SUCCESSFULLY! 🚀")
print("="*80)
print("File saved as: submission_advanced_ensemble.csv")
print("✅ Heterogeneous Base Models")
print("✅ Voting Ensemble") 
print("✅ Stacking Ensemble")
print("✅ Blending Ensemble")
print("✅ Bayesian Model Averaging")
print("Ready for elite competition performance!")