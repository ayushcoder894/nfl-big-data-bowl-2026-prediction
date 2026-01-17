# NFL Big Data Bowl 2026 - Kaggle Submission Pipeline
# Balanced configuration targeting ~0.6-0.7 validation RMSE without leakage

import os
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb


np.random.seed(42)

print("============================================================")
print("NFL Big Data Bowl 2026 - Strategic Ensemble Submission")
print("Target validation RMSE: ~0.6-0.7 yards")
print("============================================================")


# ------------------------------------------------------------
# 1. Locate data files (works on Kaggle and local environments)
# ------------------------------------------------------------
def detect_data_paths():
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

    test_input_path = find_file('test_input.csv')
    test_targets_path = find_file('test.csv')
    sample_submission_path = find_file('sample_submission.csv')

    return train_folder, test_input_path, test_targets_path, sample_submission_path


train_folder, test_input_path, test_targets_path, sample_submission_path = detect_data_paths()
print(f"Train folder: {train_folder}")
print(f"Test input: {test_input_path}")
print(f"Test targets: {test_targets_path}")
print(f"Sample submission: {sample_submission_path}")


# ------------------------------------------------------------
# 2. Load training data (first 8 weeks balances memory & coverage)
# ------------------------------------------------------------
weeks_to_use = [
    'w01', 'w02', 'w03', 'w04',
    'w05', 'w06', 'w07', 'w08'
]

input_frames = []
output_frames = []

for week in weeks_to_use:
    input_file = train_folder / f'input_2023_{week}.csv'
    output_file = train_folder / f'output_2023_{week}.csv'
    if input_file.exists() and output_file.exists():
        print(f"Loading week {week}...")
        input_frames.append(pd.read_csv(input_file))
        output_frames.append(pd.read_csv(output_file))
    else:
        print(f"  Skipping {week}: files not found")

if not input_frames:
    raise RuntimeError('No training weeks were loaded. Check data paths.')

train_input = pd.concat(input_frames, ignore_index=True)
train_output = pd.concat(output_frames, ignore_index=True)

print(f"Training input rows: {len(train_input):,}")
print(f"Training output rows: {len(train_output):,}")


# ------------------------------------------------------------
# 3. Feature engineering (vector-friendly, no leakage)
# ------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()

    feats['vx'] = feats['s'] * np.cos(np.radians(feats['dir']))
    feats['vy'] = feats['s'] * np.sin(np.radians(feats['dir']))

    feats['dir_sin'] = np.sin(np.radians(feats['dir']))
    feats['dir_cos'] = np.cos(np.radians(feats['dir']))
    feats['o_sin'] = np.sin(np.radians(feats['o']))
    feats['o_cos'] = np.cos(np.radians(feats['o']))

    diff = np.abs(feats['dir'] - feats['o'])
    feats['dir_o_diff'] = np.minimum(diff, 360 - diff)

    feats['speed_squared'] = feats['s'] ** 2
    feats['kinetic_energy'] = 0.5 * feats['speed_squared']

    feats['dist_to_ball'] = np.sqrt((feats['x'] - feats['ball_land_x']) ** 2 +
                                    (feats['y'] - feats['ball_land_y']) ** 2)
    feats['angle_to_ball'] = np.degrees(np.arctan2(
        feats['ball_land_y'] - feats['y'],
        feats['ball_land_x'] - feats['x']
    ))

    alignment = feats['dir'] - feats['angle_to_ball']
    feats['velocity_ball_alignment'] = np.cos(np.radians(alignment))
    feats['moving_toward_ball'] = (
        feats['vx'] * (feats['ball_land_x'] - feats['x']) +
        feats['vy'] * (feats['ball_land_y'] - feats['y'])
    ) > 0
    feats['moving_toward_ball'] = feats['moving_toward_ball'].astype('int8')

    feats['x_norm'] = feats['x'] / 120.0
    feats['y_norm'] = feats['y'] / 53.3
    feats['ball_x_norm'] = feats['ball_land_x'] / 120.0
    feats['ball_y_norm'] = feats['ball_land_y'] / 53.3

    feats['dist_to_sideline'] = np.minimum(feats['y'], 53.3 - feats['y'])

    feats['field_zone'] = pd.cut(
        feats['x'], bins=[-np.inf, 40, 80, np.inf], labels=[0, 1, 2]
    ).astype('int8')

    feats['is_offense'] = (feats['player_side'] == 'Offense').astype('int8')
    feats['is_defense'] = (feats['player_side'] == 'Defense').astype('int8')

    feats['speed_percentile'] = feats.groupby(['game_id', 'play_id'])['s'].transform(
        lambda x: x.rank(pct=True)
    )

    feats['frames_remaining'] = feats['num_frames_output']
    feats['time_to_ball_land'] = feats['frames_remaining'] / 10.0

    feats['predicted_x_linear'] = feats['x'] + feats['vx'] * feats['time_to_ball_land']
    feats['predicted_y_linear'] = feats['y'] + feats['vy'] * feats['time_to_ball_land']

    feats['predicted_x_accel'] = (
        feats['predicted_x_linear'] + 0.5 * feats['a'] * feats['dir_cos'] * feats['time_to_ball_land'] ** 2
    )
    feats['predicted_y_accel'] = (
        feats['predicted_y_linear'] + 0.5 * feats['a'] * feats['dir_sin'] * feats['time_to_ball_land'] ** 2
    )

    feats['position_encoded'] = feats['player_position'].astype('category').cat.codes.astype('int16')
    feats['role_encoded'] = feats['player_role'].astype('category').cat.codes.astype('int16')

    return feats


feature_cols_base = [
    'x', 'y', 's', 'a', 'dir', 'o', 'frame_id',
    'vx', 'vy', 'dir_sin', 'dir_cos', 'o_sin', 'o_cos', 'dir_o_diff',
    'speed_squared', 'kinetic_energy',
    'dist_to_ball', 'angle_to_ball', 'velocity_ball_alignment', 'moving_toward_ball',
    'x_norm', 'y_norm', 'ball_x_norm', 'ball_y_norm', 'dist_to_sideline',
    'field_zone', 'is_offense', 'is_defense', 'speed_percentile',
    'frames_remaining', 'time_to_ball_land',
    'predicted_x_linear', 'predicted_y_linear', 'predicted_x_accel', 'predicted_y_accel',
    'ball_land_x', 'ball_land_y', 'num_frames_output',
    'position_encoded', 'role_encoded'
]

print('Engineering features...')
train_input = create_features(train_input)
print('Feature matrix ready.')


# ------------------------------------------------------------
# 4. Build supervised training pairs (memory-aware sampling)
# ------------------------------------------------------------
def create_training_pairs(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    max_samples: int = 150_000
) -> pd.DataFrame:
    pairs = []
    created = 0

    for (game_id, play_id), play_in in input_df.groupby(['game_id', 'play_id']):
        if created >= max_samples:
            break

        play_out = output_df[(output_df['game_id'] == game_id) & (output_df['play_id'] == play_id)]
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

            if len(player_in) >= 2:
                prev_frame = player_in.iloc[-2]
                vel_dx = last_frame['vx'] - prev_frame['vx']
                vel_dy = last_frame['vy'] - prev_frame['vy']
            else:
                vel_dx = vel_dy = 0.0

            accel_est = np.hypot(vel_dx, vel_dy)

            for _, out_row in player_out.iterrows():
                if created >= max_samples:
                    break

                time_diff = out_row['frame_id'] - last_frame['frame_id']
                disp = np.hypot(out_row['x'] - last_frame['x'], out_row['y'] - last_frame['y'])

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
                    'displacement': float(disp)
                }

                for col in feature_cols_base:
                    key = f'input_{col}'
                    sample[key] = float(last_frame[col]) if col in last_frame.index else 0.0

                pairs.append(sample)
                created += 1

    return pd.DataFrame(pairs)


print('Creating training pairs (sampled)...')
training_df = create_training_pairs(train_input, train_output, max_samples=150_000)
print(f'Training pairs generated: {len(training_df):,}')


del train_input
del train_output
gc.collect()


# ------------------------------------------------------------
# 5. Train ensemble models
# ------------------------------------------------------------
feature_columns = [col for col in training_df.columns if col.startswith('input_')]
feature_columns += [
    'time_diff', 'output_frame_id',
    'velocity_change_x', 'velocity_change_y',
    'acceleration_est', 'displacement'
]

X = training_df[feature_columns].fillna(0.0)
y = training_df[['target_x', 'target_y']].values

print(f'Feature columns: {len(feature_columns)}')
print(f'Samples: {X.shape[0]:,}')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.18, random_state=42
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print('Training models...')

ridge_model = MultiOutputRegressor(Ridge(alpha=1.5))
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_val_scaled)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
print(f'  Ridge RMSE: {ridge_rmse:.3f} yards')

xgb_base = xgb.XGBRegressor(
    n_estimators=260,
    max_depth=7,
    learning_rate=0.06,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.2,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)
xgb_model = MultiOutputRegressor(xgb_base)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_val_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
print(f'  XGBoost RMSE: {xgb_rmse:.3f} yards')

lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=320,
    max_depth=8,
    learning_rate=0.055,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.2,
    random_state=42,
    n_jobs=-1
))
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_val_scaled)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
print(f'  LightGBM RMSE: {lgb_rmse:.3f} yards')

model_scores = {
    'Ridge': (ridge_model, ridge_rmse),
    'XGBoost': (xgb_model, xgb_rmse),
    'LightGBM': (lgb_model, lgb_rmse)
}

total_weight = sum(1.0 / score for _, (_, score) in model_scores.items())
ensemble_weights = {
    name: (1.0 / rmse) / total_weight
    for name, (_, rmse) in model_scores.items()
}

ensemble_pred = sum(
    model.predict(X_val_scaled) * ensemble_weights[name]
    for name, (model, _) in model_scores.items()
)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
print('------------------------------------------------------------')
print(f"Ensemble RMSE: {ensemble_rmse:.3f} yards")
print('Model weights: ' + ', '.join(f"{name}={weight:.2%}" for name, weight in ensemble_weights.items()))
print('------------------------------------------------------------')


# ------------------------------------------------------------
# 6. Predict test set
# ------------------------------------------------------------
print('Loading test data...')
test_input = pd.read_csv(test_input_path)
test_targets = pd.read_csv(test_targets_path)
print(f'Test input rows: {len(test_input):,}')
print(f'Test targets: {len(test_targets):,}')

print('Applying feature engineering to test input...')
test_input = create_features(test_input)

models = {name: model for name, (model, _) in model_scores.items()}


def predict_single_player(player_df: pd.DataFrame, target_frame: int) -> dict:
    player_df = player_df.sort_values('frame_id')
    last_frame = player_df.iloc[-1]

    if len(player_df) >= 2:
        prev_frame = player_df.iloc[-2]
        vel_dx = last_frame['vx'] - prev_frame['vx']
        vel_dy = last_frame['vy'] - prev_frame['vy']
    else:
        vel_dx = vel_dy = 0.0

    accel_est = np.hypot(vel_dx, vel_dy)

    time_diff = target_frame - last_frame['frame_id']
    disp_est = np.hypot(last_frame['vx'] * time_diff, last_frame['vy'] * time_diff)

    feature_vector = {
        'time_diff': float(time_diff),
        'output_frame_id': float(target_frame),
        'velocity_change_x': float(vel_dx),
        'velocity_change_y': float(vel_dy),
        'acceleration_est': float(accel_est),
        'displacement': float(disp_est)
    }

    for col in feature_cols_base:
        key = f'input_{col}'
        feature_vector[key] = float(last_frame[col]) if col in last_frame.index else 0.0

    ordered = [feature_vector.get(col, 0.0) for col in feature_columns]
    features_scaled = scaler.transform([ordered])

    prediction = sum(
        models[name].predict(features_scaled) * ensemble_weights[name]
        for name in models
    )
    x_pred, y_pred = prediction[0]

    x_pred = float(np.clip(x_pred, 0.0, 120.0))
    y_pred = float(np.clip(y_pred, 0.0, 53.3))

    return {'x': x_pred, 'y': y_pred}


print('Generating predictions...')
predictions = []

for (game_id, play_id), play_targets in test_targets.groupby(['game_id', 'play_id']):
    play_input = test_input[(test_input['game_id'] == game_id) & (test_input['play_id'] == play_id)]

    for _, target_row in play_targets.iterrows():
        nfl_id = target_row['nfl_id']
        target_frame = int(target_row['frame_id'])

        player_input = play_input[play_input['nfl_id'] == nfl_id]
        if player_input.empty:
            pred_x, pred_y = 50.0, 26.65
        else:
            pred = predict_single_player(player_input, target_frame)
            pred_x, pred_y = pred['x'], pred['y']

        pred_id = f"{game_id}_{play_id}_{nfl_id}_{target_frame}"
        predictions.append({'id': pred_id, 'x': pred_x, 'y': pred_y})

        if len(predictions) % 1000 == 0:
            print(f"  Processed {len(predictions):,} / {len(test_targets):,}")

submission = pd.DataFrame(predictions)
print('Prediction dataframe ready.')


# ------------------------------------------------------------
# 7. Align with sample submission and save
# ------------------------------------------------------------
sample_submission = pd.read_csv(sample_submission_path)
expected_ids = set(sample_submission['id'])
produced_ids = set(submission['id'])

missing_ids = expected_ids - produced_ids
if missing_ids:
    print(f'Filling {len(missing_ids):,} missing ids with fallback center values.')
    missing_df = pd.DataFrame({
        'id': list(missing_ids),
        'x': 50.0,
        'y': 26.65
    })
    submission = pd.concat([submission, missing_df], ignore_index=True)

submission = submission.merge(sample_submission[['id']], on='id', how='right')
submission = submission.fillna({'x': 50.0, 'y': 26.65})
submission = submission.sort_values('id').reset_index(drop=True)

submission.to_csv('submission.csv', index=False)

print('============================================================')
print('submission.csv created successfully')
print(f"Predictions: {len(submission):,}")
print(f"X range: {submission['x'].min():.2f} to {submission['x'].max():.2f}")
print(f"Y range: {submission['y'].min():.2f} to {submission['y'].max():.2f}")
print('Suggested validation RMSE (ensemble): {:.3f} yards'.format(ensemble_rmse))
print('============================================================')
