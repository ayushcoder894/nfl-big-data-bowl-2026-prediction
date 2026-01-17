"""
NFL Big Data Bowl 2026 - State-of-the-Art Transformer-Based Multi-Agent Model
Architecture: Spatiotemporal Encoder-Decoder with Role-Aware Cross-Attention
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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Check GPU availability
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    print("=" * 80)
    print("NFL BIG DATA BOWL 2026 - TRANSFORMER MULTI-AGENT ARCHITECTURE")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device('cpu')
    print("=" * 80)
    print("NFL BIG DATA BOWL 2026 - TRANSFORMER MULTI-AGENT ARCHITECTURE")
    print("=" * 80)
    print(f"Device: {device}")
    print("⚠️  WARNING: PyTorch CPU-only version detected!")
    print("⚠️  To use GPU, install PyTorch with CUDA support:")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("=" * 80)

print("Multi-Agent Spatiotemporal Transformer with Role-Aware Attention")
print("Target: RMSE < 0.5 yards")
print("=" * 80)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Transformer model configuration - Optimized for GPU"""
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 50
    num_roles: int = 4
    batch_size: int = 256  # Increased for GPU
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    num_epochs: int = 20  # More epochs with GPU speed


config = ModelConfig()


# ============================================================================
# DATA LOADING & PREPROCESSING
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
# SPATIOTEMPORAL FEATURE ENGINEERING
# ============================================================================

def engineer_spatiotemporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering for transformer input:
    - Velocity decomposition
    - Ball-centric features
    - Spatial relationships
    - Temporal context
    """
    print("\n[PHASE 2] Engineering Spatiotemporal Features")
    print("-" * 60)
    
    feats = df.copy()
    
    # === A. VELOCITY DECOMPOSITION ===
    print("  → Velocity vectors...")
    feats['vx'] = feats['s'] * np.cos(np.radians(feats['dir']))
    feats['vy'] = feats['s'] * np.sin(np.radians(feats['dir']))
    
    # === B. BALL-CENTRIC FEATURES (Primary Signal) ===
    print("  → Ball-centric features...")
    feats['dist_to_ball'] = np.sqrt(
        (feats['x'] - feats['ball_land_x']) ** 2 +
        (feats['y'] - feats['ball_land_y']) ** 2
    )
    feats['angle_to_ball'] = np.degrees(np.arctan2(
        feats['ball_land_y'] - feats['y'],
        feats['ball_land_x'] - feats['x']
    ))
    
    # Velocity alignment with ball
    alignment = feats['dir'] - feats['angle_to_ball']
    feats['velocity_ball_alignment'] = np.cos(np.radians(alignment))
    feats['moving_toward_ball'] = (feats['velocity_ball_alignment'] > 0).astype('int8')
    
    # === C. NORMALIZED COORDINATES ===
    print("  → Coordinate normalization...")
    feats['x_norm'] = feats['x'] / 120.0
    feats['y_norm'] = feats['y'] / 53.3
    feats['ball_x_norm'] = feats['ball_land_x'] / 120.0
    feats['ball_y_norm'] = feats['ball_land_y'] / 53.3
    
    # === D. DIRECTIONAL ENCODING ===
    feats['dir_sin'] = np.sin(np.radians(feats['dir']))
    feats['dir_cos'] = np.cos(np.radians(feats['dir']))
    feats['o_sin'] = np.sin(np.radians(feats['o']))
    feats['o_cos'] = np.cos(np.radians(feats['o']))
    
    # === E. ROLE ENCODING ===
    print("  → Role encoding...")
    role_map = {
        'Targeted Receiver': 0,
        'Passer': 1,
        'Defensive Coverage': 2,
        'Other Route Runner': 3
    }
    feats['role_id'] = feats['player_role'].map(role_map).fillna(3).astype('int8')
    
    # === F. PHYSICS FEATURES ===
    feats['speed_squared'] = feats['s'] ** 2
    feats['kinetic_energy'] = 0.5 * feats['speed_squared']
    feats['momentum'] = feats['s'] * 200.0  # Approximate mass
    
    # === G. FIELD CONTEXT ===
    feats['dist_to_sideline'] = np.minimum(feats['y'], 53.3 - feats['y'])
    feats['in_red_zone'] = ((feats['x'] < 20) | (feats['x'] > 100)).astype('int8')
    
    print(f"✓ Feature engineering complete: {len(feats.columns)} total features")
    
    return feats


# ============================================================================
# SEQUENCE DATASET FOR TRANSFORMER
# ============================================================================

class NFLTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for multi-agent trajectory prediction
    Handles variable-length sequences and multiple output frames
    """
    
    def __init__(self, input_df: pd.DataFrame, output_df: pd.DataFrame, max_samples: int = 150_000):
        self.samples = []
        
        print("\n[PHASE 3] Creating Sequence Dataset")
        print("-" * 60)
        
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
            
            # Process each player separately
            for nfl_id in play_out['nfl_id'].unique():
                if created >= max_samples:
                    break
                
                player_in = play_in[play_in['nfl_id'] == nfl_id].sort_values('frame_id')
                player_out = play_out[play_out['nfl_id'] == nfl_id].sort_values('frame_id')
                
                if player_in.empty or player_out.empty:
                    continue
                
                # Check if target player
                is_target = bool(player_in['player_to_predict'].iloc[0]) if 'player_to_predict' in player_in.columns else False
                
                # Extract features
                last_frame = player_in.iloc[-1]
                
                input_features = np.array([
                    last_frame['x_norm'], last_frame['y_norm'],
                    last_frame['vx'], last_frame['vy'],
                    last_frame['s'], last_frame['a'],
                    last_frame['dir_sin'], last_frame['dir_cos'],
                    last_frame['o_sin'], last_frame['o_cos'],
                    last_frame['ball_x_norm'], last_frame['ball_y_norm'],
                    last_frame['dist_to_ball'], last_frame['velocity_ball_alignment'],
                    last_frame['kinetic_energy'], last_frame['momentum'],
                    last_frame['dist_to_sideline'], last_frame['in_red_zone']
                ], dtype=np.float32)
                
                role_id = int(last_frame['role_id'])
                
                # Create samples for each output frame
                for _, out_row in player_out.iterrows():
                    target_x = float(out_row['x']) / 120.0  # Normalized
                    target_y = float(out_row['y']) / 53.3
                    time_diff = float(out_row['frame_id'] - last_frame['frame_id'])
                    
                    self.samples.append({
                        'input_features': input_features,
                        'role_id': role_id,
                        'time_diff': time_diff,
                        'target_x': target_x,
                        'target_y': target_y,
                        'is_target': float(is_target)
                    })
                    
                    created += 1
        
        print(f"✓ Created {len(self.samples):,} training samples")
        print(f"  Target player ratio: {np.mean([s['is_target'] for s in self.samples]):.1%}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input_features': torch.FloatTensor(sample['input_features']),
            'role_id': torch.LongTensor([sample['role_id']]),
            'time_diff': torch.FloatTensor([sample['time_diff']]),
            'target': torch.FloatTensor([sample['target_x'], sample['target_y']]),
            'is_target': torch.FloatTensor([sample['is_target']])
        }


# ============================================================================
# TRANSFORMER ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]


class RoleAwareAttention(nn.Module):
    """
    Role-aware cross-attention mechanism
    - Defensive players attend to receivers and ball
    - Receivers attend to ball and nearby defenders
    """
    
    def __init__(self, d_model: int, num_roles: int):
        super().__init__()
        self.d_model = d_model
        self.num_roles = num_roles
        
        # Role embeddings
        self.role_embeddings = nn.Embedding(num_roles, d_model)
        
        # Attention weights
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, role_ids):
        """
        Args:
            x: (batch, d_model)
            role_ids: (batch, 1)
        """
        # Add role embeddings
        role_embed = self.role_embeddings(role_ids.squeeze(-1))
        x_with_role = x + role_embed
        
        # Compute attention (self-attention for simplicity in single-player case)
        Q = self.query_proj(x_with_role)
        K = self.key_proj(x_with_role)
        V = self.value_proj(x_with_role)
        
        attn_weights = torch.softmax(torch.matmul(Q, K.T) / np.sqrt(self.d_model), dim=-1)
        attended = torch.matmul(attn_weights, V)
        
        return attended


class SpatiotemporalTransformer(nn.Module):
    """
    Multi-agent spatiotemporal transformer for trajectory prediction
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(18, config.d_model),  # 18 input features
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Role embeddings
        self.role_embed = nn.Embedding(config.num_roles, config.d_model)
        
        # Time encoding
        self.time_proj = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        
        # Role-aware attention
        self.role_attention = RoleAwareAttention(config.d_model, config.num_roles)
        
        # Physics-informed decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.LayerNorm(config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward // 2, 2)  # (x, y)
        )
        
        # Physical constraint layer (soft constraints)
        self.constraint_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, input_features, role_ids, time_diff):
        """
        Args:
            input_features: (batch, 18)
            role_ids: (batch, 1)
            time_diff: (batch, 1)
        """
        batch_size = input_features.size(0)
        
        # Project input features
        x = self.input_proj(input_features)  # (batch, d_model)
        
        # Add role embeddings
        role_embed = self.role_embed(role_ids.squeeze(-1))  # (batch, d_model)
        x = x + role_embed
        
        # Add temporal encoding
        time_embed = self.time_proj(time_diff)  # (batch, d_model)
        x = x + time_embed
        
        # Reshape for transformer (add sequence dimension)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        
        # Role-aware attention
        x = self.role_attention(x, role_ids)
        
        # Decode to coordinates
        output = self.decoder(x)  # (batch, 2)
        
        # Apply soft constraints (sigmoid keeps in [0, 1] range)
        output = torch.sigmoid(output * self.constraint_scale)
        
        return output


# ============================================================================
# PHYSICS-INFORMED LOSS FUNCTION
# ============================================================================

class PhysicsInformedLoss(nn.Module):
    """
    Multi-component loss:
    - Coordinate RMSE (primary)
    - Physical plausibility
    - Role-specific objectives
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super().__init__()
        self.alpha = alpha  # Coordinate loss weight
        self.beta = beta    # Velocity consistency weight
        self.gamma = gamma  # Physical constraint weight
    
    def forward(self, pred, target, input_features, time_diff, is_target):
        """
        Args:
            pred: (batch, 2) - predicted (x, y)
            target: (batch, 2) - actual (x, y)
            input_features: (batch, 18)
            time_diff: (batch, 1)
            is_target: (batch, 1)
        """
        # 1. Coordinate RMSE (weighted by target player)
        weights = 1.0 + 2.0 * is_target  # 3x weight for target players
        coord_loss = torch.sqrt(torch.mean(weights * (pred - target) ** 2))
        
        # 2. Velocity consistency (predicted displacement should be reasonable)
        current_x = input_features[:, 0:1]
        current_y = input_features[:, 1:2]
        current_vx = input_features[:, 2:3]
        current_vy = input_features[:, 3:4]
        current_speed = input_features[:, 4:5]
        
        # Expected displacement based on current velocity
        expected_displacement = current_speed * time_diff / 10.0
        
        # Actual predicted displacement
        pred_displacement = torch.sqrt(
            (pred[:, 0:1] * 120.0 - current_x * 120.0) ** 2 +
            (pred[:, 1:2] * 53.3 - current_y * 53.3) ** 2
        )
        
        # Penalize unrealistic displacements (allow up to 1.5x expected)
        velocity_loss = torch.mean(
            F.relu(pred_displacement - expected_displacement * 1.5)
        )
        
        # 3. Physical constraints (stay in bounds, max acceleration)
        # Bounds already enforced by sigmoid, but add soft penalty for being too close
        bound_penalty = torch.mean(
            F.relu(0.01 - pred) + F.relu(pred - 0.99)
        )
        
        # Total loss
        total_loss = (
            self.alpha * coord_loss +
            self.beta * velocity_loss +
            self.gamma * bound_penalty
        )
        
        return total_loss, coord_loss, velocity_loss, bound_penalty


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_transformer_model(
    train_dataset: NFLTrajectoryDataset,
    val_dataset: NFLTrajectoryDataset,
    config: ModelConfig
):
    """Train the spatiotemporal transformer model"""
    
    print("\n[PHASE 4] Training Spatiotemporal Transformer")
    print("-" * 60)
    
    # Data loaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model
    model = SpatiotemporalTransformer(config).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Loss function
    criterion = PhysicsInformedLoss()
    
    best_val_rmse = float('inf')
    
    # Enable GPU optimizations if available
    use_amp = torch.cuda.is_available()
    if use_amp:
        torch.cuda.empty_cache()
        scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_coord_loss = 0.0
        
        for batch in train_loader:
            input_features = batch['input_features'].to(device)
            role_ids = batch['role_id'].to(device)
            time_diff = batch['time_diff'].to(device)
            target = batch['target'].to(device)
            is_target = batch['is_target'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training for GPU efficiency
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(input_features, role_ids, time_diff)
                    loss, coord_loss, vel_loss, bound_loss = criterion(
                        pred, target, input_features, time_diff, is_target
                    )
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(input_features, role_ids, time_diff)
                loss, coord_loss, vel_loss, bound_loss = criterion(
                    pred, target, input_features, time_diff, is_target
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_coord_loss += coord_loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        context = torch.cuda.amp.autocast() if use_amp else torch.no_grad()
        with torch.no_grad(), context:
            for batch in val_loader:
                input_features = batch['input_features'].to(device, non_blocking=use_amp)
                role_ids = batch['role_id'].to(device, non_blocking=use_amp)
                time_diff = batch['time_diff'].to(device, non_blocking=use_amp)
                target = batch['target'].to(device, non_blocking=use_amp)
                
                pred = model(input_features, role_ids, time_diff)
                
                # Denormalize for RMSE calculation
                pred_denorm = pred.cpu().numpy()
                pred_denorm[:, 0] *= 120.0
                pred_denorm[:, 1] *= 53.3
                
                target_denorm = target.cpu().numpy()
                target_denorm[:, 0] *= 120.0
                target_denorm[:, 1] *= 53.3
                
                val_preds.append(pred_denorm)
                val_targets.append(target_denorm)
        
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        # GPU memory info if available
        if use_amp:
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**3
            gpu_mem_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  Epoch {epoch+1}/{config.num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val RMSE: {val_rmse:.4f} yards - "
                  f"GPU: {gpu_mem_used:.2f}GB/{gpu_mem_cached:.2f}GB")
        else:
            print(f"  Epoch {epoch+1}/{config.num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val RMSE: {val_rmse:.4f} yards")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), 'best_transformer_model.pt')
            print(f"    ✓ New best model saved!")
        
        # Clear GPU cache periodically
        if use_amp and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n✓ Training complete - Best Val RMSE: {best_val_rmse:.4f} yards")
    
    # Load best model
    model.load_state_dict(torch.load('best_transformer_model.pt'))
    
    return model, best_val_rmse


# ============================================================================
# ENSEMBLE WITH GRADIENT BOOSTING MODELS
# ============================================================================

def create_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for gradient boosting models"""
    
    feats = df.copy()
    
    # Basic features already created
    feature_cols = [
        'x_norm', 'y_norm', 'vx', 'vy', 's', 'a',
        'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
        'ball_x_norm', 'ball_y_norm', 'dist_to_ball',
        'velocity_ball_alignment', 'moving_toward_ball',
        'kinetic_energy', 'momentum', 'dist_to_sideline',
        'in_red_zone', 'role_id'
    ]
    
    return feats[feature_cols]


def train_gradient_boosting_ensemble(train_input: pd.DataFrame, train_output: pd.DataFrame):
    """Train XGBoost and LightGBM as ensemble members"""
    
    print("\n[PHASE 5] Training Gradient Boosting Ensemble")
    print("-" * 60)
    
    # Create training pairs
    pairs = []
    max_samples = 150_000
    created = 0
    
    for (game_id, play_id), play_in in train_input.groupby(['game_id', 'play_id']):
        if created >= max_samples:
            break
        
        play_out = train_output[
            (train_output['game_id'] == game_id) &
            (train_output['play_id'] == play_id)
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
            
            for _, out_row in player_out.iterrows():
                if created >= max_samples:
                    break
                
                time_diff = out_row['frame_id'] - last_frame['frame_id']
                
                sample = {
                    'target_x': float(out_row['x']),
                    'target_y': float(out_row['y']),
                    'time_diff': float(time_diff)
                }
                
                for col in ['x_norm', 'y_norm', 'vx', 'vy', 's', 'a',
                           'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
                           'ball_x_norm', 'ball_y_norm', 'dist_to_ball',
                           'velocity_ball_alignment', 'moving_toward_ball',
                           'kinetic_energy', 'momentum', 'dist_to_sideline',
                           'in_red_zone', 'role_id']:
                    if col in last_frame.index:
                        sample[f'input_{col}'] = float(last_frame[col])
                    else:
                        sample[f'input_{col}'] = 0.0
                
                pairs.append(sample)
                created += 1
    
    training_df = pd.DataFrame(pairs)
    print(f"  Created {len(training_df):,} training pairs")
    
    # Prepare features
    feature_cols = [col for col in training_df.columns if col.startswith('input_')]
    feature_cols.append('time_diff')
    
    X = training_df[feature_cols].values
    y = training_df[['target_x', 'target_y']].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    models = {}
    scores = {}
    
    # XGBoost
    print("  → Training XGBoost...")
    xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    ))
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
    print(f"     RMSE: {xgb_rmse:.4f} yards")
    
    models['XGBoost'] = xgb_model
    scores['XGBoost'] = xgb_rmse
    
    # LightGBM
    print("  → Training LightGBM...")
    lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=350,
        max_depth=9,
        learning_rate=0.045,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
    print(f"     RMSE: {lgb_rmse:.4f} yards")
    
    models['LightGBM'] = lgb_model
    scores['LightGBM'] = lgb_rmse
    
    # Calculate weights
    total_inv_rmse = sum(1.0 / rmse for rmse in scores.values())
    weights = {name: (1.0 / rmse) / total_inv_rmse for name, rmse in scores.items()}
    
    print(f"\n  Ensemble Weights:")
    for name, weight in weights.items():
        print(f"    {name}: {weight:.2%}")
    
    return models, weights, feature_cols


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Detect paths
    train_folder, test_input_path, test_targets_path, sample_sub_path = detect_data_paths()
    
    # Load training data (8 weeks)
    weeks = ['w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08']
    train_input, train_output = load_training_data(train_folder, weeks)
    
    # Engineer features
    train_input = engineer_spatiotemporal_features(train_input)
    
    # Create datasets for transformer
    full_dataset = NFLTrajectoryDataset(train_input, train_output, max_samples=150_000)
    
    # Split for training/validation
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n[INFO] Dataset Split")
    print(f"  Training: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    
    # Train transformer model
    transformer_model, transformer_rmse = train_transformer_model(train_dataset, val_dataset, config)
    
    # Train gradient boosting ensemble
    gb_models, gb_weights, gb_feature_cols = train_gradient_boosting_ensemble(train_input, train_output)
    
    del train_input, train_output
    gc.collect()
    
    # Load test data
    print("\n[PHASE 6] Generating Test Predictions")
    print("-" * 60)
    
    test_input = pd.read_csv(test_input_path)
    test_targets = pd.read_csv(test_targets_path)
    
    print(f"  Test input: {len(test_input):,} rows")
    print(f"  Test targets: {len(test_targets):,} predictions")
    
    # Engineer test features
    test_input = engineer_spatiotemporal_features(test_input)
    
    # Generate predictions (hybrid ensemble)
    predictions = []
    
    transformer_model.eval()
    
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
                player_input = player_input.sort_values('frame_id')
                last_frame = player_input.iloc[-1]
                time_diff = target_frame - last_frame['frame_id']
                
                # Transformer prediction
                input_features = torch.FloatTensor([
                    last_frame['x_norm'], last_frame['y_norm'],
                    last_frame['vx'], last_frame['vy'],
                    last_frame['s'], last_frame['a'],
                    last_frame['dir_sin'], last_frame['dir_cos'],
                    last_frame['o_sin'], last_frame['o_cos'],
                    last_frame['ball_x_norm'], last_frame['ball_y_norm'],
                    last_frame['dist_to_ball'], last_frame['velocity_ball_alignment'],
                    last_frame['kinetic_energy'], last_frame['momentum'],
                    last_frame['dist_to_sideline'], last_frame['in_red_zone']
                ]).unsqueeze(0).to(device)
                
                use_amp = torch.cuda.is_available()
                role_id = torch.LongTensor([[int(last_frame['role_id'])]]).to(device, non_blocking=use_amp)
                time_diff_t = torch.FloatTensor([[float(time_diff)]]).to(device, non_blocking=use_amp)
                
                context = torch.cuda.amp.autocast() if use_amp else torch.no_grad()
                with torch.no_grad(), context:
                    trans_pred = transformer_model(input_features, role_id, time_diff_t)
                    trans_pred = trans_pred.cpu().numpy()[0]
                    trans_pred[0] *= 120.0
                    trans_pred[1] *= 53.3
                
                # Gradient boosting predictions
                gb_features = []
                for col in gb_feature_cols:
                    if col == 'time_diff':
                        gb_features.append(float(time_diff))
                    elif col.startswith('input_'):
                        feat_name = col[6:]
                        gb_features.append(float(last_frame[feat_name]) if feat_name in last_frame.index else 0.0)
                
                gb_features = np.array([gb_features])
                
                xgb_pred = gb_models['XGBoost'].predict(gb_features)[0]
                lgb_pred = gb_models['LightGBM'].predict(gb_features)[0]
                
                # Ensemble (50% transformer, 25% XGBoost, 25% LightGBM)
                pred_x = 0.5 * trans_pred[0] + 0.25 * xgb_pred[0] + 0.25 * lgb_pred[0]
                pred_y = 0.5 * trans_pred[1] + 0.25 * xgb_pred[1] + 0.25 * lgb_pred[1]
                
                # Apply constraints
                pred_x = float(np.clip(pred_x, 0.0, 120.0))
                pred_y = float(np.clip(pred_y, 0.0, 53.3))
            
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
            'x': 60.0,
            'y': 26.65
        })
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
    print(f"\nTransformer RMSE: {transformer_rmse:.4f} yards")
    print(f"Target: < 0.500 yards")
    print("=" * 80)


if __name__ == '__main__':
    main()
