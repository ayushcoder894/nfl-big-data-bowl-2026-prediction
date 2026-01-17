"""
Quick test to verify checkpoint functionality works correctly.
"""
from pathlib import Path
import pickle
import json
from datetime import datetime

# Test checkpoint directory creation
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

def save_checkpoint(name: str, data: dict, suffix: str = ""):
    """Save checkpoint data to file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}{suffix}.pkl"
    filepath = CHECKPOINT_DIR / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    
    log_checkpoint(f"✓ Checkpoint saved: {filename}")
    return filepath

def save_metrics(name: str, metrics: dict):
    """Save metrics to JSON with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    filepath = CHECKPOINT_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    log_checkpoint(f"✓ Metrics saved: {filename}")
    return filepath

if __name__ == "__main__":
    print("Testing checkpoint functionality...\n")
    
    # Test 1: Log checkpoint
    log_checkpoint("=" * 60)
    log_checkpoint("🧪 TEST: Checkpoint System Verification")
    log_checkpoint("=" * 60)
    
    # Test 2: Save checkpoint data
    test_data = {
        "test_name": "checkpoint_test",
        "timestamp": datetime.now().isoformat(),
        "data_shape": (1000, 50),
        "features": ["feature_1", "feature_2", "feature_3"]
    }
    checkpoint_path = save_checkpoint("test_checkpoint", test_data)
    
    # Test 3: Save metrics
    test_metrics = {
        "rmse": 0.3350,
        "mae": 0.2150,
        "r2_score": 0.8725
    }
    metrics_path = save_metrics("test_metrics", test_metrics)
    
    # Test 4: Verify files exist
    log_checkpoint("\n📋 Verification:")
    log_checkpoint(f"  ✓ Checkpoint directory exists: {CHECKPOINT_DIR.exists()}")
    log_checkpoint(f"  ✓ Training log exists: {CHECKPOINT_LOG.exists()}")
    log_checkpoint(f"  ✓ Test checkpoint saved: {checkpoint_path.exists()}")
    log_checkpoint(f"  ✓ Test metrics saved: {metrics_path.exists()}")
    
    # Test 5: Read back the data
    log_checkpoint("\n🔍 Reading back data:")
    with open(checkpoint_path, "rb") as f:
        loaded_data = pickle.load(f)
    log_checkpoint(f"  ✓ Checkpoint data loaded: {loaded_data['test_name']}")
    
    with open(metrics_path, "r") as f:
        loaded_metrics = json.load(f)
    log_checkpoint(f"  ✓ Metrics loaded: RMSE={loaded_metrics['rmse']:.4f}")
    
    log_checkpoint("\n" + "=" * 60)
    log_checkpoint("✅ ALL TESTS PASSED! Checkpoint system is ready.")
    log_checkpoint("=" * 60)
    
    print(f"\n📁 Files created in '{CHECKPOINT_DIR}/':")
    print(f"   - {CHECKPOINT_LOG.name}")
    print(f"   - {checkpoint_path.name}")
    print(f"   - {metrics_path.name}")
    print(f"\nYou can view the log with:")
    print(f"   Get-Content {CHECKPOINT_LOG} -Wait")
