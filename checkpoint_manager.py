"""
Checkpoint Manager - View and analyze training checkpoints
"""
import pickle
import json
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = Path("checkpoints")

def format_timestamp(filename):
    """Extract and format timestamp from checkpoint filename."""
    parts = filename.stem.split('_')
    if len(parts) >= 2:
        timestamp_str = '_'.join(parts[-2:])
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp_str
    return "Unknown"

def list_checkpoints():
    """List all available checkpoints."""
    if not CHECKPOINT_DIR.exists():
        print("No checkpoints directory found.")
        return
    
    checkpoints = {
        "01_data_loaded": [],
        "02_features_engineered": [],
        "03_features_selected": [],
        "04_model_": [],  # Partial match
        "04_training_complete": [],
        "06_final_submission": []
    }
    
    # Collect all checkpoint files
    for pkl_file in CHECKPOINT_DIR.glob("*.pkl"):
        for key in checkpoints.keys():
            if pkl_file.name.startswith(key):
                checkpoints[key].append(pkl_file)
                break
    
    # Also check JSON files
    for json_file in CHECKPOINT_DIR.glob("*.json"):
        if json_file.name.startswith("04_training_complete"):
            checkpoints["04_training_complete"].append(json_file)
    
    print("\n" + "=" * 80)
    print("CHECKPOINT SUMMARY")
    print("=" * 80)
    
    phases = [
        ("01_data_loaded", "Phase 1: Data Loading"),
        ("02_features_engineered", "Phase 2: Feature Engineering"),
        ("03_features_selected", "Phase 3: Feature Selection"),
        ("04_model_", "Phase 4: Model Training (Individual Roles)"),
        ("04_training_complete", "Phase 4: Training Complete"),
        ("06_final_submission", "Phase 6: Final Submission")
    ]
    
    latest_phase = 0
    latest_checkpoint = None
    
    for key, description in phases:
        files = checkpoints[key]
        if files:
            # Sort by modification time
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest = files[0]
            timestamp = format_timestamp(latest)
            
            print(f"\n✓ {description}")
            print(f"  Latest: {latest.name}")
            print(f"  Time: {timestamp}")
            print(f"  Count: {len(files)} checkpoint(s)")
            
            # Determine phase number
            if key.startswith("06"):
                phase_num = 6
            elif key.startswith("04"):
                phase_num = 4
            elif key.startswith("03"):
                phase_num = 3
            elif key.startswith("02"):
                phase_num = 2
            elif key.startswith("01"):
                phase_num = 1
            else:
                phase_num = 0
            
            if phase_num > latest_phase:
                latest_phase = phase_num
                latest_checkpoint = latest
        else:
            print(f"\n✗ {description}")
            print(f"  Status: Not started")
    
    print("\n" + "=" * 80)
    
    if latest_checkpoint:
        print(f"\n📍 LATEST CHECKPOINT: Phase {latest_phase}")
        print(f"   File: {latest_checkpoint.name}")
        print(f"   Time: {format_timestamp(latest_checkpoint)}")
        
        if latest_phase == 6:
            print("\n   ✅ Training completed successfully!")
            print("   Your submission.csv should be ready.")
        elif latest_phase >= 4:
            print(f"\n   ⚠️  Training was in progress (Phase {latest_phase})")
            print("   Some models may have been trained.")
            print("   Consider restarting training from beginning for consistency.")
        else:
            print(f"\n   ⚠️  Training stopped at Phase {latest_phase}")
            print("   You can review the data but should restart training.")
    else:
        print("\n📍 No checkpoints found - ready for fresh start")
    
    print("\n" + "=" * 80)

def view_checkpoint_details(checkpoint_file):
    """View details of a specific checkpoint file."""
    filepath = CHECKPOINT_DIR / checkpoint_file
    
    if not filepath.exists():
        print(f"Checkpoint file not found: {checkpoint_file}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"CHECKPOINT DETAILS: {checkpoint_file}")
    print(f"{'=' * 80}")
    
    try:
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            print("\nJSON Content:")
            print(json.dumps(data, indent=2))
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print("\nCheckpoint Data:")
            for key, value in data.items():
                if isinstance(value, (list, tuple)) and len(value) > 10:
                    print(f"  {key}: {type(value).__name__} with {len(value)} items")
                    print(f"    First 3: {value[:3]}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n⚠️  Error loading checkpoint: {e}")
    
    print(f"\n{'=' * 80}")

def view_training_log(lines=50):
    """View recent training log entries."""
    log_file = CHECKPOINT_DIR / "training_log.txt"
    
    if not log_file.exists():
        print("No training log found.")
        return
    
    print(f"\n{'=' * 80}")
    print(f"TRAINING LOG (Last {lines} lines)")
    print(f"{'=' * 80}\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    recent_lines = all_lines[-lines:]
    for line in recent_lines:
        print(line.rstrip())
    
    print(f"\n{'=' * 80}")
    print(f"Total log lines: {len(all_lines)}")
    print(f"{'=' * 80}")

def clean_old_checkpoints(keep_latest=1):
    """Remove old checkpoint files, keeping only the most recent."""
    if not CHECKPOINT_DIR.exists():
        print("No checkpoints directory found.")
        return
    
    print(f"\n🗑️  Cleaning old checkpoints (keeping {keep_latest} most recent)...")
    
    checkpoint_groups = {}
    
    # Group files by prefix
    for pkl_file in CHECKPOINT_DIR.glob("*.pkl"):
        # Extract prefix (everything before the last two underscore segments which are timestamp)
        name_parts = pkl_file.stem.split('_')
        if len(name_parts) >= 3:
            prefix = '_'.join(name_parts[:-2])
        else:
            prefix = pkl_file.stem
        
        if prefix not in checkpoint_groups:
            checkpoint_groups[prefix] = []
        checkpoint_groups[prefix].append(pkl_file)
    
    deleted_count = 0
    kept_count = 0
    
    for prefix, files in checkpoint_groups.items():
        if len(files) <= keep_latest:
            kept_count += len(files)
            continue
        
        # Sort by modification time, newest first
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Keep the newest, delete the rest
        for file_to_delete in files[keep_latest:]:
            print(f"  Deleting: {file_to_delete.name}")
            file_to_delete.unlink()
            deleted_count += 1
        
        kept_count += keep_latest
    
    print(f"\n✓ Cleanup complete:")
    print(f"  Kept: {kept_count} files")
    print(f"  Deleted: {deleted_count} files")

def main():
    """Main checkpoint manager interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("\n🔧 Checkpoint Manager")
        print("\nUsage:")
        print("  python checkpoint_manager.py list              - List all checkpoints")
        print("  python checkpoint_manager.py log [N]          - View last N lines of log (default: 50)")
        print("  python checkpoint_manager.py view <filename>  - View checkpoint details")
        print("  python checkpoint_manager.py clean [N]        - Delete old checkpoints, keep N latest (default: 1)")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_checkpoints()
    
    elif command == "log":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        view_training_log(lines)
    
    elif command == "view":
        if len(sys.argv) < 3:
            print("Please specify a checkpoint filename")
            print("Example: python checkpoint_manager.py view 04_training_complete_20251019_163006.json")
        else:
            view_checkpoint_details(sys.argv[2])
    
    elif command == "clean":
        keep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        response = input(f"⚠️  This will delete old checkpoints, keeping only {keep} most recent. Continue? (y/n): ")
        if response.lower() == 'y':
            clean_old_checkpoints(keep)
        else:
            print("Cancelled.")
    
    else:
        print(f"Unknown command: {command}")
        print("Use: list, log, view, or clean")

if __name__ == "__main__":
    main()
