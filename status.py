"""
Quick Status Checker - See training status at a glance
"""
from pathlib import Path
from datetime import datetime
import subprocess
import sys

CHECKPOINT_DIR = Path("checkpoints")

def check_status():
    """Quick status check of training."""
    print("\n" + "="*80)
    print("NFL BIG DATA BOWL 2026 - TRAINING STATUS")
    print("="*80)
    
    # Check if Python is running
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Measure-Object"],
                capture_output=True,
                text=True
            )
            python_running = "Count" in result.stdout and int(result.stdout.split()[-1]) > 0
        else:
            result = subprocess.run(["pgrep", "-f", "python"], capture_output=True)
            python_running = len(result.stdout.strip()) > 0
    except:
        python_running = None
    
    print(f"\n📍 Python Process Status:")
    if python_running is None:
        print("   ⚠️  Could not check (run manually: Get-Process python)")
    elif python_running:
        print("   ✅ Python is running")
    else:
        print("   ❌ No Python processes found")
    
    # Check for submission.csv
    print(f"\n📄 Submission File:")
    submission = Path("submission.csv")
    if submission.exists():
        size = submission.stat().st_size
        mtime = datetime.fromtimestamp(submission.stat().st_mtime)
        print(f"   ✅ EXISTS - {size:,} bytes")
        print(f"   📅 Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick check of content
        try:
            with open(submission, 'r') as f:
                lines = f.readlines()
            print(f"   📊 Lines: {len(lines):,} (expected: 5,838 including header)")
            if len(lines) >= 2:
                print(f"   Sample: {lines[1].strip()[:60]}...")
        except:
            print("   ⚠️  Could not read file")
    else:
        print("   ❌ Not found")
    
    # Check latest checkpoint
    print(f"\n💾 Latest Checkpoint:")
    if not CHECKPOINT_DIR.exists():
        print("   ❌ No checkpoints directory")
    else:
        all_checkpoints = list(CHECKPOINT_DIR.glob("*.pkl"))
        if not all_checkpoints:
            print("   ❌ No checkpoint files")
        else:
            all_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest = all_checkpoints[0]
            mtime = datetime.fromtimestamp(latest.stat().st_mtime)
            print(f"   📁 {latest.name}")
            print(f"   📅 {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Determine phase
            if latest.name.startswith("06"):
                print("   ✅ Phase 6: TRAINING COMPLETE!")
            elif latest.name.startswith("04"):
                print("   ⚠️  Phase 4: Training in progress or stopped")
            elif latest.name.startswith("03"):
                print("   ⚠️  Phase 3: Feature selection done, training not started")
            elif latest.name.startswith("02"):
                print("   ⚠️  Phase 2: Feature engineering done")
            elif latest.name.startswith("01"):
                print("   ⚠️  Phase 1: Data loaded only")
    
    # Check training log
    print(f"\n📝 Training Log:")
    log_file = CHECKPOINT_DIR / "training_log.txt"
    if not log_file.exists():
        print("   ❌ No training log found")
    else:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if lines:
                # Get last few non-empty lines
                last_lines = [l.strip() for l in lines[-10:] if l.strip()]
                if last_lines:
                    print(f"   📊 Total entries: {len(lines)}")
                    print(f"   🕐 Last entry: {last_lines[-1][:70]}...")
                    
                    # Check for completion
                    if "SUCCESS" in last_lines[-1] or "🎉" in last_lines[-1]:
                        print("   ✅ Training completed successfully!")
                    elif "Training:" in last_lines[-1] or "→" in last_lines[-1]:
                        print("   ⚙️  Training in progress...")
                    else:
                        print("   ⚠️  Status unclear - check full log")
        except:
            print("   ⚠️  Could not read log file")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    
    if submission.exists() and latest.name.startswith("06"):
        print("✅ STATUS: COMPLETE")
        print("   Your submission.csv is ready!")
        print("   Run: python checkpoint_manager.py view 06_final_submission_*.pkl")
    elif python_running:
        print("⚙️  STATUS: TRAINING IN PROGRESS")
        print("   Monitor: Get-Content checkpoints\\training_log.txt -Wait")
        print("   Check: python checkpoint_manager.py list")
    else:
        print("⚠️  STATUS: STOPPED OR NOT STARTED")
        print("   Check: python checkpoint_manager.py list")
        print("   Start: python generate_advanced_submission.py")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    check_status()
