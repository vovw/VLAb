#!/usr/bin/env python3
"""
Script to clean up unused files from SmolVLA2 pretraining library.
This removes files that are not needed for pretraining-only workflows.
"""

import os
import shutil
from pathlib import Path

# Files and directories to remove (not needed for pretraining)
UNUSED_FILES = [
    # Utils - Robot/Environment specific
    "src/lerobot/utils/control_utils.py",
    "src/lerobot/utils/robot_utils.py", 
    "src/lerobot/utils/benchmark.py",
    "src/lerobot/utils/buffer.py",
    "src/lerobot/utils/queue.py",
    "src/lerobot/utils/visualization_utils.py",
    "src/lerobot/utils/process.py",
    
    # Dataset conversion utilities
    "src/lerobot/datasets/v2/",
    "src/lerobot/datasets/v21/",
    "src/lerobot/datasets/online_buffer.py",
    "src/lerobot/datasets/push_dataset_to_hub/",
]

# Files to keep but could be moved to optional directory
OPTIONAL_FILES = [
    "src/lerobot/configs/eval.py",  # Keep for compatibility
]

def cleanup_unused_files(dry_run=True):
    """Remove unused files from the pretraining library."""
    base_dir = Path(__file__).parent
    
    print("🧹 SmolVLA2 Pretraining Library Cleanup")
    print("=" * 50)
    
    if dry_run:
        print("🔍 DRY RUN - No files will be deleted")
    else:
        print("⚠️  LIVE RUN - Files will be permanently deleted!")
    
    print()
    
    removed_count = 0
    for file_path in UNUSED_FILES:
        full_path = base_dir / file_path
        
        if full_path.exists():
            if full_path.is_file():
                print(f"📄 {'[DRY RUN] ' if dry_run else ''}Remove file: {file_path}")
                if not dry_run:
                    full_path.unlink()
                removed_count += 1
            elif full_path.is_dir():
                print(f"📁 {'[DRY RUN] ' if dry_run else ''}Remove directory: {file_path}")
                if not dry_run:
                    shutil.rmtree(full_path)
                removed_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print()
    print(f"✅ {'Would remove' if dry_run else 'Removed'} {removed_count} files/directories")
    
    if dry_run:
        print()
        print("To actually remove files, run:")
        print("python cleanup_unused_files.py --live")

if __name__ == "__main__":
    import sys
    
    # Check if --live flag is provided
    live_run = "--live" in sys.argv
    
    if live_run:
        response = input("⚠️  Are you sure you want to permanently delete unused files? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Cleanup cancelled")
            sys.exit(1)
    
    cleanup_unused_files(dry_run=not live_run)
