#!/usr/bin/env python3
"""Sync files from upstream GitHub repo and generate detailed report."""
import os
import sys
import json
from pathlib import Path
import urllib.request
import urllib.error

GITHUB_OWNER = "Yatharthg12"
GITHUB_REPO = "FirePBD-Engine"
GITHUB_BRANCH = "main"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"

LOCAL_DIR = r"E:\FirePBD_Engine"

FILES_TO_SYNC = [
    "README.md",
    "requirements.txt",
    "run_project.bat",
    "smoke_test.py",
    "frontend/dashboard.js",
    "frontend/index.html",
    "frontend/simulation.js",
    "frontend/style.css",
    "frontend/viewer.js",
    "backend/__init__.py",
    "backend/config.py",
    "backend/main.py",
    "backend/agents/__init__.py",
    "backend/agents/blueprint_agent.py",
    "backend/agents/evacuation_agent.py",
    "backend/agents/fire_agent.py",
    "backend/agents/optimization_agent.py",
    "backend/agents/report_agent.py",
    "backend/agents/risk_agent.py",
    "backend/agents/topology_agent.py",
    "backend/core/__init__.py",
    "backend/core/constants.py",
    "backend/core/geometry.py",
    "backend/core/graph_model.py",
    "backend/core/grid_model.py",
    "backend/core/simulation_state.py",
    "backend/utils/__init__.py",
    "backend/utils/floorplan_generator.py",
    "backend/utils/image_processing.py",
    "backend/utils/logger.py",
    "backend/utils/math_utils.py",
    "backend/utils/repair.py",
    "backend/utils/validation.py",
    "docs/compliance_mapping.md",
    "docs/evacuation_model_spec.md",
    "docs/fire_model_spec.md",
    "docs/system_architecture.md",
    "training/yolo_train.py",
    "training/data.yaml",
    "training/augmentation_config.yaml",
]

VERIFY_FILES = [
    "frontend/dashboard.js",
    "frontend/index.html",
    "backend/main.py",
]

def fetch_file(file_path):
    """Fetch file content from GitHub raw URL."""
    url = BASE_URL + file_path.replace("\\", "/")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)

def write_file(local_path, content):
    """Write file to local disk."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(content)
        return True
    except Exception as e:
        return False

def get_file_size(file_path):
    """Get size of file from GitHub."""
    url = BASE_URL + file_path.replace("\\", "/")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return len(response.read())
    except:
        return None

def main():
    os.chdir(LOCAL_DIR)
    
    updated_files = []
    unchanged_files = []
    errors = {}
    verify_results = {}
    
    print("="*70)
    print("FIREPBD ENGINE UPSTREAM SYNC")
    print("="*70)
    print(f"Source: {GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}")
    print(f"Target: {LOCAL_DIR}")
    print(f"Files to sync: {len(FILES_TO_SYNC)}")
    print("="*70 + "\n")
    
    for i, file_path in enumerate(FILES_TO_SYNC, 1):
        print(f"[{i:2d}/{len(FILES_TO_SYNC)}] {file_path:<50}", end=" ", flush=True)
        
        # Fetch from GitHub
        content = fetch_file(file_path)
        if isinstance(content, tuple) or content is None:
            error_msg = content[1] if isinstance(content, tuple) else "Unknown error"
            errors[file_path] = error_msg
            print(f"FAIL ({error_msg})")
            continue
        
        # Write to local
        local_path = Path(LOCAL_DIR) / file_path
        if write_file(local_path, content):
            size = len(content)
            updated_files.append(file_path)
            print(f"✓ ({size:,} bytes)")
            
            # Track sizes for verification files
            if file_path in VERIFY_FILES:
                verify_results[file_path] = {
                    "remote_size": size,
                    "local_path": str(local_path)
                }
        else:
            errors[file_path] = "Write failed"
            print("WRITE ERROR")
    
    # Verify sizes for specified files
    print("\n" + "="*70)
    print("VERIFICATION (Size Match)")
    print("="*70)
    
    verify_passed = True
    for file_path in VERIFY_FILES:
        if file_path in verify_results:
            local_path = Path(LOCAL_DIR) / file_path
            if local_path.exists():
                local_size = local_path.stat().st_size
                remote_size = verify_results[file_path]["remote_size"]
                match = local_size == remote_size
                status = "✓" if match else "✗"
                print(f"{status} {file_path}")
                print(f"    Local:  {local_size:,} bytes")
                print(f"    Remote: {remote_size:,} bytes")
                if not match:
                    verify_passed = False
            else:
                print(f"✗ {file_path} - FILE NOT FOUND")
                verify_passed = False
        else:
            print(f"✗ {file_path} - NOT IN VERIFICATION SET")
            verify_passed = False
    
    # Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files to sync:    {len(FILES_TO_SYNC)}")
    print(f"Successfully updated:   {len(updated_files)}")
    print(f"Failed/errors:          {len(errors)}")
    
    if errors:
        print("\nErrors encountered:")
        for f, e in sorted(errors.items()):
            print(f"  ✗ {f}: {e}")
    
    print(f"\nVerification: {'PASSED ✓' if verify_passed else 'FAILED ✗'}")
    print("="*70)
    
    return len(errors) == 0 and verify_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
