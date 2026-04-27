#!/usr/bin/env python3
"""Sync files from upstream GitHub repo."""
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
        with urllib.request.urlopen(url) as response:
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
        with urllib.request.urlopen(url) as response:
            return len(response.read())
    except:
        return None

def main():
    os.chdir(LOCAL_DIR)
    
    updated_files = []
    unchanged_files = []
    errors = {}
    sizes_upstream = {}
    sizes_local = {}
    
    print(f"Starting sync from {GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}")
    print(f"Target directory: {LOCAL_DIR}\n")
    
    for file_path in FILES_TO_SYNC:
        print(f"Fetching {file_path}...", end=" ")
        
        # Fetch from GitHub
        content = fetch_file(file_path)
        if isinstance(content, tuple) or content is None:
            error_msg = content[1] if isinstance(content, tuple) else "Unknown error"
            errors[file_path] = error_msg
            print(f"FAILED ({error_msg})")
            continue
        
        # Write to local
        local_path = Path(LOCAL_DIR) / file_path
        if write_file(local_path, content):
            size = len(content)
            updated_files.append(file_path)
            print(f"OK ({size} bytes)")
            
            # Track sizes for verification files
            if file_path in VERIFY_FILES:
                sizes_local[file_path] = size
                sizes_upstream[file_path] = size
        else:
            errors[file_path] = "Write failed"
            print("WRITE FAILED")
    
    # Verify sizes for specified files
    print("\n" + "="*60)
    print("SIZE VERIFICATION")
    print("="*60)
    
    verify_results = {}
    for file_path in VERIFY_FILES:
        if file_path in updated_files:
            local_path = Path(LOCAL_DIR) / file_path
            local_size = local_path.stat().st_size
            
            # Get upstream size
            upstream_size = get_file_size(file_path)
            
            match = "✓ MATCH" if local_size == upstream_size else "✗ MISMATCH"
            verify_results[file_path] = {
                "upstream": upstream_size,
                "local": local_size,
                "match": local_size == upstream_size
            }
            print(f"{file_path}: upstream={upstream_size}, local={local_size} {match}")
        else:
            print(f"{file_path}: NOT UPDATED")
    
    # Print summary
    print("\n" + "="*60)
    print("SYNC SUMMARY")
    print("="*60)
    print(f"Updated: {len(updated_files)} files")
    if updated_files:
        for f in updated_files:
            print(f"  ✓ {f}")
    
    print(f"\nErrors: {len(errors)} files")
    if errors:
        for f, e in errors.items():
            print(f"  ✗ {f}: {e}")
    
    print(f"\nVerification:")
    all_match = True
    for f, result in verify_results.items():
        status = "✓" if result["match"] else "✗"
        print(f"  {status} {f}: {result['local']} bytes (upstream: {result['upstream']})")
        if not result["match"]:
            all_match = False
    
    return len(errors) == 0 and all_match

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
