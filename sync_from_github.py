#!/usr/bin/env python
"""
Sync local workspace with upstream GitHub repository.
Downloads all tracked files from main branch.
"""
import urllib.request
import ssl
import os
from pathlib import Path

# Handle SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

github_base = 'https://raw.githubusercontent.com/Yatharthg12/FirePBD-Engine/main'
workspace_path = r'e:\FirePBD_Engine'

files = [
    'README.md', 'requirements.txt', 'run_project.bat', 'smoke_test.py',
    'backend/config.py', 'backend/__init__.py', 'backend/main.py',
    'backend/agents/__init__.py', 'backend/agents/blueprint_agent.py',
    'backend/agents/evacuation_agent.py', 'backend/agents/fire_agent.py',
    'backend/agents/optimization_agent.py', 'backend/agents/report_agent.py',
    'backend/agents/risk_agent.py', 'backend/agents/topology_agent.py',
    'backend/core/__init__.py', 'backend/core/constants.py',
    'backend/core/geometry.py', 'backend/core/graph_model.py',
    'backend/core/grid_model.py', 'backend/core/simulation_state.py',
    'backend/utils/__init__.py', 'backend/utils/floorplan_generator.py',
    'backend/utils/image_processing.py', 'backend/utils/logger.py',
    'backend/utils/math_utils.py', 'backend/utils/repair.py',
    'backend/utils/validation.py', 'frontend/index.html', 'frontend/style.css',
    'frontend/dashboard.js', 'frontend/simulation.js', 'frontend/viewer.js',
    'docs/compliance_mapping.md', 'docs/evacuation_model_spec.md',
    'docs/fire_model_spec.md', 'docs/system_architecture.md',
    'training/data.yaml', 'training/yolo_train.py', 'training/augmentation_config.yaml'
]

print(f'Starting to download {len(files)} files from GitHub...')
print(f'Workspace: {workspace_path}')
print()

success_count = 0
failure_count = 0
downloaded = []

for i, file in enumerate(files, 1):
    url = f'{github_base}/{file}'
    local_path = os.path.join(workspace_path, file)
    
    # Create directory if needed
    directory = os.path.dirname(local_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
        with open(local_path, 'wb') as f:
            f.write(content)
        downloaded.append(file)
        success_count += 1
        print(f'[{i:2d}/{len(files)}] ✓ {file}')
    except Exception as e:
        failure_count += 1
        error_msg = str(e)[:45]
        print(f'[{i:2d}/{len(files)}] ✗ {file} ({error_msg})')

print()
print('=' * 70)
print('Download Summary:')
print(f'  Successful: {success_count}')
print(f'  Failed: {failure_count}')
print('=' * 70)
print('Downloaded files:')
for f in sorted(downloaded):
    print(f'  • {f}')

if success_count == len(files):
    print()
    print('SUCCESS: All files synchronized from GitHub! ✓')
else:
    print()
    print(f'WARNING: {failure_count} files failed to download.')
