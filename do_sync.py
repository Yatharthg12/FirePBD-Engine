#!/usr/bin/env python
"""
Comprehensive GitHub Repository Synchronizer
Efficiently downloads and synchronizes all tracked files from the upstream repository.
"""
import urllib.request
import urllib.error
import json
import os
import sys
from pathlib import Path

class GitHubSync:
    def __init__(self, repo_owner, repo_name, branch, local_dir):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.local_dir = local_dir
        self.github_api_base = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.raw_base = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}"
        self.success_count = 0
        self.failure_count = 0
        self.downloaded_files = []
        
    def _make_request(self, url, timeout=15):
        """Make HTTP request with proper headers."""
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            raise Exception(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise Exception(f"URL Error: {e.reason}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def _get_tree_recursively(self, tree_url, path_prefix=""):
        """Recursively get all file paths from GitHub tree."""
        files = []
        try:
            response = self._make_request(tree_url + "?recursive=1")
            data = json.loads(response)
            if 'tree' in data:
                for item in data['tree']:
                    if item['type'] == 'blob':  # It's a file
                        files.append(path_prefix + item['path'])
        except:
            pass
        return files
    
    def get_all_files(self):
        """Get list of all tracked files to sync."""
        return [
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
    
    def download_file(self, relative_path):
        """Download and save a single file."""
        url = f"{self.raw_base}/{relative_path}"
        local_path = os.path.join(self.local_dir, relative_path)
        
        # Create parent directory
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            content = self._make_request(url, timeout=20)
            with open(local_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def sync(self):
        """Synchronize all tracked files."""
        files = self.get_all_files()
        total = len(files)
        
        print(f"\n{'='*80}")
        print(f"FirePBD-Engine Repository Synchronizer")
        print(f"{'='*80}")
        print(f"Repository: {self.repo_owner}/{self.repo_name}")
        print(f"Branch: {self.branch}")
        print(f"Local Directory: {self.local_dir}")
        print(f"Files to sync: {total}")
        print(f"{'='*80}\n")
        
        for idx, file_path in enumerate(files, 1):
            status, error = self.download_file(file_path)
            if status:
                self.success_count += 1
                self.downloaded_files.append(file_path)
                status_str = "✓"
            else:
                self.failure_count += 1
                status_str = "✗"
                error_short = error[:50] if error else "Unknown error"
                print(f"[{idx:2d}/{total}] {status_str} {file_path:<50} {error_short}")
                continue
            
            print(f"[{idx:2d}/{total}] {status_str} {file_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Synchronization Complete")
        print(f"{'='*80}")
        print(f"Successful: {self.success_count}/{total}")
        print(f"Failed: {self.failure_count}/{total}")
        print(f"{'='*80}")
        
        if self.downloaded_files:
            print(f"\nDownloaded Files ({len(self.downloaded_files)}):")
            for f in sorted(self.downloaded_files):
                print(f"  • {f}")
        
        if self.failure_count == 0:
            print(f"\n✓ SUCCESS: All files synchronized from GitHub!")
            return 0
        else:
            print(f"\n⚠ {self.failure_count} file(s) failed to download.")
            return 1

if __name__ == "__main__":
    syncer = GitHubSync(
        repo_owner="Yatharthg12",
        repo_name="FirePBD-Engine",
        branch="main",
        local_dir=r"e:\FirePBD_Engine"
    )
    exit_code = syncer.sync()
    sys.exit(exit_code)
