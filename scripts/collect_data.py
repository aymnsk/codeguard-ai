#!/usr/bin/env python3
"""
Data Collection Script for CodeGuard AI
Collects commit data from GitHub repositories
"""

import os
import sys
import pandas as pd
import git
from datetime import datetime
import time
from tqdm import tqdm
import re
import random
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeDataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.repos_dir = self.data_dir / "repos"
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        
        # Small, popular, well-maintained Python repos
        self.repos = [
            "https://github.com/psf/requests",
            "https://github.com/numpy/numpy",
            "https://github.com/pandas-dev/pandas",
            "https://github.com/scikit-learn/scikit-learn",
            "https://github.com/django/django",
            "https://github.com/pallets/flask",
            "https://github.com/encode/httpx",
            "https://github.com/tiangolo/fastapi",
            "https://github.com/python/cpython",
            "https://github.com/pytest-dev/pytest"
        ]
        
        # System-friendly limits
        self.max_commits_per_repo = 80
        self.max_file_size_kb = 100
        
        # Keywords for bug detection
        self.bug_keywords = ['fix', 'bug', 'error', 'issue', 'crash', 'fail', 'memory', 'leak', 
                            'segfault', 'exception', 'patch', 'resolve', 'solve']
        
        self.feature_keywords = ['add', 'feature', 'implement', 'new', 'support', 'enhancement']
        
    def clone_repository(self, repo_url):
        """Clone a repository with error handling"""
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = self.repos_dir / repo_name
        
        if repo_path.exists():
            logger.info(f"Repository {repo_name} already exists, skipping clone")
            return repo_path
        
        logger.info(f"Cloning {repo_name}...")
        try:
            # Shallow clone for speed and space
            git.Repo.clone_from(repo_url, repo_path, depth=30, branch='main')
            logger.info(f"✅ Successfully cloned {repo_name}")
            return repo_path
        except Exception as e:
            logger.error(f"❌ Failed to clone {repo_name}: {e}")
            return None
    
    def extract_commit_info(self, commit):
        """Extract structured information from a commit"""
        try:
            message = commit.message.strip()
            
            # Categorize commit type
            is_bug_fix = any(keyword in message.lower() for keyword in self.bug_keywords)
            is_feature = any(keyword in message.lower() for keyword in self.feature_keywords)
            
            # Extract code changes
            code_changes = []
            files_changed = []
            insertions = 0
            deletions = 0
            
            if commit.parents:
                try:
                    diff = commit.parents[0].diff(commit, create_patch=True)
                    
                    for file_diff in diff:
                        if file_diff.a_path:
                            files_changed.append(file_diff.a_path)
                        
                        if file_diff.diff:
                            diff_text = file_diff.diff.decode('utf-8', errors='ignore')
                            if len(diff_text) < 5000:  # Limit size
                                code_changes.append(diff_text[:1000])
                            
                            # Count + and - lines
                            insertions += diff_text.count('\n+') - diff_text.count('\n+++')
                            deletions += diff_text.count('\n-') - diff_text.count('\n---')
                except:
                    pass
            
            # Extract issue numbers
            issue_numbers = re.findall(r'#(\d+)', message)
            
            return {
                'repo': commit.repo.git_dir.split('/')[-2],
                'commit_hash': commit.hexsha[:8],
                'author': str(commit.author),
                'message': message,
                'is_bug_fix': is_bug_fix,
                'is_feature': is_feature,
                'files_changed': len(set(files_changed)),
                'files_list': ', '.join(list(set(files_changed))[:3]),  # First 3 files
                'insertions': insertions,
                'deletions': deletions,
                'total_changes': insertions + deletions,
                'issue_numbers': ', '.join(issue_numbers[:3]),
                'timestamp': commit.committed_datetime.isoformat(),
                'date': commit.committed_datetime.date().isoformat(),
                'code_changes': ' '.join(code_changes)[:1500] if code_changes else '',
                'message_length': len(message),
                'message_word_count': len(message.split()),
                'has_test_change': any('test' in f.lower() for f in files_changed)
            }
        except Exception as e:
            logger.warning(f"Error processing commit: {e}")
            return None
    
    def extract_commit_data(self, repo_path):
        """Extract commit data from a repository"""
        try:
            repo = git.Repo(repo_path)
            commits_data = []
            repo_name = os.path.basename(repo_path)
            
            logger.info(f"Processing commits from {repo_name}...")
            
            # Get commits with progress bar
            commits = list(repo.iter_commits(max_count=self.max_commits_per_repo))
            
            for commit in tqdm(commits, desc=f"Commits from {repo_name}"):
                commit_info = self.extract_commit_info(commit)
                if commit_info:
                    commits_data.append(commit_info)
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            
            logger.info(f"✅ Extracted {len(commits_data)} commits from {repo_name}")
            return commits_data
            
        except Exception as e:
            logger.error(f"❌ Error processing {repo_path}: {e}")
            return []
    
    def collect_all(self, max_repos=5):
        """Main collection function"""
        all_commits = []
        selected_repos = random.sample(self.repos, min(max_repos, len(self.repos)))
        
        logger.info(f"Collecting data from {len(selected_repos)} repositories")
        
        for i, repo_url in enumerate(selected_repos, 1):
            logger.info(f"\n[{i}/{len(selected_repos)}] Processing {repo_url}")
            
            repo_path = self.clone_repository(repo_url)
            if repo_path:
                commits = self.extract_commit_data(repo_path)
                all_commits.extend(commits)
                
                # Be nice to GitHub
                time.sleep(2)
        
        # Create DataFrame
        if all_commits:
            df = pd.DataFrame(all_commits)
            
            # Save to CSV
            output_path = self.data_dir / "commits_dataset.csv"
            df.to_csv(output_path, index=False)
            
            # Also save a sample for testing
            sample_path = self.data_dir / "commits_sample.csv"
            df.sample(min(100, len(df))).to_csv(sample_path, index=False)
            
            # Print statistics
            logger.info(f"\n{'='*50}")
            logger.info("📊 COLLECTION STATISTICS")
            logger.info(f"{'='*50}")
            logger.info(f"Total commits collected: {len(df)}")
            logger.info(f"Bug fixes: {df['is_bug_fix'].sum()} ({df['is_bug_fix'].mean():.1%})")
            logger.info(f"Features: {df['is_feature'].sum()} ({df['is_feature'].mean():.1%})")
            logger.info(f"Average files changed: {df['files_changed'].mean():.1f}")
            logger.info(f"Average changes per commit: {df['total_changes'].mean():.1f}")
            logger.info(f"Dataset saved to: {output_path}")
            logger.info(f"Sample saved to: {sample_path}")
            
            return df
        else:
            logger.error("❌ No commits collected")
            return None
    
    def generate_sample_data(self, num_samples=200):
        """Generate sample data if GitHub collection fails"""
        logger.info("Generating sample data...")
        
        sample_commits = []
        bug_messages = [
            "Fix memory leak in data parser", "Fix segmentation fault on null input",
            "Bug fix: handle empty response", "Fix issue with concurrent access",
            "Patch security vulnerability", "Fix race condition in worker pool",
            "Resolve memory corruption", "Fix infinite loop in parser",
            "Correct buffer overflow", "Fix type error in validation"
        ]
        
        feature_messages = [
            "Add new API endpoint", "Implement caching layer",
            "Add support for JSON export", "New feature: bulk operations",
            "Add user authentication", "Implement rate limiting",
            "Add monitoring dashboard", "New config options",
            "Add test coverage", "Implement logging system"
        ]
        
        other_messages = [
            "Update documentation", "Refactor code structure",
            "Optimize performance", "Clean up imports",
            "Update dependencies", "Code style improvements",
            "Add comments", "Remove unused code"
        ]
        
        for i in range(num_samples):
            if i < num_samples * 0.3:  # 30% bug fixes
                message = random.choice(bug_messages)
                is_bug = True
                is_feature = False
            elif i < num_samples * 0.6:  # 30% features
                message = random.choice(feature_messages)
                is_bug = False
                is_feature = True
            else:  # 40% other
                message = random.choice(other_messages)
                is_bug = False
                is_feature = False
            
            sample_commits.append({
                'repo': random.choice(['requests', 'numpy', 'pandas', 'django', 'flask']),
                'commit_hash': f"abc{i:03d}",
                'author': f"dev{i}@example.com",
                'message': message,
                'is_bug_fix': is_bug,
                'is_feature': is_feature,
                'files_changed': random.randint(1, 5),
                'files_list': 'src/main.py, tests/test_main.py',
                'insertions': random.randint(5, 50),
                'deletions': random.randint(0, 20),
                'total_changes': random.randint(5, 70),
                'issue_numbers': f"#{random.randint(100, 999)}" if random.random() > 0.7 else "",
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().date().isoformat(),
                'code_changes': f"diff --git... (+{random.randint(5, 20)} lines)",
                'message_length': len(message),
                'message_word_count': len(message.split()),
                'has_test_change': random.random() > 0.7
            })
        
        df = pd.DataFrame(sample_commits)
        output_path = self.data_dir / "commits_dataset.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Generated {len(df)} sample commits")
        logger.info(f"Dataset saved to: {output_path}")
        
        return df

def main():
    """Main function with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect commit data from GitHub')
    parser.add_argument('--max-repos', type=int, default=5, help='Maximum repositories to process')
    parser.add_argument('--sample-only', action='store_true', help='Generate sample data only')
    parser.add_argument('--output', type=str, default='data/commits_dataset.csv', help='Output file path')
    
    args = parser.parse_args()
    
    collector = CodeDataCollector()
    
    if args.sample_only:
        df = collector.generate_sample_data()
    else:
        try:
            df = collector.collect_all(max_repos=args.max_repos)
            if df is None:
                logger.warning("GitHub collection failed, generating sample data...")
                df = collector.generate_sample_data()
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            df = collector.generate_sample_data()
    
    if df is not None:
        print("\n📋 Dataset Preview:")
        print(df[['repo', 'message', 'is_bug_fix', 'files_changed']].head(10).to_string())
    
    return df

if __name__ == "__main__":
    df = main()
