#!/usr/bin/env python
"""
Test script for config_utils module.
This allows us to verify that our common configuration loading works properly.
"""
import argparse
import os
from pprint import pprint
import reddit_analysis.config_utils as config_utils

def main():
    """Test the config_utils module."""
    print("Testing config_utils.py")
    
    # Load the configuration
    cfg = config_utils.setup_config()
    
    # Print the configuration (excluding sensitive values)
    print("\nConfiguration:")
    print("--------------")
    print(f"Project root: {cfg['paths']['root']}")
    print(f"Repo ID: {cfg['config'].get('repo_id', 'Not specified')}")
    
    # Print directory configurations
    print("\nLocal Directory Paths:")
    print("--------------------")
    print(f"Raw data directory: {cfg['paths']['raw_dir']}")
    print(f"Scored data directory: {cfg['paths']['scored_dir']}")
    print(f"Logs directory: {cfg['paths']['logs_dir']}")
    print(f"Summary file: {cfg['paths']['summary_file']}")
    
    # Print HF repository paths
    print("\nHugging Face Repository Paths:")
    print("---------------------------")
    print(f"HF Raw data directory: {cfg['paths']['hf_raw_dir']}")
    print(f"HF Scored data directory: {cfg['paths']['hf_scored_dir']}")
    
    # Check if these directories exist
    print("\nDirectory Status:")
    print("----------------")
    for dir_name, dir_path in [
        ('Raw data', cfg['paths']['raw_dir']),
        ('Scored data', cfg['paths']['scored_dir']),
        ('Logs', cfg['paths']['logs_dir'])
    ]:
        exists = os.path.exists(dir_path)
        status = "Exists" if exists else "Does not exist"
        print(f"{dir_name} directory ({dir_path}): {status}")
    
    # Check if summary file exists
    summary_exists = os.path.exists(cfg['paths']['summary_file'])
    print(f"Summary file ({cfg['paths']['summary_file']}): {'Exists' if summary_exists else 'Does not exist'}")
    
    # Check if essential secrets are present (without printing their values)
    print("\nSecrets available:")
    print("-----------------")
    print(f"HF_TOKEN: {'Present' if 'HF_TOKEN' in cfg['secrets'] else 'Missing'}")
    print(f"REPLICATE_API_TOKEN: {'Present' if 'REPLICATE_API_TOKEN' in cfg['secrets'] else 'Missing'}")
    
    # Check Reddit API credentials
    for key in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']:
        print(f"{key}: {'Present' if key in cfg['secrets'] or os.getenv(key) else 'Missing'}")
    
    # List the subreddits from config if available
    if 'subreddits' in cfg['config']:
        print("\nConfigured subreddits:")
        print("---------------------")
        for sub in cfg['config']['subreddits']:
            print(f"- {sub.get('name', 'unnamed')}: {sub.get('post_limit', 'N/A')} posts, {sub.get('comment_limit', 'N/A')} comments")

if __name__ == "__main__":
    main() 