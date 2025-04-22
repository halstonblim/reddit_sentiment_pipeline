#!/usr/bin/env python
"""
Summarise scored shards into one daily_summary.csv
CLI examples
------------
# Summarize data for a specific date
python -m reddit_analysis.summarizer.summarize --date 2025-04-20
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    login,
    upload_file,
    HfApi
)

from reddit_analysis.config_utils import setup_config
from reddit_analysis.summarizer.aggregator import summary_from_df

class FileManager:
    """Wrapper class for file operations that can be mocked for testing."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def read_csv(self, filename: str) -> pd.DataFrame:
        path = self.base_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(columns=["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"])
        return pd.read_csv(path)
    
    def write_csv(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.base_dir / filename
        df.to_csv(path, index=False)
        return path
    
    def read_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

class HuggingFaceManager:
    """Wrapper class for HuggingFace Hub operations that can be mocked for testing."""
    def __init__(self, token: str, repo_id: str, repo_type: str = "dataset"):
        self.token = token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(token=token)
    
    def download_file(self, path_in_repo: str) -> Path:
        return Path(hf_hub_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            filename=path_in_repo,
            token=self.token
        ))
    
    def upload_file(self, local_path: str, path_in_repo: str):
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.token
        )

class SummaryManager:
    def __init__(
        self,
        cfg: Dict[str, Any],
        file_manager: Optional[FileManager] = None,
        hf_manager: Optional[HuggingFaceManager] = None
    ):
        self.config = cfg['config']
        self.secrets = cfg['secrets']
        self.paths = cfg['paths']
        
        # Initialize services with dependency injection
        self.file_manager = file_manager or FileManager(self.paths['root'])
        self.hf_manager = hf_manager or HuggingFaceManager(
            token=self.secrets['HF_TOKEN'],
            repo_id=self.config['repo_id'],
            repo_type=self.config.get('repo_type', 'dataset')
        )
    
    def get_processed_combinations(self) -> Set[Tuple[date, str]]:
        """Get set of already processed date-subreddit combinations."""
        df_summary = self.file_manager.read_csv(self.paths['summary_file'].name)
        if df_summary.empty or "date" not in df_summary.columns or "subreddit" not in df_summary.columns:
            return set()
        
        # Convert date column to datetime if it exists
        df_summary["date"] = pd.to_datetime(df_summary["date"]).dt.date
        return {(row["date"], row["subreddit"]) for _, row in df_summary.iterrows()}
    
    def process_date(self, date_str: str, overwrite: bool = False) -> None:
        """Process and summarize data for a specific date."""
        # Get processed combinations
        processed = self.get_processed_combinations()
        
        # Download and process the file
        remote_path = f"{self.paths['hf_scored_dir']}/{date_str}.parquet"
        try:
            local_path = self.hf_manager.download_file(remote_path)
        except Exception as e:
            print(f"Error: Could not download file for date {date_str}: {str(e)}")
            return
        
        # Process the file
        df_day = self.file_manager.read_parquet(local_path)
        
        # Column sanity check
        if not {"retrieved_at", "subreddit", "sentiment", "score"}.issubset(df_day.columns):
            raise ValueError(f"{remote_path} missing expected columns")
        
        # Get summary by date and subreddit
        df_summary_day = summary_from_df(df_day)
        
        # Filter out already processed date-subreddit combinations
        if processed and not overwrite: 
            df_summary_day = df_summary_day[
                ~df_summary_day.apply(lambda row: (row["date"], row["subreddit"]) in processed, axis=1)
            ]
        
        # Check if there's anything new to add
        if df_summary_day.empty:
            print("Nothing new to summarise for this date.")
            return
        
        # Load existing summary
        df_summary = self.file_manager.read_csv(self.paths['summary_file'].name)
        
        # If overwrite is set, remove existing rows for the same date
        if overwrite:
            df_summary = df_summary[df_summary['date'] != date_str]  
        
        # Combine with existing summary
        if df_summary.empty:
            df_out = df_summary_day
        else:
            df_out = pd.concat([df_summary, df_summary_day], ignore_index=True)
        df_out['date'] = pd.to_datetime(df_out['date']).dt.date
        
        # Sort the combined DataFrame by date and subreddit
        df_out.sort_values(["date", "subreddit"], inplace=True)  # Ensure sorting before saving
        
        # Save updated summary
        self.file_manager.write_csv(df_out, self.paths['summary_file'].name)
        print(f"Updated {self.paths['summary_file'].name}  →  {len(df_out)} rows")
        
        # Push back to HF Hub
        self.hf_manager.upload_file(
            str(self.paths['summary_file']),
            self.paths['summary_file'].name,
        )

def main(date_str: str = None, overwrite: bool = False) -> None:
    if date_str is None:
        raise ValueError("Date argument is required")
    
    # Parse date string to date object
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}, expected YYYY-MM-DD")
    
    # Load configuration
    cfg = setup_config()
    
    # Initialize summary manager
    manager = SummaryManager(cfg)
    
    # Process the date
    manager.process_date(date_str, overwrite)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize scored Reddit data for a specific date.')
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD date to process')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing rows in the summary CSV')
    args = parser.parse_args()
    main(args.date, args.overwrite)
