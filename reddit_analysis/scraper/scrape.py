#!/usr/bin/env python
"""
Scrape Reddit posts and comments.
CLI examples
------------
# Scrape data for a specific date
python -m reddit_analysis.scraper.scrape --date 2025-04-20
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    login,
    upload_file,
    HfApi
)
import praw
import logging
import pytz
from tqdm import tqdm

from reddit_analysis.config_utils import setup_config

class RedditAPI:
    """Wrapper class for Reddit API interactions that can be mocked for testing."""
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def get_subreddit(self, name: str):
        return self.reddit.subreddit(name)
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        return {
            'used': self.reddit.auth.limits.get('used'),
            'remaining': self.reddit.auth.limits.get('remaining'),
            'reset_timestamp': self.reddit.auth.limits.get('reset_timestamp')
        }

class FileManager:
    """Wrapper class for file operations that can be mocked for testing."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.base_dir / f"{filename}.csv"
        df.to_csv(path, index=False)
        return path
    
    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.base_dir / f"{filename}.parquet"
        df.to_parquet(path, index=False)
        return path
    
    def read_parquet(self, filename: str) -> pd.DataFrame:
        path = self.base_dir / f"{filename}.parquet"
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
    
    def list_files(self, prefix: str) -> List[str]:
        return self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )

class RedditScraper:
    def __init__(
        self,
        cfg: Dict[str, Any],
        reddit_api: Optional[RedditAPI] = None,
        file_manager: Optional[FileManager] = None,
        hf_manager: Optional[HuggingFaceManager] = None
    ):
        self.config = cfg['config']
        self.secrets = cfg['secrets']
        self.paths = cfg['paths']
        self.logger = logging.getLogger(__name__)
        
        # Initialize services with dependency injection
        self.reddit_api = reddit_api or RedditAPI(
            client_id=self.secrets.get('REDDIT_CLIENT_ID'),
            client_secret=self.secrets.get('REDDIT_CLIENT_SECRET'),
            user_agent=self.secrets.get('REDDIT_USER_AGENT')
        )
        
        self.file_manager = file_manager or FileManager(self.paths['raw_dir'])
        
        if self.config.get('push_to_hf', False):
            self.hf_manager = hf_manager or HuggingFaceManager(
                token=self.secrets.get('HF_TOKEN'),
                repo_id=self.config.get('repo_id'),
                repo_type=self.config.get('repo_type', 'dataset')
            )
        else:
            self.hf_manager = hf_manager
        
        self.timezone = pytz.timezone(self.config['timezone'])
        self.logger.info(f"Output directory set to: {self.paths['raw_dir']}")

    def get_posts(self, subreddit_config: Dict[str, Any]) -> pd.DataFrame:
        """Fetch posts and comments from a subreddit based on configuration."""
        subreddit_name = subreddit_config['name']
        post_limit = subreddit_config['post_limit']
        comment_limit = subreddit_config['comment_limit']
        retrieved_at = datetime.now(self.timezone)
        records = []

        subreddit = self.reddit_api.get_subreddit(subreddit_name)
        
        self.logger.info(f"Fetching {post_limit} posts from r/{subreddit_name}")
        
        for submission in tqdm(
            subreddit.top(time_filter="day", limit=post_limit),
            total=post_limit,
            desc=f"Processing r/{subreddit_name}"
        ):
            # Add post record
            records.append({
                "subreddit": subreddit_name,
                "created_at": datetime.fromtimestamp(submission.created_utc, tz=self.timezone),
                "retrieved_at": retrieved_at,
                "type": "post",
                "text": submission.title + "\n\n" + submission.selftext,
                "score": submission.score,
                "post_id": submission.id,
                "parent_id": None
            })

            # Get top comments if comment_limit > 0
            if comment_limit > 0:
                submission.comment_sort = 'top'
                submission.comments.replace_more(limit=0)
                comments = getattr(submission.comments, '_comments', [])[:comment_limit]
                for comment in comments:
                    records.append({
                        "subreddit": subreddit_name,
                        "created_at": datetime.fromtimestamp(comment.created_utc, tz=self.timezone),
                        "retrieved_at": retrieved_at,
                        "type": "comment",
                        "text": comment.body,
                        "score": comment.score,
                        "post_id": comment.id,
                        "parent_id": comment.parent_id
                    })

        return pd.DataFrame(records)

    def print_rate_limit_info(self):
        """Print current Reddit API rate limit information."""
        limits = self.reddit_api.get_rate_limit_info()
        reset_ts = limits.get('reset_timestamp')
        reset_time = (
            datetime.fromtimestamp(reset_ts, tz=self.timezone)
            .strftime("%Y-%m-%d %I:%M:%S %p %Z")
            if reset_ts else "Unknown"
        )

        self.logger.info("Reddit API Rate Limit Info")
        self.logger.info(f"Requests used:      {limits.get('used')}")
        self.logger.info(f"Requests remaining: {limits.get('remaining')}")
        self.logger.info(f"Resets at:          {reset_time}")

    def process_date(self, date_str: str) -> None:
        """Process data for a specific date."""
        self.logger.info(f"Processing data for date: {date_str}")

        all_records = []
        for sub_cfg in self.config['subreddits']:
            self.logger.info(f"Processing subreddit: {sub_cfg['name']}")
            df = self.get_posts(sub_cfg)
            all_records.append(df)

        combined_df = pd.concat(all_records, ignore_index=True)
        self.logger.info(f"Total records collected: {len(combined_df)}")

        # Save to CSV
        self.file_manager.save_csv(combined_df, date_str)
        
        # Upload to HuggingFace if configured
        if self.config.get('push_to_hf', False):
            self._upload_to_hf(combined_df, date_str)
        
        self.print_rate_limit_info()
        self.logger.info("Reddit scraper completed successfully")

    def _upload_to_hf(self, df: pd.DataFrame, date_str: str) -> None:
        """Upload data to HuggingFace Hub."""
        try:
            current_date = datetime.strptime(date_str, "%Y-%m-%d")
            prev_date = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
            prev_file_path = f"{self.paths['hf_raw_dir']}/{prev_date}.parquet"

            self.logger.info(f"Checking for previous day's file: {prev_file_path}")
            try:
                downloaded_path = self.hf_manager.download_file(prev_file_path)
                existing_df = pd.read_parquet(downloaded_path)
                existing_ids = set(existing_df["post_id"].tolist())
                Path(downloaded_path).unlink()

                original_count = len(df)
                df = df[~df["post_id"].isin(existing_ids)]
                filtered_count = len(df)
                self.logger.info(f"Filtered {original_count - filtered_count} duplicates")

                if df.empty:
                    self.logger.info("No new posts to upload after deduplication")
                    return
                    
            except Exception as e:
                self.logger.warning(f"Could not fetch/process previous file: {e}")

            parquet_path = self.file_manager.save_parquet(df, date_str)
            path_in_repo = f"{self.paths['hf_raw_dir']}/{date_str}.parquet"
            self.hf_manager.upload_file(str(parquet_path), path_in_repo)
            self.logger.info(f"Uploaded {len(df)} rows for {date_str} → {path_in_repo}")
        except Exception as e:
            self.logger.error(f"Failed to upload to Hugging Face: {e}")
            raise

def setup_logging(logs_dir: Path) -> logging.Logger:
    """Set up logging configuration using logs_dir from config."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with current date
    log_file = logs_dir / f"reddit_scraper_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def main(date_str: str = None) -> None:
    # Load configuration first
    cfg = setup_config()

    # Initialize logging with configured logs_dir
    logs_dir = cfg['paths']['logs_dir']
    logger = setup_logging(logs_dir)
    logger.info("Starting Reddit scraper...")

    # Validate environment variables
    required_env_vars = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
    if cfg['config'].get('push_to_hf', False):
        required_env_vars.append("HF_TOKEN")
    missing = [v for v in required_env_vars if not cfg['secrets'].get(v) and not os.getenv(v)]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Instantiate and run
    logger.info("Initializing Reddit scraper...")
    scraper = RedditScraper(cfg)

    if date_str is None:
        date_str = datetime.now(pytz.timezone(cfg['config']['timezone'])).strftime("%Y-%m-%d")
    
    scraper.process_date(date_str)

if __name__ == "__main__":
    from reddit_analysis.common_metrics import run_with_metrics
    parser = argparse.ArgumentParser(description='Scrape Reddit posts and comments.')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD date to process')
    args = parser.parse_args()
    run_with_metrics("scrape", main, args.date)
