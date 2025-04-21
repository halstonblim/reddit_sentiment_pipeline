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

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    login,
    upload_file,
)

from reddit_analysis.config_utils import setup_config
import praw
import logging
import pytz
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from datasets.data_files import EmptyDatasetError
from huggingface_hub import HfApi
import yaml

# Add parent directory to path to allow importing config_utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config_utils


def setup_logging(logs_dir: Path):
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

class RedditScraper:
    def __init__(self, cfg):
        self.config = cfg['config']
        self.secrets = cfg['secrets']
        self.paths = cfg['paths']
        self.logger = logging.getLogger(__name__)
        
        # Initialize Reddit client
        self.reddit = praw.Reddit(
            client_id=self.secrets.get('REDDIT_CLIENT_ID'),
            client_secret=self.secrets.get('REDDIT_CLIENT_SECRET'),
            user_agent=self.secrets.get('REDDIT_USER_AGENT')
        )
        
        self.timezone = pytz.timezone(self.config['timezone'])
        
        # Ensure output directory exists
        self.paths['raw_dir'].mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory set to: {self.paths['raw_dir']}")

    def get_posts(self, subreddit_config):
        """Fetch posts and comments from a subreddit based on configuration."""
        subreddit_name = subreddit_config['name']
        post_limit = subreddit_config['post_limit']
        comment_limit = subreddit_config['comment_limit']
        retrieved_at = datetime.now(self.timezone)
        records = []

        subreddit = self.reddit.subreddit(subreddit_name)
        
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
                for comment in submission.comments[:comment_limit]:
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
        reset_ts = self.reddit.auth.limits.get('reset_timestamp')
        reset_time = (
            datetime.fromtimestamp(reset_ts, tz=self.timezone)
            .strftime("%Y-%m-%d %I:%M:%S %p %Z")
            if reset_ts else "Unknown"
        )

        self.logger.info("Reddit API Rate Limit Info")
        self.logger.info(f"Requests used:      {self.reddit.auth.limits.get('used')}")
        self.logger.info(f"Requests remaining: {self.reddit.auth.limits.get('remaining')}")
        self.logger.info(f"Resets at:          {reset_time}")

    def save_to_csv(self, df, date_str):
        """Save DataFrame to CSV file."""
        output_path = self.paths['raw_dir'] / f"{date_str}.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")
        return output_path

    def save_to_parquet(self, df, date_str):
        """Save DataFrame to parquet file."""
        output_path = self.paths['raw_dir'] / f"{date_str}.parquet"
        df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")
        return output_path

    def upload_to_hf(self, df, date_str):
        if not self.config.get('push_to_hf', False):
            self.logger.info("Skipping Hugging Face upload as configured")
            return

        try:
            api = HfApi(token=self.secrets.get('HF_TOKEN'))
            repo_id = self.config.get('repo_id')
            hf_raw_dir = self.paths['hf_raw_dir']

            try:
                current_date = datetime.strptime(date_str, "%Y-%m-%d")
                prev_date = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
                prev_file_path = f"{hf_raw_dir}/{prev_date}.parquet"

                self.logger.info(f"Checking for previous day's file: {prev_file_path}")
                downloaded_path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename=prev_file_path,
                    repo_type="dataset",
                )

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

            parquet_path = self.save_to_parquet(df, date_str)
            path_in_repo = f"{hf_raw_dir}/{date_str}.parquet"
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=self.secrets.get('HF_TOKEN')
            )
            self.logger.info(f"Uploaded {len(df)} rows for {date_str} → {path_in_repo}")

        except Exception as e:
            self.logger.error(f"Failed to upload to Hugging Face: {e}", exc_info=True)


def main(date_str=None):
    # Load configuration first
    cfg = config_utils.setup_config()

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
    logger.info(f"Processing data for date: {date_str}")

    all_records = []
    for sub_cfg in cfg['config']['subreddits']:
        logger.info(f"Processing subreddit: {sub_cfg['name']}")
        df = scraper.get_posts(sub_cfg)
        all_records.append(df)

    combined_df = pd.concat(all_records, ignore_index=True)
    logger.info(f"Total records collected: {len(combined_df)}")

    scraper.save_to_csv(combined_df, date_str)
    scraper.upload_to_hf(combined_df, date_str)
    scraper.print_rate_limit_info()

    logger.info("Reddit scraper completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape today's top posts, label and dedupe them using --date (YYYY-MM-DD).")
    parser.add_argument('--date', type=str, required=True, help="Date label for this run (used in filenames and deduplication)")
    args = parser.parse_args()
    main(args.date)
