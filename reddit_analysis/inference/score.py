#!/usr/bin/env python
"""
Score Reddit posts and comments using Replicate.
CLI examples
------------
# Score data for a specific date
python -m reddit_analysis.inference.score --date 2025-04-20
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
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
import replicate
import json
import httpx
import re

from reddit_analysis.config_utils import setup_config

import json
import time
from typing import List, Dict

import httpx
import replicate


def setup_logging(logs_dir: Path) -> logging.Logger:
    """Set up logging configuration using logs_dir from config."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with current date
    log_file = logs_dir / f"reddit_scorer_{date.today().strftime('%Y-%m-%d')}.log"
    
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


class ReplicateAPI:
    """Wrapper class for Replicate API interactions."""
    def __init__(self, api_token: str, model: str, timeout_s: int = 1200):
        # Replicate accepts an httpx.Timeout via the `timeout=` kwarg
        self.client = replicate.Client(
            api_token=api_token,
            timeout=httpx.Timeout(timeout_s)  # same limit for connect/read/write/pool
        )
        self.model = model
        self.retries = 3                     # total attempts per batch
        self.logger = logging.getLogger(__name__)

    def predict(self, texts: List[str]) -> Dict[str, List[float]]:
        """Run sentiment analysis on a batch of texts.

        Sends payload as a *JSON string* (your requirement) and
        retries on transient HTTP/1.1 disconnects or timeouts.
        """
        payload = {"texts": json.dumps(texts)}   # keep JSON string

        for attempt in range(self.retries):
            try:
                result = self.client.run(self.model, input=payload)

                # Expected Replicate output structure
                return {
                    "predicted_labels": result.get("predicted_labels", []),
                    "confidences":      result.get("confidences", []),
                }

            except (httpx.RemoteProtocolError, httpx.ReadTimeout) as err:
                if attempt == self.retries - 1:
                    raise  # re‑raise on final failure
                backoff = 2 ** attempt            # 1 s, 2 s, 4 s …
                self.logger.warning(f"{err!s} – retrying in {backoff}s")
                time.sleep(backoff)


class FileManager:
    """Wrapper class for file operations that can be mocked for testing."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.base_dir / f"{filename}.parquet"
        df.to_parquet(path, index=False)
        return path
    
    def read_parquet(self, filename: str) -> pd.DataFrame:
        path = self.base_dir / f"{filename}"
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
        files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
        return [file for file in files if file.startswith(prefix)]
        

class SentimentScorer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        replicate_api: Optional[ReplicateAPI] = None,
        file_manager: Optional[FileManager] = None,
        hf_manager: Optional[HuggingFaceManager] = None
    ):
        self.config = cfg['config']
        self.secrets = cfg['secrets']
        self.paths = cfg['paths']
        self.logger = logging.getLogger(__name__)
        
        # Initialize services with dependency injection
        self.replicate_api = replicate_api or ReplicateAPI(
            api_token=self.secrets['REPLICATE_API_TOKEN'],
            model=self.config['replicate_model']
        )
        
        self.file_manager = file_manager or FileManager(self.paths['scored_dir'])
        
        self.hf_manager = hf_manager or HuggingFaceManager(
            token=self.secrets['HF_TOKEN'],
            repo_id=self.config['repo_id'],
            repo_type=self.config.get('repo_type', 'dataset')
        )
    
    def process_batch(self, texts: List[str]) -> tuple[List[float], List[float]]:
        """Process a batch of texts through the sentiment model."""
        result = self.replicate_api.predict(texts)
        return result['predicted_labels'], result['confidences']
    
    def get_existing_subreddits(self, date_str: str) -> set:
        """Get set of subreddits that already have scored files for the given date."""
        scored_files = self.hf_manager.list_files("data_scored_subreddit/")
        existing_subreddits = set()
        for fn in scored_files:
            if fn.startswith(f"data_scored_subreddit/{date_str}__") and fn.endswith('.parquet'):
                # Extract subreddit from filename: data_scored_subreddit/{date}__{subreddit}.parquet
                subreddit = Path(fn).stem.split('__', 1)[1]
                existing_subreddits.add(subreddit)
        return existing_subreddits
    
    def _sanitize(self, name: str) -> str:
        """
        Make subreddit safe for filenames (removes slashes, spaces, etc.).
        """
        name = name.strip().lower()
        name = re.sub(r"[^\w\-\.]", "_", name)
        return name
    
    def score_date(self, date_str: str, overwrite: bool = False) -> None:
        """Process a single date: download, score, save, and upload separate files per subreddit."""
        self.logger.info(f"Scoring date: {date_str}")
        
        # Get existing subreddits if not overwriting
        existing_subreddits = set()
        if not overwrite:
            existing_subreddits = self.get_existing_subreddits(date_str)
            if existing_subreddits:
                self.logger.info(f"Found {len(existing_subreddits)} existing subreddit files for {date_str}")
        
        # Download raw file
        raw_path = f"{self.paths['hf_raw_dir']}/{date_str}.parquet"
        local_path = self.hf_manager.download_file(raw_path)
        df = self.file_manager.read_parquet(str(local_path))
        
        # Validate required columns
        required_columns = {'text', 'score', 'post_id', 'subreddit'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Filter out existing subreddits if not overwriting
        subreddits_to_process = df['subreddit'].unique()
        if not overwrite and existing_subreddits:
            subreddits_to_process = [s for s in subreddits_to_process if s not in existing_subreddits]
            if not subreddits_to_process:
                self.logger.info(f"All subreddits already processed for {date_str}")
                return
            df = df[df['subreddit'].isin(subreddits_to_process)].copy()
            self.logger.info(f"Processing {len(subreddits_to_process)} new subreddits for {date_str}")
        
        # Process in batches
        batch_size = self.config.get('batch_size', 16)
        texts = df['text'].tolist()
        sentiments = []
        confidences = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_sentiments, batch_confidences = self.process_batch(batch)
            sentiments.extend(batch_sentiments[:len(batch)])  # Only take as many results as input texts
            confidences.extend(batch_confidences[:len(batch)])  # Only take as many results as input texts
        
        # Add results to DataFrame
        df['sentiment'] = sentiments
        df['confidence'] = confidences
        
        # Group by subreddit and save separate files
        subreddits = df['subreddit'].unique()
        self.logger.info(f"Found {len(subreddits)} subreddits to process for {date_str}")
        
        for subreddit in subreddits:
            subreddit_df = df[df['subreddit'] == subreddit].copy()
            
            # Save scored file per subreddit using sanitized subreddit
            safe_sub = self._sanitize(subreddit)
            filename = f"{date_str}__{safe_sub}"
            scored_path = self.file_manager.save_parquet(subreddit_df, filename)
            
            # Upload to HuggingFace with new path structure
            path_in_repo = f"data_scored_subreddit/{date_str}__{safe_sub}.parquet"
            self.hf_manager.upload_file(str(scored_path), path_in_repo)
            self.logger.info(f"Uploaded scored file for {date_str}/{subreddit} ({len(subreddit_df)} rows) to {self.config['repo_id']}/{path_in_repo}")

def main(date_arg: str = None, overwrite: bool = False) -> None:
    if date_arg is None:
        raise ValueError("Date argument is required")
        
    # Load configuration
    cfg = setup_config()
    
    # Initialize logging
    logger = setup_logging(cfg['paths']['logs_dir'])
    
    # Check if REPLICATE_API_TOKEN is available
    if 'REPLICATE_API_TOKEN' not in cfg['secrets']:
        raise ValueError("REPLICATE_API_TOKEN is required for scoring")
    
    # Initialize scorer
    scorer = SentimentScorer(cfg)
    
    # Check if date exists in raw files
    raw_dates = set()
    for fn in scorer.hf_manager.list_files(scorer.paths['hf_raw_dir']):
        if fn.endswith('.parquet'):
            raw_dates.add(Path(fn).stem)
    
    if date_arg not in raw_dates:
        logger.warning(f"No raw file found for date {date_arg}")
        return
    
    # Check if date already exists in scored files (check subreddit files)
    if not overwrite:
        # Get existing scored files for this date
        scored_files = scorer.hf_manager.list_files("data_scored_subreddit/")
        existing_subreddits = set()
        for fn in scored_files:
            if fn.startswith(f"data_scored_subreddit/{date_arg}__") and fn.endswith('.parquet'):
                # Extract subreddit from filename: data_scored_subreddit/{date}__{subreddit}.parquet
                subreddit = Path(fn).stem.split('__', 1)[1]
                existing_subreddits.add(subreddit)
        
        # Check what subreddits are in the raw data
        raw_path = f"{scorer.paths['hf_raw_dir']}/{date_arg}.parquet"
        try:
            local_path = scorer.hf_manager.download_file(raw_path)
            df = scorer.file_manager.read_parquet(str(local_path))
            raw_subreddits = set(df['subreddit'].unique())
            
            # If all subreddits already exist, skip processing
            if raw_subreddits.issubset(existing_subreddits):
                logger.info(f"All subreddits for date {date_arg} already scored ({len(existing_subreddits)} files)")
                return
            else:
                missing_subreddits = raw_subreddits - existing_subreddits
                logger.info(f"Some subreddits missing for {date_arg}: {missing_subreddits}")
        except Exception as e:
            logger.warning(f"Could not check existing subreddits for {date_arg}: {e}")
    
    # Score the specified date
    scorer.score_date(date_arg, overwrite)

if __name__ == '__main__':
    from reddit_analysis.common_metrics import run_with_metrics
    parser = argparse.ArgumentParser(description='Score raw HF dataset files via Replicate.')
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD date to process')    
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing scored file')
    args = parser.parse_args()
    run_with_metrics("score", main, args.date, args.overwrite)
