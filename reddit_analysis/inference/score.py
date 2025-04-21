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
from datetime import date, timedelta
from pathlib import Path

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


from reddit_analysis.config_utils import setup_config

def list_dates_in_dir(prefix: str, hf_api, repo_id, repo_type) -> set:
    """List dates (YYYY-MM-DD) for Parquet files under a folder on the HF repo."""
    files = hf_api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    dates = set()
    for fn in files:
        if fn.startswith(f"{prefix}/") and fn.endswith('.parquet'):
            parts = Path(fn).stem
            dates.add(parts)
    return dates


def download_raw_file(date: str, hf_api, repo_id, repo_type, hf_raw_dir, hf_token) -> Path:
    """Download raw Parquet for given date."""
    out_path = hf_hub_download(repo_id=repo_id,
                               repo_type=repo_type,
                               filename=f"{hf_raw_dir}/{date}.parquet",
                               token=hf_token)
    return Path(out_path)


def score_date(date: str, cfg):
    """Process a single date: download, score, save, and upload."""
    print(f"Scoring date: {date}")
    
    # Extract config values
    repo_id = cfg['config']['repo_id']
    repo_type = cfg['config'].get('repo_type', 'dataset')
    raw_dir = cfg['paths']['raw_dir']
    scored_dir = cfg['paths']['scored_dir']
    hf_raw_dir = cfg['paths']['hf_raw_dir']
    hf_scored_dir = cfg['paths']['hf_scored_dir']
    batch_size = cfg['config'].get('batch_size', 16)
    replicate_model = cfg['config']['replicate_model']
    hf_token = cfg['secrets']['HF_TOKEN']
    replicate_api_token = cfg['secrets']['REPLICATE_API_TOKEN']
    
    # Initialize clients
    hf_api = HfApi(token=hf_token)
    replicate_client = replicate.Client(api_token=replicate_api_token)
    
    raw_path = download_raw_file(date, hf_api, repo_id, repo_type, hf_raw_dir, hf_token)
    df = pd.read_parquet(raw_path)

    sentiments = []
    confidences = []
    texts = df['text'].tolist()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # call replicate model with JSON-formatted list of strings
        result = replicate_client.run(
            replicate_model,
            input={'texts': json.dumps(batch)}
        )
        # extract outputs
        preds = result.get('predicted_labels', [])
        confs = result.get('confidences', [])
        sentiments.extend(preds)
        confidences.extend(confs)

    df['sentiment'] = sentiments
    df['confidence'] = confidences

    # Write out scored Parquet
    scored_path = scored_dir / f"{date}.parquet"
    scored_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scored_path, index=False)

    # Upload scored file to HF dataset
    hf_api.upload_file(
        path_or_fileobj=str(scored_path),
        path_in_repo=f"{hf_scored_dir}/{date}.parquet",
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )
    print(f"Uploaded scored file for {date} to {repo_id}/{hf_scored_dir}/{date}.parquet")


def main(date_arg: str = None):
    if date_arg is None:
        raise ValueError("Date argument is required")
        
    # Load configuration
    cfg = setup_config()
    
    # Check if REPLICATE_API_TOKEN is available
    if 'REPLICATE_API_TOKEN' not in cfg['secrets']:
        raise ValueError("REPLICATE_API_TOKEN is required for scoring")
    
    # Extract required config values
    repo_id = cfg['config']['repo_id']
    repo_type = cfg['config'].get('repo_type', 'dataset')
    hf_raw_dir = cfg['paths']['hf_raw_dir']
    hf_scored_dir = cfg['paths']['hf_scored_dir']
    hf_token = cfg['secrets']['HF_TOKEN']
    
    # Initialize HF API
    hf_api = HfApi(token=hf_token)
    
    # Fetch available dates
    raw_dates = list_dates_in_dir(hf_raw_dir, hf_api, repo_id, repo_type)
    scored_dates = list_dates_in_dir(hf_scored_dir, hf_api, repo_id, repo_type)

    print(raw_dates)
    
    # Check if date_arg exists in raw_dates
    if date_arg not in raw_dates:
        print(f"No raw file found for date {date_arg}")
        return
        
    # Check if date_arg already exists in scored_dates
    if date_arg in scored_dates:
        print(f"Scored file already exists for date {date_arg}")
        return
    
    # Score the specified date
    score_date(date_arg, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score raw HF dataset files via Replicate.')
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD date to process')
    args = parser.parse_args()
    main(args.date)
