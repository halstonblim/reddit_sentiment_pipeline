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

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    login,
    upload_file,
)

from reddit_analysis.config_utils import setup_config
from reddit_analysis.summarizer.aggregator import summary_from_df

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def daterange(start: date, end: date):
    """Generate a range of dates from start to end, inclusive."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(date_str: str = None) -> None:
    if date_str is None:
        raise ValueError("Date argument is required")
    
    # Parse date string to date object
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}, expected YYYY-MM-DD")
    
    # Load configuration
    cfg = setup_config()
    
    # Extract configuration values
    repo_id = cfg['config']['repo_id']
    repo_type = cfg['config'].get('repo_type', 'dataset')
    scored_dir = cfg['paths']['scored_dir']
    hf_scored_dir = cfg['paths']['hf_scored_dir']
    summary_file = cfg['paths']['summary_file']
    hf_token = cfg['secrets']['HF_TOKEN']
    
    # Load existing summary (if any)
    if Path(summary_file).exists() and Path(summary_file).stat().st_size > 0:
        df_summary = pd.read_csv(summary_file)
        # Convert date column to datetime if it exists
        if "date" in df_summary.columns:
            df_summary["date"] = pd.to_datetime(df_summary["date"]).dt.date
    else:
        df_summary = pd.DataFrame(columns=["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"])

    # Get processed date-subreddit combinations
    processed = set()
    if not df_summary.empty and "date" in df_summary.columns and "subreddit" in df_summary.columns:
        processed = {(row["date"], row["subreddit"]) for _, row in df_summary.iterrows()}

    # Check if the target date file exists
    remote_path = f"{hf_scored_dir}/{date_str}.parquet"
    try:
        # Try to download the file
        local = hf_hub_download(
            repo_id=repo_id, 
            repo_type=repo_type, 
            filename=remote_path,
            token=hf_token
        )
    except Exception as e:
        print(f"Error: Could not download file for date {date_str}: {str(e)}")
        return

    # Process the file
    df_day = pq.read_table(local).to_pandas()
    
    # Column sanity check
    if not {"retrieved_at", "subreddit", "sentiment", "score"}.issubset(df_day.columns):
        raise ValueError(f"{remote_path} missing expected columns")
    
    # Get summary by date and subreddit
    df_summary_day = summary_from_df(df_day)
    
    # Filter out already processed date-subreddit combinations
    if processed:
        df_summary_day = df_summary_day[
            ~df_summary_day.apply(lambda row: (row["date"], row["subreddit"]) in processed, axis=1)
        ]
    
    # Check if there's anything new to add
    if df_summary_day.empty:
        print("Nothing new to summarise for this date.")
        return

    # Combine with existing summary
    df_out = pd.concat([df_summary, df_summary_day], ignore_index=True)
    df_out.sort_values(["date", "subreddit"], inplace=True)
    df_out.to_csv(summary_file, index=False)
    print(f"Updated {summary_file}  →  {len(df_out)} rows")

    # Push back to HF Hub
    upload_file(
        path_or_fileobj=summary_file,
        path_in_repo=Path(summary_file).name,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Update subreddit_daily_summary.csv for {date_str}",
        token=hf_token
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize scored Reddit data for a specific date.')
    parser.add_argument('--date', type=str, required=True, help='YYYY-MM-DD date to process')
    args = parser.parse_args()
    main(args.date)
