#!/usr/bin/env python
"""
Summarise scored shards into one daily_summary.csv
CLI examples
------------
# Re‑summarise an explicit range
python summarizer/summarize.py --start 2025-04-18 --end 2025-04-20

# Auto‑detect missing dates (common nightly mode)
python summarizer/summarize.py
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    login,
    upload_file,
)

from aggregator import summary_from_df

# ──────────────────────────────────────────────────────────────────────────────
# Config & secrets
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # project root

# If running locally or Streamlit is not available, use dotenv
try:
    import streamlit as st
    has_streamlit = True
except ImportError:
    has_streamlit = False

if os.getenv("ENV") == "local" or not has_streamlit:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

# Helper function to get secrets
def get_secret(key, default=None):
    """Get a secret from environment variables or Streamlit secrets."""
    value = os.getenv(key)
    if value is None and has_streamlit:
        value = st.secrets.get(key, default)
    return value

# 1️⃣  repo_id from config.yaml (version‑controlled, non‑secret)
with open(ROOT / "config.yaml", "r") as fh:
    cfg = yaml.safe_load(fh)
REPO_ID: str = cfg["repo_id"]

# 2️⃣  HF_TOKEN from .env (git‑ignored) or CI secrets or Streamlit secrets
HF_TOKEN = get_secret("HF_TOKEN")  # raise at runtime if missing

SCORED_DIR = "data_scored"
SUMMARY_FILE = ROOT / "subreddit_daily_summary.csv"  # write at repo root


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    # ── Load existing summary (if any) ────────────────────────────────────────────
    if SUMMARY_FILE.exists() and SUMMARY_FILE.stat().st_size > 0:
        df_summary = pd.read_csv(SUMMARY_FILE)
        # Convert date column to datetime if it exists
        if "date" in df_summary.columns:
            df_summary["date"] = pd.to_datetime(df_summary["date"]).dt.date
    else:
        df_summary = pd.DataFrame(columns=["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"])

    # Get processed date-subreddit combinations
    processed = set()
    if not df_summary.empty and "date" in df_summary.columns and "subreddit" in df_summary.columns:
        processed = {(row["date"], row["subreddit"]) for _, row in df_summary.iterrows()}

    # Determine target dates
    if args.start and args.end:
        targets = list(
            daterange(date.fromisoformat(args.start), date.fromisoformat(args.end))
        )
    else:
        # infer from files in HF repo
        repo_files = list_repo_files(REPO_ID, repo_type="dataset")
        targets = [
            date.fromisoformat(Path(f).stem)
            for f in repo_files
            if f.startswith(f"{SCORED_DIR}/")
        ]

    new_rows = []
    for d in sorted(targets):
        remote_path = f"{SCORED_DIR}/{d}.parquet"
        try:
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=remote_path
            )
        except Exception:
            # shard for that day missing (e.g. scrape failed)
            continue

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
        
        if not df_summary_day.empty:
            new_rows.append(df_summary_day)

    if not new_rows:
        print("Nothing new to summarise.")
        return

    df_out = pd.concat([df_summary] + new_rows, ignore_index=True)
    df_out.sort_values(["date", "subreddit"], inplace=True)
    df_out.to_csv(SUMMARY_FILE, index=False)
    print(f"Updated {SUMMARY_FILE}  →  {len(df_out)} rows")

    # Push back to HF Hub
    login(token=HF_TOKEN)
    upload_file(
        path_or_fileobj=SUMMARY_FILE,
        path_in_repo=SUMMARY_FILE.name,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Update subreddit_daily_summary.csv ({date.today()})",
    )


if __name__ == "__main__":
    main()
