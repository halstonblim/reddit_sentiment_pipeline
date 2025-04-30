from __future__ import annotations
from pathlib import Path
import os
import yaml
import pandas as pd
import numpy as np
from huggingface_hub import HfApi
from datetime import datetime, timezone

# Root directory of the project
ROOT = Path(__file__).resolve().parent.parent

# Detect Streamlit runtime
try:
    import streamlit as st
    has_streamlit = True
except ImportError:
    has_streamlit = False

# Load environment variables when running locally
if os.getenv("ENV") == "local" or not has_streamlit:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

# Read Hugging Face dataset repo ID from config
with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)
REPO_ID: str = cfg["repo_id"]

# Initialize Hugging Face API client
api = HfApi()

# URL for the summary CSV in the dataset
CSV_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/subreddit_daily_summary.csv"
)


def get_secret(key: str, default=None) -> str | None:
    """Fetch a secret from environment variables or Streamlit secrets."""
    val = os.getenv(key)
    if val is None and has_streamlit:
        val = st.secrets.get(key, default)
    return val


import streamlit as st

@st.cache_data(ttl=6000, show_spinner=False)
def load_summary() -> pd.DataFrame:
    """Download and return the subreddit daily summary as a DataFrame. Cached for 10 minutes."""
    df = pd.read_csv(CSV_URL, parse_dates=["date"])
    needed = {"date", "subreddit", "mean_sentiment", "community_weighted_sentiment", "count"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise ValueError(f"Missing columns in summary CSV: {missing}")
    return df


@st.cache_data(show_spinner=False, ttl=60*60)
def load_day(date: str, subreddit: str) -> pd.DataFrame:
    """Lazy-download the parquet shard for one YYYY-MM-DD and return df slice.
    
    Args:
        date: Date string in YYYY-MM-DD format
        subreddit: Subreddit name to filter by
        
    Returns:
        DataFrame containing posts from the specified subreddit on the given day
    """
    fname = f"data_scored/{date}.parquet"
    local = api.hf_hub_download(REPO_ID, fname, repo_type="dataset")
    df_day = pd.read_parquet(local)
    return df_day[df_day["subreddit"].str.lower() == subreddit.lower()].reset_index(drop=True)


def get_last_updated_hf(repo_id: str) -> datetime:
    """
    Retrieve the dataset repo's last modified datetime via HF Hub API.
    Returns a timezone-aware datetime in UTC.
    """
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    dt: datetime = info.lastModified  # already a datetime object
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt


def get_last_updated_hf_caption() -> str:
    """
    Build a markdown-formatted caption string showing the dataset source and last update.
    Uses REPO_ID and the HF Hub API to fetch the timestamp.
    """
    # Generate dataset link and timestamp
    dataset_url = f"https://huggingface.co/datasets/{REPO_ID}"
    last_update_dt = get_last_updated_hf(REPO_ID)
    last_update = last_update_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Return the small-caption HTML/markdown string
    return (
        f"<small>"
        f"Data source: <a href='{dataset_url}' target='_blank'>{REPO_ID}</a> &bull; "
        f"Last updated: {last_update}"
        f"</small>"
    )


def add_rolling(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add a rolling mean for community_weighted_sentiment over the specified window."""
    out = df.copy()
    for sub, grp in out.groupby("subreddit"):
        grp_sorted = grp.sort_values("date")
        roll = grp_sorted["community_weighted_sentiment"].rolling(window).mean()
        out.loc[grp_sorted.index, f"roll_{window}"] = roll
    return out


def get_subreddit_colors(subreddits: list[str]) -> dict[str, str]:
    """Provide a consistent color map for each subreddit."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    return {sub: palette[i % len(palette)] for i, sub in enumerate(sorted(subreddits))}
