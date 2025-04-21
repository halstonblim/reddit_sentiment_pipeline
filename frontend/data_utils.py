from __future__ import annotations
from pathlib import Path
import os, yaml, pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# If running locally, load from .env file
# If running in Streamlit Cloud, use st.secrets
try:
    import streamlit as st
    has_streamlit = True
except ImportError:
    has_streamlit = False

# If local environment or Streamlit not available, use dotenv
if os.getenv("ENV") == "local" or not has_streamlit:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

# read repo_id from config.yaml (committed)
with open(ROOT / "config.yaml") as f:
    REPO_ID: str = yaml.safe_load(f)["repo_id"]

# Get HF_TOKEN from env vars or streamlit secrets
def get_secret(key, default=None):
    """Get a secret from environment variables or Streamlit secrets."""
    value = os.getenv(key)
    if value is None and has_streamlit:
        value = st.secrets.get(key, default)
    return value

CSV_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/subreddit_daily_summary.csv"
)

def load_summary() -> pd.DataFrame:
    """Load subreddit_daily_summary.csv from HF Hub into a DataFrame."""
    df = pd.read_csv(CSV_URL, parse_dates=["date"])
    # guarantee expected columns
    needed = {"date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Missing columns in summary CSV: {needed - set(df.columns)}")
    return df

def add_rolling(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add 7‑day (default) rolling mean columns for each subreddit."""
    df = df.copy()
    # Group by subreddit and sort by date within each group
    for subreddit, group in df.groupby("subreddit"):
        group_sorted = group.sort_values("date")
        # Calculate rolling average for this subreddit
        roll_values = group_sorted["weighted_sentiment"].rolling(window).mean()
        # Map back to original dataframe using index
        df.loc[group_sorted.index, f"roll_{window}"] = roll_values
    return df

def get_subreddit_colors(subreddits):
    """Return a consistent color scheme for subreddits."""
    # Standard color palette for visualization
    color_palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
    ]
    
    # Map each subreddit to a color
    return {sub: color_palette[i % len(color_palette)] for i, sub in enumerate(sorted(subreddits))}
