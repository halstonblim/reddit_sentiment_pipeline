"""Pure‑function helpers for daily aggregation."""

from __future__ import annotations
import pandas as pd


def summary_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with daily & subreddit aggregates.

    Expects columns:
        retrieved_at  - UTC timestamp or ISO-date string
        subreddit     - subreddit name
        sentiment     - numeric score (e.g. −1 … 1)
        score         - numeric weight / post score

    Output columns:
        date               (datetime.date)
        subreddit          (string)
        mean_sentiment
        weighted_sentiment
        count
    """
    # Normalize retrieved_at to datetime and extract calendar day
    df = df.copy()
    df["date"] = pd.to_datetime(df["retrieved_at"]).dt.date
    
    # Group by date and subreddit
    grouped = df.groupby(["date", "subreddit"])
    
    # Aggregate metrics
    result = grouped.agg(
        mean_sentiment=("sentiment", "mean"),
        count=("sentiment", "count"),
    ).reset_index()
    
    # Calculate weighted sentiment for each group
    weighted_sentiments = []
    
    for (date, subreddit), group in grouped:
        w_sum = (group["score"] * group["sentiment"]).sum()
        w_tot = group["score"].sum() or 1.0  # Avoid division by zero
        weighted_sent = w_sum / w_tot
        weighted_sentiments.append(weighted_sent)
    
    result["weighted_sentiment"] = weighted_sentiments
    
    # Ensure consistent column order
    result = result[["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"]]
    
    return result
