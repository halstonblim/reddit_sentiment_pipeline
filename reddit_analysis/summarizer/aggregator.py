"""Pure‑function helpers for daily aggregation."""

from __future__ import annotations
import pandas as pd
import numpy as np


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
        community_weighted_sentiment
        count
    """
    # Normalize retrieved_at to datetime and extract calendar day
    df = df.copy()
    df["date"] = pd.to_datetime(df["retrieved_at"]).dt.date
    
    # Group by date and subreddit
    grouped = df.groupby(["date", "subreddit"])
    
    # Aggregate metrics
    result = grouped.agg(
        # First calculate raw mean_sentiment
        raw_mean_sentiment=("sentiment", "mean"),
        count=("sentiment", "count"),
    ).reset_index()
    
    # Apply transformation to raw_mean_sentiment to get values in range [-1, 1] instead of [0, 1]
    result["mean_sentiment"] = 2 * result["raw_mean_sentiment"] - 1
    
    # Remove the raw mean column
    result = result.drop(columns="raw_mean_sentiment")
    
    # Calculate community weighted sentiment for each group
    community_weighted_sentiments = []
    
    for (date, subreddit), group in grouped:
        # Community weighted sentiment calculation
        # Using the formula: community_weighted_sentiment = log1p(max(0, score)) × sentiment_signed
        # where sentiment_signed = (sentiment * 2 − 1)
        # Ensure scores are non-negative for log1p
        sentiment_signed = (group["sentiment"] * 2 - 1)
        # Floor scores at 0 to prevent log1p from receiving negative values
        non_negative_scores = np.maximum(group["score"], 0)
        log_scores = np.log1p(non_negative_scores)
        community_weighted = (log_scores * sentiment_signed).mean()
        community_weighted_sentiments.append(community_weighted)
    
    result["community_weighted_sentiment"] = community_weighted_sentiments
    
    # Ensure consistent column order
    result = result[["date", "subreddit", "mean_sentiment", "community_weighted_sentiment", "count"]]
    
    return result
