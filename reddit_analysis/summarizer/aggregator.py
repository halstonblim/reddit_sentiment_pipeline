"""Pure‑function helpers for daily aggregation."""

from __future__ import annotations
import pandas as pd
import numpy as np


def summary_from_df(df: pd.DataFrame, gamma_post: float = 0.3) -> pd.DataFrame:
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
    
    # Calculate engagement-adjusted sentiment (EAS) for each group
    # 1. Ensure 'score' is numeric
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    # 2. Compute base weights (1 + log1p(score))
    weights_base = 1 + np.log1p(df["score_num"].clip(lower=0))
    # 3. Apply post weight multiplier
    weights = weights_base * np.where(df.get("type", None) == "post", gamma_post, 1.0)
    df["weight"] = weights
    # 4. Compute EAS per group: weighted average of sentiment
    community_weighted_sentiments = []
    for (date, subreddit), group in grouped:
        w = group["weight"]
        s = group["sentiment"]
        eas = (w * s).sum() / w.sum() if w.sum() > 0 else 0
        community_weighted_sentiments.append(eas)
    result["community_weighted_sentiment"] = community_weighted_sentiments
    
    # Normalize community_weighted_sentiment to range [-1,1]
    result["community_weighted_sentiment"] = 2 * result["community_weighted_sentiment"] - 1
    
    # Ensure consistent column order
    result = result[["date", "subreddit", "mean_sentiment", "community_weighted_sentiment", "count"]]
    
    return result
