# summarizer/test_summarize.py
import pandas as pd
import datetime
from summarize import summary_from_df

def test_compute_daily_summary():
    # Prepare dummy data with same date and subreddit
    data = [
        {"retrieved_at": "2024-01-01 09:20:03", "subreddit": "python", "sentiment": 0.6, "score": 10},
        {"retrieved_at": "2024-01-01 10:45:30", "subreddit": "python", "sentiment": -0.2, "score": 5},
        {"retrieved_at": "2024-01-01 12:30:45", "subreddit": "python", "sentiment": 1.0, "score": 8},
    ]
    df = pd.DataFrame(data)

    # Compute summary
    summary = summary_from_df(df)

    # Expected date
    date1 = pd.to_datetime("2024-01-01").date()

    # Calculate expected values
    mean_sent = df['sentiment'].mean()
    count = len(df)
    weighted_sent = ((df['score'] * df['sentiment']).sum() / df['score'].sum())

    # Build expected DataFrame with columns in the correct order
    expected = pd.DataFrame([
        {
            "date": date1,
            "subreddit": "python",
            "mean_sentiment": mean_sent,
            "weighted_sentiment": weighted_sent,
            "count": count,
        },
    ])

    # Ensure column order is correct
    expected = expected[["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"]]
    
    # Sort and reset indices for comparison
    summary = summary.sort_values(["date", "subreddit"]).reset_index(drop=True)
    expected = expected.sort_values(["date", "subreddit"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(summary, expected)

def test_compute_summary_by_date_and_subreddit():
    # Prepare dummy data: same date, different subreddits, mixed sentiments
    data = [
        {"retrieved_at": "2024-01-01 09:20:03", "subreddit": "python", "sentiment": 0.6, "score": 10},
        {"retrieved_at": "2024-01-01 10:45:30", "subreddit": "python", "sentiment": -0.2, "score": 5},
        {"retrieved_at": "2024-01-01 12:30:45", "subreddit": "datascience", "sentiment": 1.0, "score": 8},
        {"retrieved_at": "2024-01-01 14:15:22", "subreddit": "datascience", "sentiment": 0.5, "score": 3},
        {"retrieved_at": "2024-01-02 09:00:00", "subreddit": "python", "sentiment": 0.8, "score": 15},
    ]
    df = pd.DataFrame(data)

    # Compute summary
    summary = summary_from_df(df)

    # Expected dates
    date1 = pd.to_datetime("2024-01-01").date()
    date2 = pd.to_datetime("2024-01-02").date()

    # Group data for calculations
    df['date'] = pd.to_datetime(df['retrieved_at']).dt.date
    grouped = df.groupby(['date', 'subreddit'])

    # Create expected DataFrame manually for comparison
    expected_data = []
    
    for (date, subreddit), group in grouped:
        mean_sent = group['sentiment'].mean()
        count = len(group)
        weighted_sent = ((group['score'] * group['sentiment']).sum() / group['score'].sum())
        
        expected_data.append({
            "date": date,
            "subreddit": subreddit,
            "mean_sentiment": mean_sent,
            "weighted_sentiment": weighted_sent,
            "count": count,
        })
    
    expected = pd.DataFrame(expected_data)
    
    # Ensure column order is correct
    expected = expected[["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"]]

    # Sort for consistent comparison
    summary = summary.sort_values(["date", "subreddit"]).reset_index(drop=True)
    expected = expected.sort_values(["date", "subreddit"]).reset_index(drop=True)
    
    # Verify results
    pd.testing.assert_frame_equal(summary, expected)

