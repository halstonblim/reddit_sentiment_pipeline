import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, timedelta

from data_utils import load_summary, add_rolling, get_subreddit_colors

st.set_page_config(page_title="Reddit Sentiment Trends", layout="wide")
st.title("Reddit Sentiment Monitor")

# ── Load & transform data ────────────────────────────────────────────────────
df = load_summary()

# Get colors for each subreddit
subreddits = df["subreddit"].unique()
subreddit_colors = get_subreddit_colors(subreddits)

# Define time format to use across all charts
time_format = "%m/%d/%Y"

# ── Weighted sentiment line chart for all subreddits ───────────────────────────
st.subheader("Score-Weighted Sentiment by Subreddit")

# Create line chart for weighted sentiment by subreddit
weighted_chart = alt.Chart(df).mark_line().encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format=time_format)),
    y=alt.Y("weighted_sentiment:Q", title="Weighted Sentiment Score"),
    color=alt.Color(
        "subreddit:N", 
        scale=alt.Scale(domain=list(subreddits), range=list(subreddit_colors.values())),
        legend=alt.Legend(title="Subreddit")
    ),
    tooltip=["date", "subreddit", "weighted_sentiment"]
).properties(height=300).interactive()

st.altair_chart(weighted_chart, use_container_width=True)

# ── Average sentiment line chart for all subreddits ───────────────────────────
st.subheader("Average Sentiment by Subreddit")

# Create a selection for tooltip highlighting
nearest = alt.selection_point(
    nearest=True, on="mouseover", fields=["date"], empty=False
)

# Create line chart for mean sentiment by subreddit
line_chart = alt.Chart(df).mark_line().encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format=time_format)),
    y=alt.Y("mean_sentiment:Q", title="Mean Sentiment"),
    color=alt.Color(
        "subreddit:N", 
        scale=alt.Scale(domain=list(subreddits), range=list(subreddit_colors.values())),
        legend=alt.Legend(title="Subreddit")
    ),
    tooltip=["date", "subreddit", "mean_sentiment"]
).properties(height=300)

# Add points for interactive selection
points = line_chart.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
).add_selection(nearest)

# Create tooltip rule
tooltip_rule = alt.Chart(df).mark_rule(color="gray").encode(
    x="date:T"
).transform_filter(nearest)

# Combine charts
combined_chart = (line_chart + points + tooltip_rule).interactive()
st.altair_chart(combined_chart, use_container_width=True)

# ── Bar chart for post counts by subreddit (side-by-side) ────────────────────
st.subheader("Daily Post Counts by Subreddit")

# Create grouped bar chart for post counts by date and subreddit
bar_chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format=time_format)),
    y=alt.Y("count:Q", title="Post Count"),
    xOffset="subreddit:N",  # This creates the side-by-side grouping
    color=alt.Color(
        "subreddit:N", 
        scale=alt.Scale(domain=list(subreddits), range=list(subreddit_colors.values())),
        legend=alt.Legend(title="Subreddit")
    ),
    tooltip=["date", "subreddit", "count"]
).properties(height=300).interactive()

st.altair_chart(bar_chart, use_container_width=True)

# ── Latest metrics for each subreddit ─────────────────────────────────────────
st.subheader("Latest Metrics")

# Get the most recent data for each subreddit
latest_by_subreddit = df.sort_values("date").groupby("subreddit").last().reset_index()

# Display metrics in columns
cols = st.columns(len(latest_by_subreddit))
for i, (_, row) in enumerate(latest_by_subreddit.iterrows()):
    with cols[i]:
        st.markdown(f"**{row['subreddit']}**")
        st.metric("Weighted Sentiment", f"{row['weighted_sentiment']:.2f}")
        st.metric("Mean Sentiment", f"{row['mean_sentiment']:.2f}")
        st.metric("Posts", int(row["count"]))

st.caption("Data source: Hugging Face dataset – updated nightly")
