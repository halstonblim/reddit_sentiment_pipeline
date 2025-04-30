import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta, datetime

# Import from local modules
from data_utils import load_summary, load_day, add_rolling, get_subreddit_colors, get_last_updated_hf_caption
from text_analysis import keywords_for_df, keyword_stats

st.set_page_config(page_title="Reddit Sentiment Trends", layout="wide")
st.title("Reddit Sentiment Monitor")


# ── Load & transform data ────────────────────────────────────────────────────
df = load_summary()
last_update_caption = get_last_updated_hf_caption()

# Get colors for each subreddit
subreddits = df["subreddit"].unique()
subreddit_colors = get_subreddit_colors(subreddits)

# Define time format to use across all charts
time_format = "%m/%d/%Y"

# Get date range from the dataset for the form
min_date = df["date"].min().date()
max_date = df["date"].max().date()

# ── Community weighted sentiment line chart for all subreddits ───────────────
st.subheader("Community Weighted Sentiment by Subreddit")

# Create a selection for tooltip highlighting
nearest = alt.selection_point(
    nearest=True, on="mouseover", fields=["date"], empty=False
)

# Create line chart for community weighted sentiment by subreddit
line_chart = alt.Chart(df).mark_line().encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format=time_format)),
    y=alt.Y("community_weighted_sentiment:Q", title="Community Weighted Sentiment"),
    color=alt.Color(
        "subreddit:N", 
        scale=alt.Scale(domain=list(subreddits), range=list(subreddit_colors.values())),
        legend=alt.Legend(title="Subreddit")
    ),
    tooltip=["date", "subreddit", "community_weighted_sentiment"]
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
        st.metric("Community Weighted", f"{row['community_weighted_sentiment']:.2f}")
        st.metric("Posts", int(row["count"]))

# ── Keyword Analysis Form at the bottom of the page ────────────────────────
st.header("Extract Keywords from Reddit Posts")

# Create a form for user input
with st.form("keyword_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Subreddit selection dropdown
        selected_subreddit = st.selectbox(
            "Select Subreddit", 
            options=subreddits
        )
    
    with col2:
        # Date selection with calendar widget
        selected_date = st.date_input(
            "Select Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Number of keywords to extract
    num_keywords = st.slider(
        "Number of Keywords to Extract", 
        min_value=3, 
        max_value=15, 
        value=8
    )
    
    # Similarity threshold
    sim_threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.3,
        step=0.05,
        help="Minimum cosine similarity threshold for matching posts to keywords"
    )
    
    # Submit button
    submit_button = st.form_submit_button("Analyze Keywords")

# Process the form submission
if submit_button:
    # Convert date to string format
    date_str = selected_date.strftime("%Y-%m-%d")
    
    # Display loading spinner while fetching data
    with st.spinner(f"Loading data for r/{selected_subreddit} on {date_str}..."):
        # Load the data for the selected day and subreddit
        posts_df = load_day(date_str, selected_subreddit)
        
        if posts_df.empty:
            st.error(f"No posts found for r/{selected_subreddit} on {date_str}")
        else:
            # Display basic stats
            st.subheader(f"r/{selected_subreddit} on {date_str}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Posts", len(posts_df))
            col2.metric("Avg Sentiment", f"{posts_df['sentiment'].mean():.2f}")
            col3.metric("Total Score", f"{posts_df['score'].sum():,}")
            
            # Extract keywords
            with st.spinner("Extracting keywords..."):
                keywords = keywords_for_df(posts_df, top_n=num_keywords)
                
                if not keywords:
                    st.warning("Could not extract meaningful keywords from this content.")
                else:
                    # Calculate keyword stats
                    kw_stats, kw_subsets = keyword_stats(
                        posts_df, 
                        keywords, 
                        sim_thresh=sim_threshold
                    )
                    
                    if kw_stats.empty:
                        st.warning("No keywords met the similarity threshold.")
                    else:
                        # Display keyword stats as a table
                        st.subheader("Keyword Statistics")
                        
                        # Format the dataframe for display
                        display_df = kw_stats.copy()
                        display_df["mean_sentiment"] = display_df["mean_sentiment"].map("{:.2f}".format)
                        display_df["score_weighted_sentiment"] = display_df["score_weighted_sentiment"].map("{:.2f}".format)
                        
                        # Rename columns for better display
                        display_df = display_df.rename(columns={
                            "keyword": "Keyword",
                            "mean_sentiment": "Avg Sentiment",
                            "score_weighted_sentiment": "Weighted Sentiment",
                            "n_posts": "# Posts",
                            "total_score": "Total Score"
                        })
                        
                        # Display the table
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Display expandable sections for each keyword
                        st.subheader("Keyword Details")
                        for kw in kw_stats["keyword"]:
                            subset = kw_subsets[kw]
                            with st.expander(f"{kw} ({len(subset)} posts)"):
                                # Show top posts for this keyword
                                top_posts = subset.sort_values("score", ascending=False).head(5)
                                for _, post in top_posts.iterrows():
                                    # Extract first line as pseudo-title (or first 50 chars)
                                    text = post['text']
                                    first_line = text.split('\n')[0][:50] if '\n' in text else text[:50]
                                    
                                    # Display post details
                                    st.markdown(f"**{first_line}...** (Score: {post['score']:,}, Sentiment: {post['sentiment']:.2f})")
                                    st.markdown(f"*{text[:300]}{'...' if len(text) > 300 else ''}*")
                                    st.markdown("---")

# Display the data source attribution
st.markdown(last_update_caption, unsafe_allow_html=True)