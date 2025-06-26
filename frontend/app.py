import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yaml
from pathlib import Path

# Call page config BEFORE importing modules that use Streamlit commands
st.set_page_config(page_title="Reddit Sentiment Trends", layout="wide")

# Import from local modules AFTER page config is set
from data_utils import (
    load_summary,
    load_day,
    get_subreddit_colors,
    get_last_updated_hf_caption,
)
from text_analysis import keywords_for_df


st.title("Reddit Sentiment Monitor")

st.markdown(
    """
    **Welcome!** This page shows how Reddit's AI communities feel day-to-day.

    A daily pipeline grabs new posts and comments, scores their tone with a sentiment model, and saves the results to a public HuggingFace [dataset](https://huggingface.co/datasets/hblim/top_reddit_posts_daily). \n

    """
)


# ── Load & transform data ────────────────────────────────────────────────────
df = load_summary()
last_update_caption = get_last_updated_hf_caption()

# Get colors for each subreddit
subreddits = df["subreddit"].unique()
subreddit_colors = get_subreddit_colors(subreddits)

# Load mean/std parameters for sentiment spike bands per subreddit
params_path = Path(__file__).resolve().parent.parent / "spike_params.yaml"
try:
    with params_path.open("r") as f:
        spike_params = yaml.safe_load(f)
except FileNotFoundError:
    spike_params = {}

# Define time format to use across all charts
time_format = "%m/%d/%Y"

# Get date range from the dataset for the form
min_date = df["date"].min().date()
max_date = df["date"].max().date()

# ── Community weighted sentiment line chart for all subreddits ───────────────
st.subheader("Daily Community-Weighted Sentiment")
st.markdown(
    """
    The line chart below plots the daily *community-weighted sentiment*, reflecting the average sentiment across all posts/comments in a subreddit community.
    
    To calculate the community-weighted sentiment:
    - First, each post or comment is assigned a sentiment score of −1 (negative) or +1 (positive) 
    - Then, the sentiment score is weighted by its upvotes so busier discussions matter more. 
    """
)
# Add date range selector for the time series
date_range = st.date_input(
    "Select date range for time series",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
start_date, end_date = date_range
filtered_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)]

# Add a dropdown (selectbox) for choosing a single subreddit to display
default_sub = "artificial" if "artificial" in subreddits else list(subreddits)[0]
selected_subreddit = st.selectbox(
    "Select subreddit",
    options=list(subreddits),
    index=list(subreddits).index(default_sub)
)
plot_df = filtered_df[filtered_df["subreddit"] == selected_subreddit]

# ── Determine shading band and dynamic y-axis domain ────────────────────────
mean_val = std_val = None
if selected_subreddit in spike_params:
    mean_val = spike_params[selected_subreddit].get("mean")
    std_val = spike_params[selected_subreddit].get("std")

# Calculate band limits if parameters exist
band_low = band_high = None
if mean_val is not None and std_val is not None:
    band_low = mean_val - 3 * std_val
    band_high = mean_val + 3 * std_val

# Determine y-axis domain based on data and (optional) band
sent_min = plot_df["community_weighted_sentiment"].min()
sent_max = plot_df["community_weighted_sentiment"].max()

if band_low is not None:
    y_min = float(min(sent_min, band_low))
    y_max = float(max(sent_max, band_high))
else:
    y_min = float(sent_min)
    y_max = float(sent_max)

# Add small padding so points are not flush with edges
padding = 0.05
y_domain = [y_min - padding, y_max + padding]

# Define hover selection for nearest point
nearest = alt.selection_single(
    name="nearest",
    on="mouseover",
    nearest=True,
    fields=["date"],
    empty="none"
)

# Base chart with refreshed y-axis range
base = alt.Chart(plot_df).encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format=time_format, labelPadding=15)),
    y=alt.Y(
        "community_weighted_sentiment:Q",
        title="Community Weighted Sentiment",
        scale=alt.Scale(domain=y_domain),
    ),
)
# Use a constant blue colour for all plot elements
line_colour = "#1f77b4"

# Draw line for the selected subreddit
line = (
    base.transform_calculate(legend='"daily community sentiment score"')
    .mark_line(color=line_colour)
    .encode(
        color=alt.Color(
            "legend:N",
            scale=alt.Scale(domain=["daily community sentiment score", "historical 3σ sentiment range", "significant sentiment outlier"], range=[line_colour, line_colour, "red"]),
            legend=None  # hide default legend; we will add a custom manual legend below the chart
        )
    )
)

# Invisible selectors to capture hover events
selectors = base.mark_point(opacity=0).add_selection(nearest)

# Draw highlighted points on hover
points_hover = base.mark_point(size=60, color=line_colour).encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Tooltip rule and popup
tooltips = base.mark_rule(color="gray").encode(
    tooltip=[
        alt.Tooltip("subreddit:N", title="Subreddit"),
        alt.Tooltip("date:T", title="Date", format=time_format),
        alt.Tooltip("community_weighted_sentiment:Q", title="Sentiment", format=".2f")
    ]
).transform_filter(nearest)

# Optional shaded band (mean ± 3σ)
band = None
outliers = None
domain_labels = [
    "daily community sentiment score",
    "historical 3σ sentiment range",
    "significant sentiment outlier",
]
domain_colors = [line_colour, line_colour, "red"]



if band_low is not None:
    band_df = pd.DataFrame({
        "date": [plot_df["date"].min(), plot_df["date"].max()],
        "low": [band_low, band_low],
        "high": [band_high, band_high],
    })
    band = (
        alt.Chart(band_df)
        .transform_calculate(legend='"historical 3σ sentiment range"')
        .mark_area(opacity=0.15)
        .encode(
            x="date:T",
            y=alt.Y("low:Q", scale=alt.Scale(domain=y_domain)),
            y2="high:Q",
            color=alt.Color(
                "legend:N",
                scale=alt.Scale(domain=domain_labels, range=domain_colors),
                legend=None  # suppress built-in legend for band
            ),
        )
    )

    # Identify significant outliers outside the band
    outlier_df = plot_df[(plot_df["community_weighted_sentiment"] < band_low) |
                         (plot_df["community_weighted_sentiment"] > band_high)].copy()
    if not outlier_df.empty:
        outliers = (
            alt.Chart(outlier_df)
            .transform_calculate(legend='"significant sentiment outlier"')
            .mark_point(shape="circle", size=100, fill="white", stroke="red", strokeWidth=2)
            .encode(
                x="date:T",
                y="community_weighted_sentiment:Q",
                color=alt.Color(
                    "legend:N",
                    scale=alt.Scale(domain=domain_labels, range=domain_colors),
                    legend=None  # suppress built-in legend for outlier
                ),
            )
        )

# Layer everything and make interactive, with title showing subreddit
layers = [line, selectors, points_hover, tooltips]
if band is not None:
    layers.insert(0, band)  # draw band behind the line
if outliers is not None:
    layers.append(outliers)

hover_chart = alt.layer(*layers).properties(
    height=400,  # increased height for more spacious plot area
).interactive(bind_y=False)

# ── Manual legend (two rows) ───────────────────────────────────────────────
legend_df = pd.DataFrame({
    "row": [0, 1],
    "label": ["significant sentiment outlier", "historical 3σ sentiment range"],
    "stroke": ["red", "lightblue"],  # outline colour
    "fill": ["white", "lightblue"],  # interior fill (blue only for historical band)
    "shape": ["circle", "square"],
})

legend_points = (
    alt.Chart(legend_df)
    .mark_point(size=100, filled=True)
    .encode(
        y=alt.Y("row:O", axis=None),
        x=alt.value(0),
        shape=alt.Shape("shape:N", legend=None),
        stroke=alt.Stroke("stroke:N", scale=None, legend=None),
        fill=alt.Fill("fill:N", scale=None, legend=None),
    )
)

legend_text = (
    alt.Chart(legend_df)
    .mark_text(align="left", baseline="middle", dx=15)
    .encode(
        y="row:O",
        x=alt.value(0),
        text="label:N",
    )
)

manual_legend = (
    legend_points + legend_text
).properties(height=60)

# Concatenate chart and manual legend vertically
final_chart = alt.vconcat(
    manual_legend,    
    hover_chart,
    spacing=0
).configure_view(strokeWidth=0)

st.altair_chart(final_chart, use_container_width=True)

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
).properties(height=400).interactive()

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

# ── Analyze sentiment driving posts ─────────────────────────────────────
st.header("Analyze sentiment driving posts")
with st.form("analysis_form"):
    col1, col2 = st.columns(2)
    with col1:
        selected_subreddit = st.selectbox("Select Subreddit", options=subreddits)
    with col2:
        selected_date = st.date_input(
            "Select Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    submit_button = st.form_submit_button("Analyze Posts")

if submit_button:
    date_str = selected_date.strftime("%Y-%m-%d")
    with st.spinner(f"Loading data for r/{selected_subreddit} on {date_str}..."):
        posts_df = load_day(date_str, selected_subreddit)

    if posts_df.empty:
        st.error(f"No posts found for r/{selected_subreddit} on {date_str}")
    else:
        # Separate posts and comments
        posts = posts_df[posts_df["type"] == "post"]
        comments = posts_df[posts_df["type"] == "comment"]

        # Overall summary metrics using engagement-adjusted sentiment (EAS)
        n_posts = len(posts)
        df_day = posts_df.copy()
        df_day["score_num"] = pd.to_numeric(df_day["score"], errors="coerce").fillna(0)
        weights_base_day = 1 + np.log1p(df_day["score_num"].clip(lower=0))
        gamma_post = 0.3
        weights_day = weights_base_day * np.where(df_day["type"] == "post", gamma_post, 1.0)
        total_weight_day = weights_day.sum()
        overall_eas = (weights_day * df_day["sentiment"]).sum() / weights_day.sum() if weights_day.sum() > 0 else 0
        # Normalize daily weighted sentiment to range [-1,1]
        overall_eas = 2 * overall_eas - 1
        overall_score = df_day["score"].sum()

        st.subheader(f"r/{selected_subreddit} on {date_str}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Posts", n_posts)
        c2.metric("Daily Weighted Sentiment, All Posts", f"{overall_eas:.2f}")
        c3.metric("Total Score, All Posts", f"{overall_score:,}")

        # Wrap analysis and rendering of top posts in a spinner
        with st.spinner("Analyzing sentiment and rendering top posts..."):
            # Build per-post analysis
            analysis_rows = []
            for _, post in posts.iterrows():
                pid = post["post_id"]
                text = post["text"]
                # Gather comments for this post
                post_comments = comments[comments["parent_id"] == f"t3_{pid}"]

                # Combine post and comments for calculations
                segment = pd.concat([pd.DataFrame([post]), post_comments], ignore_index=True)
                # Compute engagement-adjusted sentiment for this post thread
                segment_score_num = pd.to_numeric(segment["score"], errors="coerce").fillna(0)
                weights_base = 1 + np.log1p(segment_score_num.clip(lower=0))
                gamma_post = 0.3
                weights_seg = weights_base * np.where(segment["type"] == "post", gamma_post, 1.0)
                ws = (weights_seg * segment["sentiment"]).sum() / weights_seg.sum() if weights_seg.sum() > 0 else 0
                # Normalize weighted sentiment of thread to range [-1,1]
                ws = 2 * ws - 1
                ts = segment["score"].sum()
                nc = len(post_comments)

                thread_weight_sum = weights_seg.sum()
                contrib_weight = thread_weight_sum / total_weight_day if total_weight_day > 0 else 0
                total_contribution = contrib_weight * ws

                analysis_rows.append({
                    "post_id": pid,
                    "Post Keywords": "",  # placeholder; will compute for top posts only
                    "Weighted Sentiment of Thread": ws,
                    "Contribution Weight": contrib_weight,
                    "Total Sentiment Contribution": total_contribution,
                    "# Comments": nc,
                    "Total Score": ts
                })

            analysis_df = pd.DataFrame(analysis_rows)
            # Determine top 5 posts by contribution weight
            top5 = analysis_df.sort_values("Contribution Weight", ascending=False).head(5).copy()
            top5.reset_index(drop=True, inplace=True)

            # Compute keywords only for top posts
            for idx, row in top5.iterrows():
                pid = row["post_id"]
                post_text = posts[posts["post_id"] == pid].iloc[0]["text"]
                kw = keywords_for_df(pd.DataFrame({"text": [post_text]}), top_n=2)
                keywords_list = [k for k, _ in kw][:2]
                top5.at[idx, "Post Keywords"] = ", ".join(keywords_list)

            # Format numeric columns
            for df_part in (top5,):
                df_part["Weighted Sentiment of Thread"] = df_part["Weighted Sentiment of Thread"].map("{:.2f}".format)
                df_part["Total Score"] = df_part["Total Score"].map("{:,}".format)
                df_part["Contribution Weight"] = df_part["Contribution Weight"].map("{:.2%}".format)
                df_part["Total Sentiment Contribution"] = df_part["Total Sentiment Contribution"].map("{:.4f}".format)

            st.subheader("Top 5 Posts by Contribution Weight")
            st.dataframe(
                top5[["Post Keywords", "Weighted Sentiment of Thread", "Contribution Weight", "Total Sentiment Contribution", "# Comments", "Total Score"]],
                use_container_width=True
            )

            st.subheader("Post Details")
            for idx, row in top5.reset_index(drop=True).iterrows():
                pid = row["post_id"]
                post_obj = posts[posts["post_id"] == pid].iloc[0]
                post_text = post_obj["text"]
                first_line = post_text.split("\n")[0][:50] 
                with st.expander(f"{idx} - {first_line}..."):
                    # Post Metrics
                    post_sent = post_obj["sentiment"]
                    # Normalize post sentiment to [-1,1]
                    post_sent_norm = 2 * post_sent - 1
                    post_score = post_obj["score"]
                    ps = pd.to_numeric(post_score, errors="coerce")
                    post_score_num = ps if (not np.isnan(ps) and ps >= 0) else 0
                    # Compute post weight
                    post_weight = (1 + np.log1p(post_score_num)) * gamma_post
                    st.markdown("**Post:**")
                    st.markdown(f"{post_text[:300]}{'...' if len(post_text) > 300 else ''}"
                                f"(Sentiment: {post_sent_norm:.2f}, Weight: {post_weight:.2f}, Score: {post_score:,})"
                                )
                    st.markdown("---")
                    # Display top 5 comments with metrics
                    top_comments = (
                        comments[comments["parent_id"] == f"t3_{pid}"]
                        .sort_values("score", ascending=False)
                        .head(5)
                    )
                    st.markdown("**Top Comments:**")
                    for c_idx, comment in top_comments.iterrows():
                        c_text = comment["text"]
                        # Normalize comment sentiment and compute weight
                        c_sent_norm = 2 * comment["sentiment"] - 1
                        c_score = comment["score"]
                        cs = pd.to_numeric(c_score, errors="coerce")
                        c_score_num = cs if (not np.isnan(cs) and cs >= 0) else 0
                        c_weight = 1 + np.log1p(c_score_num)
                        st.markdown(
                            f"{c_idx}. {c_text[:200]}{'...' if len(c_text) > 200 else ''} "
                            f"(Sentiment: {c_sent_norm:.2f}, Weight: {c_weight:.2f}, Score: {c_score:,})"
                        )

# Display the data source attribution
# st.markdown(last_update_caption, unsafe_allow_html=True)