"""
Text analysis utilities for Reddit content insights.
Provides keyword extraction and similarity matching functions.
"""
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity

# Initialize spaCy and sentence transformer models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import streamlit as st
    with st.spinner("Downloading NLP model (first run only)..."):
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

# Cache models at module scope for reuse
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(embedder)

def keywords_for_df(df: pd.DataFrame, top_n=5):
    """
    Extract keywords from a DataFrame containing Reddit posts.
    
    Args:
        df: DataFrame with a 'text' column containing post content
        top_n: Number of top keywords to return
        
    Returns:
        List of (keyword, score) tuples
    """
    if df.empty:
        return []
    
    # Join all text from the dataframe
    raw = " ".join(df["text"].astype(str))
    
    # Process with spaCy to extract noun chunks and named entities
    doc = nlp(raw.lower())
    
    # Combine noun chunks and relevant named entities
    cand = " ".join(
        [c.text for c in doc.noun_chunks] +
        [e.text for e in doc.ents if e.label_ in {"PRODUCT", "EVENT", "ORG", "GPE"}]
    )
    
    # Quick stopword list to filter common terms
    for ex in ['google','pixel','android','iphone','apple','rationale','advice','blog','topic','locked','author','moderator','error','bot','comments','archive','support','discord']:
        cand = cand.replace(ex, " ")
    
    # Use KeyBERT to extract keywords with diversity
    return kw_model.extract_keywords(
        cand,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.8,
        top_n=top_n
    )


def keyword_stats(df_slice: pd.DataFrame,
                  keywords: list[tuple[str, float]],
                  sim_thresh: float = 0.30):
    """
    Return a DataFrame with stats for each keyword that actually hits
    >= 1 post at cosine-similarity >= sim_thresh.
    
    Args:
        df_slice: DataFrame containing Reddit posts
        keywords: List of (keyword, score) tuples from keywords_for_df
        sim_thresh: Minimum cosine similarity threshold
        
    Returns:
        Tuple of (stats_df, subsets_dict) where:
            - stats_df: DataFrame with keyword statistics
            - subsets_dict: Dictionary mapping keywords to matching post DataFrames
    """
    if df_slice.empty or not keywords:
        return pd.DataFrame(), {}
    
    # Encode all post texts
    texts = df_slice['text'].tolist()
    text_embs = embedder.encode(texts, convert_to_tensor=False)  # (n_posts, 384)

    rows = []
    subsets = {}

    for kw, _ in keywords:
        # Encode keyword and calculate similarity
        kw_emb = embedder.encode(kw, convert_to_tensor=False).reshape(1, -1)
        sims = cosine_similarity(text_embs, kw_emb).ravel()

        # Filter to posts above similarity threshold
        mask = sims >= sim_thresh
        subset = df_slice.loc[mask]

        if subset.empty:  # skip keywords with no close matches
            continue

        subsets[kw] = subset  # keep for expanders

        # Calculate statistics
        mean_sent = subset['sentiment'].mean()
        weighted = ((2*subset['sentiment']-1) *
                   np.log1p(subset['score'].clip(0))).mean()
        total_up = subset['score'].sum()

        rows.append((kw, mean_sent, weighted, len(subset), total_up))

    if not rows:
        return pd.DataFrame(), {}
        
    # Create and sort stats DataFrame
    cols = ['keyword', 'mean_sentiment',
            'score_weighted_sentiment', 'n_posts', 'total_score']
    return pd.DataFrame(rows, columns=cols).sort_values(
        'total_score', ascending=False
    ), subsets
