"""
Text analysis utilities for Reddit content insights.
Provides keyword extraction and similarity matching functions.
"""
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

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
    for ex in ['blog','topic','locked','author','moderator','error','bot','comments','archive','support','discord']:
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
