"""
Text analysis utilities for Reddit content insights.
Provides keyword extraction and similarity matching functions.
"""
import pandas as pd

# NOTE:
# Heavy NLP/ML libraries (spaCy, sentence-transformers, KeyBERT, torch, etc.) can take a
# long time to import or may not be available in constrained environments (e.g. the
# default HuggingFace Spaces CPU image).  Importing them at module import time can cause
# the module to fail to initialise which, in turn, leads to cryptic errors such as
# "cannot import name 'keywords_for_df'".  To avoid this we lazily import the heavy
# dependencies the first time they are actually needed.  The helper is cached so that
# subsequent calls are fast.

from functools import lru_cache


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_models():
    """Lazily load and cache NLP models.

    Returns
    -------
    tuple
        (nlp, kw_model) where ``nlp`` is a spaCy language model and ``kw_model`` is a
        KeyBERT instance.  If the required libraries are not available the function
        raises ImportError *inside* the helper so the caller can decide how to handle
        the failure gracefully.
    """

    import importlib

    # Import spaCy and ensure the small English model is available
    spacy = importlib.import_module("spacy")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Streamlit is only available after `st.set_page_config` is called, which the
        # main app does before importing this module.  We therefore import it lazily
        # here to avoid a hard dependency when the module is imported outside a
        # Streamlit context (e.g. unit tests).
        try:
            import streamlit as st  # noqa: WPS433 (allow late import)

            with st.spinner("Downloading spaCy model (first run only)..."):
                from spacy.cli import download  # noqa: WPS433 (late import)

                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
        except ModuleNotFoundError:
            # If Streamlit isn't available, fall back to downloading silently.
            from spacy.cli import download  # noqa: WPS433 (late import)

            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

    # Sentence-Transformers and KeyBERT (which depends on it)
    sent_trans = importlib.import_module("sentence_transformers")
    SentenceTransformer = sent_trans.SentenceTransformer

    KeyBERT = importlib.import_module("keybert").KeyBERT

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(embedder)

    return nlp, kw_model

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

    # Attempt to load heavy models.  If this fails we degrade gracefully by returning
    # an empty list rather than crashing the whole application.
    try:
        nlp, kw_model = _load_models()
    except Exception as exc:  # noqa: BLE001 (broad, but we degrade gracefully)
        # Log the failure inside Streamlit if available; otherwise swallow silently.
        try:
            import streamlit as st  # noqa: WPS433

            st.warning(
                f"Keyword extraction disabled due to model loading error: {exc}",
                icon="⚠️",
            )
        except ModuleNotFoundError:
            pass

        return []

    # Join all text from the dataframe
    raw = " ".join(df["text"].astype(str))

    # Process with spaCy to extract noun chunks and named entities
    doc = nlp(raw.lower())

    # Combine noun chunks and relevant named entities
    cand = " ".join(
        [c.text for c in doc.noun_chunks]
        + [e.text for e in doc.ents if e.label_ in {"PRODUCT", "EVENT", "ORG", "GPE"}]
    )

    # Quick stopword list to filter common terms
    for ex in [
        "blog",
        "topic",
        "locked",
        "author",
        "moderator",
        "error",
        "bot",
        "comments",
        "archive",
        "support",
        "discord",
    ]:
        cand = cand.replace(ex, " ")

    # Use KeyBERT to extract keywords with diversity
    return kw_model.extract_keywords(
        cand,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.8,
        top_n=top_n,
    )
