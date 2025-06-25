"""
Text analysis utilities for Reddit content insights.
Provides keyword extraction and similarity matching functions.
"""
import pandas as pd
import contextlib

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

    # ------------------------------------------------------------------
    # Inform the user via Streamlit (if available) that heavy models are
    # loading.  We use a spinner that is shown only on the first call; the
    # function is cached so subsequent calls skip the spinner entirely.
    # ------------------------------------------------------------------

    try:
        import streamlit as st  # noqa: WPS433 (late import)

        spinner_cm = st.spinner(
            "Initializing keyword-extraction models (first run may take ~1 min)…",
        )
    except ModuleNotFoundError:
        # If Streamlit isn't present (e.g. unit tests) simply do nothing.
        spinner_cm = contextlib.nullcontext()

    with spinner_cm:
        # Import spaCy and ensure the small English model is available
        spacy = importlib.import_module("spacy")

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            # The model is missing.  Do NOT attempt to install it at run-time
            # because the app may run under a non-privileged user (e.g. Streamlit
            # Cloud) and lack write permissions to the virtual-env.  Instead we
            # instruct the developer to add the model wheel to build-time
            # dependencies so it gets installed by pip when the image is built.
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Add 'en-core-web-sm==3.8.0' (hyphen, not underscore) to "
                "your requirements.txt so it is installed during deployment."
            ) from exc

        # Sentence-Transformers and KeyBERT (which depends on it)
        sent_trans = importlib.import_module("sentence_transformers")
        SentenceTransformer = sent_trans.SentenceTransformer

        KeyBERT = importlib.import_module("keybert").KeyBERT

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        kw_model = KeyBERT(embedder)

    # Notify user that models are ready (only on first load)
    try:
        st.success("Keyword-extraction models ready!", icon="✅")  # type: ignore[name-defined]
    except Exception:  # noqa: BLE001 (streamlit not available or other minor issue)
        pass

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
