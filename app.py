"""Entry point for the Hugging Face Spaces application.

This tiny wrapper simply imports the Streamlit app defined in
`frontend/app.py`. Importing that module is enough to launch the UI
because the Streamlit code executes at import time.
"""

# Importing `frontend.app` is sufficient to start the Streamlit app.
# The variable name is unused, but keeping the assignment suppresses
# linters complaining about unused imports.
import frontend.app  # noqa: F401  # pragma: no cover 