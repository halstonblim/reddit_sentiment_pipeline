# Reddit Sentiment Analysis Pipeline

This project analyzes sentiment from Reddit posts across multiple subreddits, processes the data, and visualizes the trends using Streamlit.

## Environment Variable Handling

The application is designed to work in both local development environments and when deployed to Streamlit Cloud:

### Setting up Environment Variables

1. **Local Development**:
   - Create a `.env` file in the project root with your secrets:
   ```
   ENV=local
   HF_TOKEN=your_huggingface_token
   REPLICATE_API_TOKEN=your_replicate_token
   ```
   - When the application detects `ENV=local` or if Streamlit is not available, it will use python-dotenv to load these variables

2. **Streamlit Cloud Deployment**:
   - Configure secrets in the Streamlit Cloud dashboard
   - The application will automatically use `st.secrets` when running in Streamlit Cloud

### Secret Access Pattern

The application uses a consistent pattern for accessing secrets across all components:

```python
# First check for environment variables, then fall back to Streamlit secrets
def get_secret(key, default=None):
    """Get a secret from environment variables or Streamlit secrets."""
    value = os.getenv(key)
    if value is None and has_streamlit:
        value = st.secrets.get(key, default)
    return value

# Usage
API_KEY = get_secret("API_KEY")
```

This ensures that your application works seamlessly in both local development and production environments.
