"""
Configuration utilities for Reddit analysis tools.
Handles loading of config from YAML and secrets from environment or Streamlit.
"""
import os
from pathlib import Path
import yaml

# Determine if Streamlit is available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Project root - now points to the project root directory
ROOT = Path(__file__).resolve().parent.parent

def is_running_streamlit():
    # The only reliable way to detect if running inside a Streamlit app
    return os.getenv("STREAMLIT_SERVER_PORT") is not None

def load_environment():
    """Load environment variables from .env if not running as a Streamlit app."""
    if not is_running_streamlit():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / '.env')

def get_secret(key, default=None):
    """Get a secret from environment variables or Streamlit secrets."""
    value = os.getenv(key)
    if value is None and HAS_STREAMLIT and is_running_streamlit():
        value = st.secrets.get(key, default)
    if value is None and default is None:
        raise ValueError(f"Required secret {key} not found in environment or Streamlit secrets")
    return value
    
def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = ROOT / "config.yaml"
    else:
        config_path = Path(config_path)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_project_root():
    """Return the project root directory."""
    return ROOT

def setup_config():
    """
    Set up and return configuration and commonly used values.
    
    Returns:
        A dictionary containing configuration and common values:
        - config: The parsed YAML config
        - secrets: A dictionary of required secrets (e.g., HF_TOKEN)
        - paths: Common file paths (all relative to project root)
    """
    # Load environment variables
    load_environment()
    
    # Load config
    config = load_config()
    
    # Common secrets
    secrets = {
        'HF_TOKEN': get_secret('HF_TOKEN')
    }
    
    # Get directory paths from config or use defaults
    raw_dir = config.get('raw_dir', 'data_raw')
    scored_dir = config.get('scored_dir', 'data_scored')
    logs_dir = config.get('logs_dir', 'logs')
    
    # Get HF repository directories (paths within the HF repo)
    hf_raw_dir = config.get('hf_raw_dir', 'data_raw')
    hf_scored_dir = config.get('hf_scored_dir', 'data_scored')
    
    # Common paths and constants (all paths are relative to project root)
    paths = {
        'root': ROOT,
        'raw_dir': ROOT / raw_dir,
        'scored_dir': ROOT / scored_dir,
        'logs_dir': ROOT / logs_dir,
        'summary_file': ROOT / config.get('summary_file', 'subreddit_daily_summary.csv'),
        'hf_raw_dir': hf_raw_dir,
        'hf_scored_dir': hf_scored_dir
    }
    
    # Add REPLICATE_API_TOKEN if it's in the environment
    try:
        secrets['REPLICATE_API_TOKEN'] = get_secret('REPLICATE_API_TOKEN')
    except ValueError:
        # This is optional for scrape.py, so we'll ignore if missing
        pass
    
    # Add Reddit API credentials if available
    for key in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']:
        try:
            secrets[key] = get_secret(key)
        except ValueError:
            # These are required by scrape.py but we'll check there
            pass
    
    return {
        'config': config,
        'secrets': secrets,
        'paths': paths
    }
