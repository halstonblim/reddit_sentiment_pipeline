import os
from pathlib import Path
import pytest
import yaml
from reddit_analysis.config_utils import load_config, get_secret, ROOT

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file with test data."""
    config = {
        'repo_id': 'test/repo',
        'repo_type': 'dataset',
        'raw_dir': 'data/raw',
        'scored_dir': 'data/scored',
        'logs_dir': 'logs',
        'summary_file': 'summary.csv',
        'hf_raw_dir': 'data/raw',
        'hf_scored_dir': 'data/scored',
        'batch_size': 16,
        'replicate_model': 'test/model',
        'subreddits': ['test1', 'test2'],
        'post_limit': 100
    }
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def test_load_config(temp_config_file, monkeypatch):
    """Test that load_config correctly reads the config file."""
    # Mock the ROOT path to point to our test directory
    monkeypatch.setattr('reddit_analysis.config_utils.ROOT', temp_config_file.parent)
    
    # Load the config
    config = load_config()  # Should now find config.yaml in the test directory
    
    # Verify the values
    assert config['repo_id'] == 'test/repo'
    assert config['repo_type'] == 'dataset'
    assert config['raw_dir'] == 'data/raw'
    assert config['scored_dir'] == 'data/scored'
    assert config['logs_dir'] == 'logs'
    assert config['summary_file'] == 'summary.csv'
    assert config['hf_raw_dir'] == 'data/raw'
    assert config['hf_scored_dir'] == 'data/scored'
    assert config['batch_size'] == 16
    assert config['replicate_model'] == 'test/model'
    assert config['subreddits'] == ['test1', 'test2']
    assert config['post_limit'] == 100

def test_get_secret_env_var(monkeypatch):
    """Test get_secret with environment variable."""
    # Set a test environment variable
    monkeypatch.setenv('TEST_SECRET', 'env_value')
    
    # Get the secret
    value = get_secret('TEST_SECRET')
    
    # Verify it returns the environment variable value
    assert value == 'env_value'

def test_get_secret_streamlit(monkeypatch):
    """Test get_secret with Streamlit secrets."""
    # Remove environment variable
    monkeypatch.delenv('TEST_SECRET', raising=False)
    
    # Mock Streamlit's HAS_STREAMLIT to True
    monkeypatch.setattr('reddit_analysis.config_utils.HAS_STREAMLIT', True)
    # Mock is_running_streamlit to True
    monkeypatch.setattr('reddit_analysis.config_utils.is_running_streamlit', lambda: True)
    # Mock Streamlit secrets
    class MockSecrets:
        def get(self, key, default=None):
            return 'streamlit_value'
    monkeypatch.setattr('streamlit.secrets', MockSecrets())
    # Get the secret
    value = get_secret('TEST_SECRET')
    # Verify it returns the Streamlit secret value
    assert value == 'streamlit_value'

def test_get_secret_missing(monkeypatch):
    """Test get_secret when secret is missing from both sources."""
    # Remove environment variable
    monkeypatch.delenv('TEST_SECRET', raising=False)
    
    # Mock Streamlit's HAS_STREAMLIT to True
    monkeypatch.setattr('reddit_analysis.config_utils.HAS_STREAMLIT', True)
    
    # Mock Streamlit secrets to return None
    class MockSecrets:
        def get(self, key, default=None):
            return default
    
    monkeypatch.setattr('streamlit.secrets', MockSecrets())
    
    # Verify it raises ValueError
    with pytest.raises(ValueError) as exc_info:
        get_secret('TEST_SECRET')
    assert "Required secret TEST_SECRET not found" in str(exc_info.value) 