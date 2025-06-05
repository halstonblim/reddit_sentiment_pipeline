import os
from pathlib import Path
import pytest
import pandas as pd
from datetime import datetime
import pytz
from unittest.mock import Mock, patch
import json

from reddit_analysis.inference.score import SentimentScorer, ReplicateAPI, FileManager, HuggingFaceManager

@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        'config': {
            'repo_id': 'test/repo',
            'repo_type': 'dataset',
            'batch_size': 2,
            'replicate_model': 'test/model'
        },
        'paths': {
            'raw_dir': Path('data/raw'),
            'scored_dir': Path('data/scored'),
            'hf_raw_dir': 'data/raw',
            'hf_scored_dir': 'data/scored'
        },
        'secrets': {
            'HF_TOKEN': 'test_token',
            'REPLICATE_API_TOKEN': 'test_token'
        }
    }

@pytest.fixture
def mock_replicate_api():
    """Create a mock ReplicateAPI."""
    mock = Mock(spec=ReplicateAPI)
    mock.predict.return_value = {
        'predicted_labels': ['positive', 'negative'],
        'confidences': [0.9, 0.8]
    }
    return mock

@pytest.fixture
def mock_file_manager():
    """Create a mock FileManager."""
    mock = Mock(spec=FileManager)
    return mock

@pytest.fixture
def mock_hf_manager():
    """Create a mock HuggingFaceManager."""
    mock = Mock(spec=HuggingFaceManager)
    return mock

def test_score_date(mock_config, mock_replicate_api, mock_file_manager, mock_hf_manager):
    """Test the score_date method."""
    # Create test input DataFrame
    input_df = pd.DataFrame({
        'text': ['Test text 1', 'Test text 2'],
        'score': [1, 2],
        'post_id': ['post1', 'post2'],
        'subreddit': ['test1', 'test1']
    })
    
    # Mock file operations
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    mock_file_manager.read_parquet.return_value = input_df
    mock_hf_manager.list_files.return_value = []  # No existing files
    
    # Initialize scorer with mocked dependencies
    scorer = SentimentScorer(
        mock_config,
        replicate_api=mock_replicate_api,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Score the data
    scorer.score_date('2025-04-20')
    
    # Verify API calls
    mock_replicate_api.predict.assert_called_once()
    mock_file_manager.save_parquet.assert_called_once()
    mock_hf_manager.upload_file.assert_called_once()

def test_score_date_missing_columns(mock_config, mock_replicate_api, mock_file_manager, mock_hf_manager):
    """Test score_date with missing required columns."""
    # Create test input DataFrame missing required columns
    input_df = pd.DataFrame({
        'text': ['Test text 1', 'Test text 2'],
        'score': [1, 2]
    })
    
    # Mock file operations
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    mock_file_manager.read_parquet.return_value = input_df
    mock_hf_manager.list_files.return_value = []  # No existing files
    
    # Initialize scorer with mocked dependencies
    scorer = SentimentScorer(
        mock_config,
        replicate_api=mock_replicate_api,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Verify it raises ValueError
    with pytest.raises(ValueError) as exc_info:
        scorer.score_date('2025-04-20')
    assert "Missing required columns" in str(exc_info.value)

def test_score_date_batch_processing(mock_config, mock_replicate_api, mock_file_manager, mock_hf_manager):
    """Test that score_date correctly processes data in batches."""
    # Create test input DataFrame with more rows than batch size
    input_df = pd.DataFrame({
        'text': [f'Test text {i}' for i in range(5)],
        'score': [i + 1 for i in range(5)],
        'post_id': [f'post{i}' for i in range(5)],
        'subreddit': ['test1'] * 5
    })
    
    # Mock file operations
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    mock_file_manager.read_parquet.return_value = input_df
    mock_hf_manager.list_files.return_value = []  # No existing files
    
    # Initialize scorer with mocked dependencies
    scorer = SentimentScorer(
        mock_config,
        replicate_api=mock_replicate_api,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Score the data
    scorer.score_date('2025-04-20')
    
    # Verify that replicate_api.predict was called the correct number of times
    assert mock_replicate_api.predict.call_count == 3  # 5 rows with batch_size=2
    
    # Verify file operations
    mock_file_manager.save_parquet.assert_called_once()
    mock_hf_manager.upload_file.assert_called_once()

def test_score_date_multiple_subreddits(mock_config, mock_replicate_api, mock_file_manager, mock_hf_manager):
    """Test that score_date correctly handles multiple subreddits."""
    # Create test input DataFrame with multiple subreddits
    input_df = pd.DataFrame({
        'text': ['Test text 1', 'Test text 2', 'Test text 3', 'Test text 4'],
        'score': [1, 2, 3, 4],
        'post_id': ['post1', 'post2', 'post3', 'post4'],
        'subreddit': ['test1', 'test1', 'test2', 'test2']
    })
    
    # Mock file operations
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    mock_file_manager.read_parquet.return_value = input_df
    mock_hf_manager.list_files.return_value = []  # No existing files
    
    # Initialize scorer with mocked dependencies
    scorer = SentimentScorer(
        mock_config,
        replicate_api=mock_replicate_api,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Score the data
    scorer.score_date('2025-04-20')
    
    # Verify API calls
    mock_replicate_api.predict.assert_called()
    
    # Verify that save_parquet was called for each subreddit
    assert mock_file_manager.save_parquet.call_count == 2  # 2 subreddits
    
    # Verify that upload_file was called for each subreddit
    assert mock_hf_manager.upload_file.call_count == 2  # 2 subreddits
    
    # Check that the upload paths are correct
    upload_calls = mock_hf_manager.upload_file.call_args_list
    upload_paths = [call[0][1] for call in upload_calls]  # Second positional argument is path_in_repo
    assert 'data_scored_subreddit/2025-04-20__test1.parquet' in upload_paths
    assert 'data_scored_subreddit/2025-04-20__test2.parquet' in upload_paths

def test_score_date_with_existing_subreddits(mock_config, mock_replicate_api, mock_file_manager, mock_hf_manager):
    """Test that score_date skips existing subreddits when overwrite=False."""
    # Create test input DataFrame with multiple subreddits
    input_df = pd.DataFrame({
        'text': ['Test text 1', 'Test text 2', 'Test text 3', 'Test text 4'],
        'score': [1, 2, 3, 4],
        'post_id': ['post1', 'post2', 'post3', 'post4'],
        'subreddit': ['test1', 'test1', 'test2', 'test2']
    })
    
    # Mock file operations
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    mock_file_manager.read_parquet.return_value = input_df
    # Mock existing files - test1 already exists
    mock_hf_manager.list_files.return_value = ['data_scored_subreddit/2025-04-20__test1.parquet']
    
    # Initialize scorer with mocked dependencies
    scorer = SentimentScorer(
        mock_config,
        replicate_api=mock_replicate_api,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Score the data (overwrite=False by default)
    scorer.score_date('2025-04-20', overwrite=False)
    
    # Verify API calls - should only process test2 subreddit (2 texts)
    mock_replicate_api.predict.assert_called()
    
    # Verify that save_parquet was called only for test2
    assert mock_file_manager.save_parquet.call_count == 1
    
    # Verify that upload_file was called only for test2
    assert mock_hf_manager.upload_file.call_count == 1
    
    # Check that only test2 was uploaded
    upload_calls = mock_hf_manager.upload_file.call_args_list
    upload_paths = [call[0][1] for call in upload_calls]  # Second positional argument is path_in_repo
    assert 'data_scored_subreddit/2025-04-20__test2.parquet' in upload_paths
    assert 'data_scored_subreddit/2025-04-20__test1.parquet' not in upload_paths

def test_get_existing_subreddits(mock_config, mock_replicate_api, mock_file_manager, mock_hf_manager):
    """Test the get_existing_subreddits method."""
    # Mock existing files
    mock_hf_manager.list_files.return_value = [
        'data_scored_subreddit/2025-04-20__test1.parquet',
        'data_scored_subreddit/2025-04-20__test2.parquet',
        'data_scored_subreddit/2025-04-21__test1.parquet',  # Different date
        'other_folder/2025-04-20__test3.parquet'  # Different folder
    ]
    
    # Initialize scorer with mocked dependencies
    scorer = SentimentScorer(
        mock_config,
        replicate_api=mock_replicate_api,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Get existing subreddits for 2025-04-20
    existing = scorer.get_existing_subreddits('2025-04-20')
    
    # Should only include test1 and test2 for the correct date
    assert existing == {'test1', 'test2'}

def test_cli_missing_token(monkeypatch, tmp_path):
    """Test CLI with missing REPLICATE_API_TOKEN."""
    # Create a temporary .env file without REPLICATE_API_TOKEN
    env_path = tmp_path / '.env'
    env_path.write_text('')
    
    # Set environment variable to point to our test .env
    monkeypatch.setenv('REDDIT_ANALYSIS_ENV', str(env_path))
    
    # Remove REPLICATE_API_TOKEN from environment
    monkeypatch.delenv('REPLICATE_API_TOKEN', raising=False)
    # Ensure HF_TOKEN is present so only REPLICATE_API_TOKEN is missing
    monkeypatch.setenv('HF_TOKEN', 'dummy_hf_token')
    
    # Mock Streamlit's HAS_STREAMLIT to True
    monkeypatch.setattr('reddit_analysis.config_utils.HAS_STREAMLIT', True)
    # Mock is_running_streamlit to True
    monkeypatch.setattr('reddit_analysis.config_utils.is_running_streamlit', lambda: True)
    # Mock Streamlit secrets
    mock_secrets = Mock()
    mock_secrets.get.return_value = None
    monkeypatch.setattr('streamlit.secrets', mock_secrets)
    # Print for debug
    import os
    print('DEBUG: REPLICATE_API_TOKEN value before main:', os.environ.get('REPLICATE_API_TOKEN'))
    # Run the CLI with --date argument
    with pytest.raises(ValueError) as exc_info:
        from reddit_analysis.inference.score import main
        main('2025-04-20')
    assert "REPLICATE_API_TOKEN is required for scoring" in str(exc_info.value)