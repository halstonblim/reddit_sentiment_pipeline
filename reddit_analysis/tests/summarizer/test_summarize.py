import os
from pathlib import Path
import pytest
import pandas as pd
from datetime import datetime, date
import pytz
from unittest.mock import Mock, patch

from reddit_analysis.summarizer.summarize import SummaryManager, FileManager, HuggingFaceManager

@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        'config': {
            'repo_id': 'test/repo',
            'repo_type': 'dataset'
        },
        'paths': {
            'root': Path('data'),
            'scored_dir': Path('data/scored'),
            'hf_scored_dir': 'data/scored',
            'summary_file': Path('summary.csv')
        },
        'secrets': {
            'HF_TOKEN': 'test_token'
        }
    }

@pytest.fixture
def mock_file_manager():
    """Create a mock FileManager."""
    mock = Mock(spec=FileManager)
    # Return empty DataFrame with expected columns
    mock.read_csv.return_value = pd.DataFrame(columns=["date", "subreddit", "mean_sentiment", "weighted_sentiment", "count"])
    return mock

@pytest.fixture
def mock_hf_manager():
    """Create a mock HuggingFaceManager."""
    mock = Mock(spec=HuggingFaceManager)
    return mock

def test_process_date(mock_config, mock_file_manager, mock_hf_manager):
    """Test the process_date method."""
    # Create sample scored data
    sample_data = pd.DataFrame({
        'subreddit': ['test1', 'test1', 'test2', 'test2'],
        'sentiment': [0.8, 0.6, 0.4, 0.2],
        'score': [10, 20, 30, 40],
        'post_id': ['post1', 'post2', 'post3', 'post4'],
        'text': ['text1', 'text2', 'text3', 'text4'],
        'retrieved_at': [datetime.now(pytz.UTC)] * 4,
        'date': ['2025-04-20'] * 4  # Add date column
    })
    
    # Mock file operations
    mock_file_manager.read_parquet.return_value = sample_data
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    
    # Initialize manager with mocked dependencies
    manager = SummaryManager(mock_config, file_manager=mock_file_manager, hf_manager=mock_hf_manager)
    
    # Process date
    manager.process_date('2025-04-20')
    
    # Verify file operations
    mock_file_manager.read_parquet.assert_called_once()
    mock_file_manager.write_csv.assert_called_once()
    
    # Verify HF operations
    mock_hf_manager.upload_file.assert_called_once()

def test_get_processed_combinations(mock_config, mock_file_manager, mock_hf_manager):
    """Test the get_processed_combinations method."""
    # Create initial summary data
    initial_summary = pd.DataFrame({
        'date': ['2025-04-19', '2025-04-19'],
        'subreddit': ['test1', 'test2'],
        'mean_sentiment': [0.5, 0.3],
        'weighted_sentiment': [0.4, 0.2],
        'count': [1, 1],
        'total_score': [10, 20]
    })
    
    # Mock file operations
    mock_file_manager.read_csv.return_value = initial_summary
    
    # Initialize manager with mocked dependencies
    manager = SummaryManager(mock_config, file_manager=mock_file_manager, hf_manager=mock_hf_manager)
    
    # Get processed combinations
    processed = manager.get_processed_combinations()
    
    # Verify processed combinations
    assert len(processed) == 2
    assert (date(2025, 4, 19), 'test1') in processed
    assert (date(2025, 4, 19), 'test2') in processed

def test_cli_invalid_date():
    """Test CLI with invalid date format."""
    with pytest.raises(ValueError) as exc_info:
        from reddit_analysis.summarizer.summarize import main
        main('2025-04-20-invalid')
    
    assert "Invalid date format" in str(exc_info.value)

def test_cli_missing_file(mock_config, mock_file_manager, mock_hf_manager):
    """Test CLI with missing scored file."""
    # Mock file operations to raise error
    mock_hf_manager.download_file.side_effect = Exception("File not found")
    
    # Initialize manager with mocked dependencies
    manager = SummaryManager(mock_config, file_manager=mock_file_manager, hf_manager=mock_hf_manager)
    
    # Process date should return None when file is not found
    result = manager.process_date('2025-04-20')
    assert result is None 