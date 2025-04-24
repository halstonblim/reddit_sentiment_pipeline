import os
from pathlib import Path
import pytest
import pandas as pd
from datetime import datetime, date
import pytz
from unittest.mock import Mock, patch

from reddit_analysis.scraper.scrape import RedditScraper, RedditAPI, FileManager, HuggingFaceManager

@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        'config': {
            'repo_id': 'test/repo',
            'repo_type': 'dataset',
            'subreddits': [
                {'name': 'test1', 'post_limit': 2, 'comment_limit': 2},
                {'name': 'test2', 'post_limit': 2, 'comment_limit': 2}
            ],
            'post_limit': 100,
            'timezone': 'UTC'
        },
        'paths': {
            'raw_dir': Path('data/raw'),
            'logs_dir': Path('logs'),
            'hf_raw_dir': 'data/raw'
        },
        'secrets': {
            'HF_TOKEN': 'test_token',
            'REDDIT_CLIENT_ID': 'test_id',
            'REDDIT_CLIENT_SECRET': 'test_secret',
            'REDDIT_USER_AGENT': 'test_agent'
        }
    }

@pytest.fixture
def mock_reddit_api():
    """Create a mock RedditAPI."""
    mock = Mock(spec=RedditAPI)
    
    # Create mock submission objects
    mock_submissions = []
    for i in range(2):
        submission = Mock()
        submission.id = f'post{i}'
        submission.title = f'Test Post {i}'
        submission.selftext = f'Test content {i}'
        submission.score = i + 1
        submission.created_utc = datetime.now(pytz.UTC).timestamp()
        submission.url = f'https://reddit.com/test{i}'
        submission.num_comments = i * 10
        
        # Mock the comments
        comment = Mock()
        comment.id = f'comment{i}'
        comment.body = f'Test comment {i}'
        comment.score = i + 5
        comment.created_utc = datetime.now(pytz.UTC).timestamp()
        comment.parent_id = submission.id
        
        # Set up comment attributes
        submission.comments = Mock()
        submission.comments._comments = [comment]
        submission.comments.replace_more = Mock(return_value=None)
        
        mock_submissions.append(submission)
    
    # Set up the mock subreddit
    mock_subreddit = Mock()
    mock_subreddit.top.return_value = mock_submissions
    mock.get_subreddit.return_value = mock_subreddit
    
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

def test_get_posts(mock_config, mock_reddit_api):
    """Test the get_posts method."""
    # Initialize scraper with mocked RedditAPI
    scraper = RedditScraper(mock_config, reddit_api=mock_reddit_api)
    
    # Get posts for a test subreddit
    df = scraper.get_posts({'name': 'test1', 'post_limit': 2, 'comment_limit': 2})
    
    # Verify DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 2 posts + 2 comments
    
    # Verify posts
    posts_df = df[df['type'] == 'post']
    assert len(posts_df) == 2
    assert posts_df['subreddit'].iloc[0] == 'test1'
    assert posts_df['post_id'].iloc[0] == 'post0'
    assert posts_df['post_id'].iloc[1] == 'post1'
    
    # Verify comments
    comments_df = df[df['type'] == 'comment']
    assert len(comments_df) == 2
    assert comments_df['subreddit'].iloc[0] == 'test1'
    assert comments_df['post_id'].iloc[0] == 'comment0'
    assert comments_df['parent_id'].iloc[0] == 'post0'

def test_upload_to_hf_deduplication(mock_config, mock_file_manager, mock_hf_manager):
    """Test the upload_to_hf method with deduplication."""
    # Create test DataFrames
    prev_df = pd.DataFrame({
        'post_id': ['post0', 'post1'],
        'title': ['Old Post 0', 'Old Post 1'],
        'text': ['Old content 0', 'Old content 1'],
        'score': [1, 2],
        'subreddit': ['test1', 'test1'],
        'created_utc': [datetime.now(pytz.UTC)] * 2,
        'url': ['https://reddit.com/old0', 'https://reddit.com/old1'],
        'num_comments': [10, 20]
    })
    
    new_df = pd.DataFrame({
        'post_id': ['post1', 'post2'],
        'title': ['New Post 1', 'New Post 2'],
        'text': ['New content 1', 'New content 2'],
        'score': [3, 4],
        'subreddit': ['test1', 'test1'],
        'created_utc': [datetime.now(pytz.UTC)] * 2,
        'url': ['https://reddit.com/new1', 'https://reddit.com/new2'],
        'num_comments': [30, 40]
    })
    
    # Mock file operations
    mock_hf_manager.download_file.return_value = Path('test.parquet')
    mock_file_manager.read_parquet.return_value = prev_df
    
    # Initialize scraper with mocked dependencies
    scraper = RedditScraper(
        mock_config,
        file_manager=mock_file_manager,
        hf_manager=mock_hf_manager
    )
    
    # Upload new data
    scraper._upload_to_hf(new_df, '2025-04-20')
    
    # Verify file operations
    mock_file_manager.save_parquet.assert_called_once()
    mock_hf_manager.upload_file.assert_called_once()

def test_cli_missing_env(monkeypatch, tmp_path):
    """Test CLI with missing environment variables."""
    # Create a temporary .env file without required variables
    env_path = tmp_path / '.env'
    env_path.write_text('')
    
    # Set environment variable to point to our test .env
    monkeypatch.setenv('REDDIT_ANALYSIS_ENV', str(env_path))
    
    # Remove any existing Reddit API credentials from environment
    for key in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']:
        monkeypatch.delenv(key, raising=False)
    # Ensure HF_TOKEN is present so only Reddit client vars are missing
    monkeypatch.setenv('HF_TOKEN', 'dummy_hf_token')
    # Mock Streamlit's HAS_STREAMLIT to True
    monkeypatch.setattr('reddit_analysis.config_utils.HAS_STREAMLIT', True)
    # Mock is_running_streamlit to True
    monkeypatch.setattr('reddit_analysis.config_utils.is_running_streamlit', lambda: True)
    # Mock Streamlit secrets to return None
    mock_secrets = Mock()
    mock_secrets.get.return_value = None
    monkeypatch.setattr('streamlit.secrets', mock_secrets)
    # Print for debug
    import os
    print('DEBUG: REDDIT_CLIENT_ID value before main:', os.environ.get('REDDIT_CLIENT_ID'))
    # Run the CLI with --date argument
    with pytest.raises(ValueError) as exc_info:
        from reddit_analysis.scraper.scrape import main
        main('2025-04-20')
    assert "Missing required environment variables: REDDIT_CLIENT_ID" in str(exc_info.value)