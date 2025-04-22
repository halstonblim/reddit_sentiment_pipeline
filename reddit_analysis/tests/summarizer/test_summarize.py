import pytest
import pandas as pd
from pathlib import Path
from datetime import date
from unittest.mock import Mock, patch

from reddit_analysis.summarizer.summarize import (
    SummaryManager,
    FileManager,
    HuggingFaceManager,
)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_config(tmp_path):
    """Minimal config dict compatible with SummaryManager."""
    return {
        "config": {
            "repo_id": "test/repo",
            "repo_type": "dataset",
        },
        "paths": {
            "root": tmp_path,
            "scored_dir": tmp_path / "scored",
            "hf_scored_dir": "scored",          # relative path in the Hub
            "summary_file": tmp_path / "summary.csv",
        },
        "secrets": {"HF_TOKEN": "fake"},
    }


@pytest.fixture
def mock_file_manager():
    """FileManager double with just the methods we need."""
    m = Mock(spec=FileManager)
    # read_parquet returns sample data we set in each test
    # write_csv just returns a Path so downstream code is happy
    m.write_csv.return_value = Path("summary.csv")
    return m


@pytest.fixture
def mock_hf_manager():
    """HuggingFaceManager double."""
    return Mock(spec=HuggingFaceManager)


# --------------------------------------------------------------------------- #
#  Tests                                                                      #
# --------------------------------------------------------------------------- #
def test_process_date(mock_config, mock_file_manager, mock_hf_manager):
    """End‑to‑end happy path."""
    # ---------- sample scored shard --------------------------------------- #
    sample = pd.DataFrame(
        {
            "subreddit": ["a", "a", "b", "b"],
            "sentiment": [0.8, 0.6, 0.4, 0.2],
            "score": [10, 20, 30, 40],
            "post_id": ["p1", "p2", "p3", "p4"],
            "text": ["t1", "t2", "t3", "t4"],
            "retrieved_at": pd.Timestamp.utcnow(),
        }
    )
    mock_file_manager.read_parquet.return_value = sample
    # first call → download scored file, second call (within _save_and_push_summary) unused here
    mock_hf_manager.download_file.return_value = Path("dummy.parquet")

    with patch.object(
        SummaryManager, "_load_remote_summary", return_value=pd.DataFrame()
    ):
        mgr = SummaryManager(
            mock_config, file_manager=mock_file_manager, hf_manager=mock_hf_manager
        )
        mgr.process_date("2025-04-20")

    # assertions
    mock_file_manager.read_parquet.assert_called_once()
    mock_file_manager.write_csv.assert_called_once()
    mock_hf_manager.upload_file.assert_called_once()


def test_get_processed_combinations(mock_config, mock_file_manager, mock_hf_manager):
    """The helper should translate the existing CSV into a set of tuples."""
    existing = pd.DataFrame(
        {
            "date": ["2025-04-19", "2025-04-19"],
            "subreddit": ["a", "b"],
            "mean_sentiment": [0.5, 0.3],
            "weighted_sentiment": [0.4, 0.2],
            "count": [1, 1],
        }
    )

    with patch.object(
        SummaryManager, "_load_remote_summary", return_value=existing
    ):
        mgr = SummaryManager(
            mock_config, file_manager=mock_file_manager, hf_manager=mock_hf_manager
        )
        processed = mgr.get_processed_combinations()

    assert processed == {(date(2025, 4, 19), "a"), (date(2025, 4, 19), "b")}


def test_cli_invalid_date():
    """main() should raise on malformed dates."""
    from reddit_analysis.summarizer.summarize import main

    with pytest.raises(ValueError):
        main("bad‑date‑format")


def test_cli_missing_scored_file(mock_config, mock_file_manager, mock_hf_manager):
    """Gracefully handles a missing *_scored.parquet on the Hub."""
    # download of scored file raises, but remote summary loads fine →
    mock_hf_manager.download_file.side_effect = Exception("not found")
    with patch.object(
        SummaryManager, "_load_remote_summary", return_value=pd.DataFrame()
    ):
        mgr = SummaryManager(
            mock_config, file_manager=mock_file_manager, hf_manager=mock_hf_manager
        )
        # Should simply return after printing error, not raise.
        assert mgr.process_date("2025-04-20") is None
