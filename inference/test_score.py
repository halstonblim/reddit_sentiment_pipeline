"""
Unit‑test for score.score_date()

• Builds a tiny Parquet in a temp dir
• Stubs download_raw_file → local parquet
• Stubs hf_api.upload_file   → no‑op
• Stubs replicate_client.run → dummy output  [1,0,1,0]

No network traffic, no Replicate token needed.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load .env so score.py can import its secrets (even though we won't use them)
# ------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from score import score_date  # noqa: E402


# ------------------------------------------------------------------
# Fixture that patches score.py in‑place
# ------------------------------------------------------------------
TEST_TEXTS = ["I love apple", "I hate apple", "This sucks", "Amazing job"]


@pytest.fixture(autouse=True)
def patch_score(tmp_path, monkeypatch):
    # 1️⃣ fake raw parquet
    raw_dir = tmp_path / "data_raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "2025-01-01.parquet"
    pd.DataFrame({"text": TEST_TEXTS}).to_parquet(raw_path)

    # 2️⃣ stub download_raw_file
    monkeypatch.setattr("score.download_raw_file", lambda date: raw_path, raising=True)

    # 3️⃣ stub upload_file (no network)
    monkeypatch.setattr("score.hf_api.upload_file", lambda *a, **k: None, raising=True)

    # 4️⃣ stub replicate_client.run to deterministic output
    dummy_out = {
        "predicted_labels": [1, 0, 1, 0],
        "confidences":     [1, 0, 1, 0],
    }
    monkeypatch.setattr("score.replicate_client.run", lambda *a, **k: dummy_out, raising=True)

    # 5️⃣ write to tmp dir
    monkeypatch.setattr("score.project_root", tmp_path, raising=True)

    yield


# ------------------------------------------------------------------
# The test
# ------------------------------------------------------------------
def test_score_date_offline(tmp_path):
    score_date("2025-01-01")

    scored = tmp_path / "data_scored" / "2025-01-01.parquet"
    assert scored.exists()

    df = pd.read_parquet(scored)
    assert list(df["sentiment"])   == [1, 0, 1, 0]
    assert list(df["confidence"])  == [1, 0, 1, 0]
