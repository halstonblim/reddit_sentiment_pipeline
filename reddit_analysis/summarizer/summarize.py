#!/usr/bin/env python
"""
Summarise scored shards into one daily_summary.csv

CLI examples
------------
# Summarize data for a specific date
python -m reddit_analysis.summarizer.summarize --date 2025-04-20
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

from reddit_analysis.config_utils import setup_config
from reddit_analysis.summarizer.aggregator import summary_from_df


# --------------------------------------------------------------------------- #
#  Utilities                                                                  #
# --------------------------------------------------------------------------- #
class FileManager:
    """Wrapper class for simple local file I/O that can be mocked for testing."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------- CSV helpers ------------------------------------------------- #
    def read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(
                columns=["date", "subreddit",
                         "mean_sentiment", "community_weighted_sentiment", "count"]
            )
        return pd.read_csv(path)

    def write_csv(self, df: pd.DataFrame, path: Path) -> Path:
        df.to_csv(path, index=False)
        return path

    # ---------- Parquet helper --------------------------------------------- #
    @staticmethod
    def read_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)


class HuggingFaceManager:
    """Thin wrapper around Hugging Face Hub file ops (mock‑friendly)."""
    def __init__(self, token: str, repo_id: str, repo_type: str = "dataset"):
        self.token = token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(token=token)

    def download_file(self, path_in_repo: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                filename=path_in_repo,
                token=self.token
            )
        )

    def upload_file(self, local_path: str, path_in_repo: str):
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.token
        )

    def list_files(self, prefix: str) -> List[str]:
        """List files in the HF repo filtered by prefix."""
        files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type
        )
        return [f for f in files if f.startswith(prefix)]


# --------------------------------------------------------------------------- #
#  Core manager                                                               #
# --------------------------------------------------------------------------- #
class SummaryManager:
    def __init__(
        self,
        cfg: Dict[str, Any],
        file_manager: Optional[FileManager] = None,
        hf_manager: Optional[HuggingFaceManager] = None
    ):
        self.config = cfg["config"]
        self.secrets = cfg["secrets"]
        self.paths = cfg["paths"]

        # I/O helpers
        self.file_manager = file_manager or FileManager(self.paths["root"])
        self.hf_manager = hf_manager or HuggingFaceManager(
            token=self.secrets["HF_TOKEN"],
            repo_id=self.config["repo_id"],
            repo_type=self.config.get("repo_type", "dataset"),
        )

        # Cache path for the combined summary file on disk
        self.local_summary_path: Path = self.paths["summary_file"]

    # --------------------------------------------------------------------- #
    #  Remote summary helpers                                               #
    # --------------------------------------------------------------------- #
    def _load_remote_summary(self) -> pd.DataFrame:
        """
        Ensure `daily_summary.csv` is present locally by downloading the
        latest version from HF Hub (if it exists) and return it as a DataFrame.
        """
        remote_name = self.paths["summary_file"].name

        try:
            cached_path = self.hf_manager.download_file(remote_name)
        except Exception:
            # first run – file doesn't exist yet on the Hub
            return pd.DataFrame(
                columns=["date", "subreddit",
                         "mean_sentiment", "community_weighted_sentiment", "count"]
            )

        return pd.read_csv(cached_path)

    def _save_and_push_summary(self, df: pd.DataFrame):
        """Persist the updated summary both locally and back to HF Hub."""
        self.file_manager.write_csv(df, self.local_summary_path)
        self.hf_manager.upload_file(str(self.local_summary_path),
                                    self.local_summary_path.name)

    # --------------------------------------------------------------------- #
    #  Public helpers                                                       #
    # --------------------------------------------------------------------- #
    def get_processed_combinations(self) -> Set[Tuple[date, str]]:
        """
        Return a set of (date, subreddit) pairs that are *already* present
        in the remote summary so we can de‑duplicate.
        """
        df_summary = self._load_remote_summary()
        if df_summary.empty:
            return set()

        df_summary["date"] = pd.to_datetime(df_summary["date"]).dt.date
        return {
            (row["date"], row["subreddit"])
            for _, row in df_summary.iterrows()
        }

    # --------------------------------------------------------------------- #
    #  Main workflow                                                        #
    # --------------------------------------------------------------------- #
    def process_date(self, date_str: str, overwrite: bool = False) -> None:
        """Download scored data for `date_str`, aggregate, and append/upload."""
        # ---------- Pull scored shards for the given date ------------------ #
        prefix = f"{self.paths['hf_scored_dir']}/{date_str}__"
        # List all remote shards
        try:
            all_files = self.hf_manager.list_files(self.paths['hf_scored_dir'])
        except Exception as err:
            print(f"Error: could not list scored shards in {self.paths['hf_scored_dir']}: {err}")
            return

        # Filter to shards matching this date
        try:
            shards = [fn for fn in all_files if fn.startswith(prefix) and fn.endswith('.parquet')]
        except TypeError:
            # fall back in case list_files returned a non-iterable (e.g., a mock)
            shards = [all_files]

        if not shards:
            print(f"No scored shards found for {date_str} under {self.paths['hf_scored_dir']}")
            return

        # Download and concatenate all shards
        dfs: List[pd.DataFrame] = []
        for shard in shards:
            try:
                local_path = self.hf_manager.download_file(shard)
            except Exception as err:
                print(f"Error: could not download scored shard {shard}: {err}")
                return
            dfs.append(self.file_manager.read_parquet(local_path))
        df_day = pd.concat(dfs, ignore_index=True)

        # sanity‑check
        required_cols = {"retrieved_at", "subreddit", "sentiment", "score"}
        if not required_cols.issubset(df_day.columns):
            raise ValueError(f"{shards[0]} missing columns {required_cols}")

        # ---------- Aggregate ------------------------------------------------ #
        df_summary_day = summary_from_df(df_day)

        # ---------- De‑duplication / overwrite ------------------------------ #
        existing_pairs = self.get_processed_combinations()
        if not overwrite:
            df_summary_day = df_summary_day[
                ~df_summary_day.apply(
                    lambda r: (r["date"], r["subreddit"]) in existing_pairs,
                    axis=1,
                )
            ]
        if df_summary_day.empty:
            print("Nothing new to summarise for this date.")
            return

        # ---------- Combine with historical summary ------------------------- #
        df_summary = self._load_remote_summary()
        if overwrite:
            df_summary = df_summary[df_summary["date"] != date_str]
            
        # Remove weighted_sentiment column if it exists
        if "weighted_sentiment" in df_summary.columns:
            df_summary = df_summary.drop(columns=["weighted_sentiment"])

        df_out = (
            pd.concat([df_summary, df_summary_day], ignore_index=True)
            if not df_summary.empty
            else df_summary_day
        )
        df_out["date"] = pd.to_datetime(df_out["date"]).dt.date
        df_out.sort_values(["date", "subreddit"], inplace=True)
        
        # Ensure the weighted_sentiment column is dropped from final output
        if "weighted_sentiment" in df_out.columns:
            df_out = df_out.drop(columns=["weighted_sentiment"])

        # Round floating point columns to 4 decimal places
        if "mean_sentiment" in df_out.columns:
            df_out["mean_sentiment"] = df_out["mean_sentiment"].round(4)
        if "community_weighted_sentiment" in df_out.columns:
            df_out["community_weighted_sentiment"] = df_out["community_weighted_sentiment"].round(4)

        # ---------- Save & upload ------------------------------------------- #
        self._save_and_push_summary(df_out)
        print(f"Updated {self.local_summary_path.name} → {len(df_out)} rows")


# --------------------------------------------------------------------------- #
#  CLI entry‑point                                                            #
# --------------------------------------------------------------------------- #
def main(date_str: str, overwrite: bool = False) -> None:
    if not date_str:
        raise ValueError("--date is required (YYYY-MM-DD)")

    # Confirm valid date
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date: {date_str} (expected YYYY‑MM‑DD)")

    cfg = setup_config()
    SummaryManager(cfg).process_date(date_str, overwrite)


if __name__ == "__main__":
    from reddit_analysis.common_metrics import run_with_metrics
    parser = argparse.ArgumentParser(
        description="Summarize scored Reddit data for a specific date."
    )
    parser.add_argument("--date", required=True,
                        help="YYYY-MM-DD date to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace any existing rows for this date")
    args = parser.parse_args()
    run_with_metrics("summarize", main, args.date, args.overwrite)
