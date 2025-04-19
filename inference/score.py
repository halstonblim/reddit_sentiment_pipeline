import os
import argparse
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import yaml
import replicate
from huggingface_hub import HfApi, hf_hub_download

# Load secrets from .env in project root
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# Load non-secret config from YAML in project root
with open(project_root / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Environment variables
HF_TOKEN = os.getenv('HF_TOKEN')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Config parameters
REPO_ID = config['repo_id']
REPO_TYPE = config.get('repo_type', 'dataset')
RAW_DIR = 'data_raw'
SCORED_DIR = 'data_scored'
BATCH_SIZE = config.get('batch_size', 16)
REPLICATE_MODEL = config['replicate_model']

# Initialize clients
hf_api = HfApi(token=HF_TOKEN)
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)


def list_dates_in_dir(prefix: str) -> set:
    """List dates (YYYY-MM-DD) for Parquet files under a folder on the HF repo."""
    files = hf_api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    dates = set()
    for fn in files:
        if fn.startswith(f"{prefix}/") and fn.endswith('.parquet'):
            parts = Path(fn).stem
            dates.add(parts)
    return dates


def download_raw_file(date: str) -> Path:
    """Download raw Parquet for given date."""
    out_path = hf_hub_download(repo_id=REPO_ID,
                               repo_type=REPO_TYPE,
                               filename=f"{RAW_DIR}/{date}.parquet",
                               token=HF_TOKEN)
    return Path(out_path)


def score_date(date: str):
    """Process a single date: download, score, save, and upload."""
    print(f"Scoring date: {date}")
    raw_path = download_raw_file(date)
    df = pd.read_parquet(raw_path)

    sentiments = []
    confidences = []
    texts = df['text'].tolist()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        # call replicate model with JSON-formatted list of strings
        result = replicate_client.run(
            REPLICATE_MODEL,
            input={'texts': json.dumps(batch)}
        )
        # extract outputs
        preds = result.get('predicted_labels', [])
        confs = result.get('confidences', [])
        sentiments.extend(preds)
        confidences.extend(confs)

    df['sentiment'] = sentiments
    df['confidence'] = confidences

    # Write out scored Parquet
    scored_path = project_root / SCORED_DIR / f"{date}.parquet"
    scored_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(scored_path, index=False)

    # Upload scored file to HF dataset
    hf_api.upload_file(
        path_or_fileobj=str(scored_path),
        path_in_repo=f"{SCORED_DIR}/{date}.parquet",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN
    )
    print(f"Uploaded scored file for {date} to {REPO_ID}/{SCORED_DIR}/{date}.parquet")


def main(date_arg: str = None):
    # Fetch available dates
    raw_dates = list_dates_in_dir(RAW_DIR)
    scored_dates = list_dates_in_dir(SCORED_DIR)
    pending = sorted(raw_dates - scored_dates)

    if date_arg:
        if date_arg not in raw_dates:
            print(f"No raw file found for date {date_arg}")
            return
        if date_arg in scored_dates:
            print(f"Scored file already exists for date {date_arg}")
            return
        to_process = date_arg
    else:
        if not pending:
            print("No new raw files to score.")
            return
        to_process = pending[0]

    score_date(to_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score raw HF dataset files via Replicate.')
    parser.add_argument('--date', type=str, help='YYYY-MM-DD to process (optional)')
    args = parser.parse_args()
    main(args.date)
