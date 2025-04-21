# Reddit Analysis Pipeline

This project provides a pipeline for scraping, analyzing, and summarizing Reddit data by:
1. Scraping top posts from configured subreddits
2. Scoring the sentiment of those posts
3. Summarizing the sentiment data by date and subreddit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit_analysis.git
cd reddit_analysis
```

2. Create a `.env` file with the required credentials:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
HF_TOKEN=your_huggingface_token
REPLICATE_API_TOKEN=your_replicate_token
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

Ensure `config.yaml` exists with required configuration:
```yaml
repo_id: "your_huggingface_repo_id"
repo_type: "dataset"
timezone: "UTC"

# Directory configuration (all paths are relative to project root)
raw_dir: "data_raw"        # Where raw scraped data is stored locally
scored_dir: "data_scored"  # Where sentiment-scored data is stored locally
logs_dir: "logs"           # Where log files are stored
summary_file: "subreddit_daily_summary.csv"  # Path to summary file

# Hugging Face repository directories (paths within the HF dataset repository)
hf_raw_dir: "data_raw"     # Directory for raw data in HF repository
hf_scored_dir: "data_scored" # Directory for scored data in HF repository

# Data processing parameters
push_to_hf: true
batch_size: 16
replicate_model: "replicate/sentiment-model"

# Subreddits to scrape
subreddits:
  - name: "python"
    post_limit: 25
    comment_limit: 10
  - name: "datascience"
    post_limit: 25
    comment_limit: 10
```

## Usage

Each script in the pipeline requires a date parameter in YYYY-MM-DD format.

### 1. Scraping Reddit Data

Scrapes posts from the configured subreddits for a specific date:

```bash
python -m reddit_analysis.scraper.scrape --date 2025-04-20
```

This produces a parquet file in the `raw_dir` directory and optionally uploads it to Hugging Face in the `hf_raw_dir` path.

### 2. Scoring Sentiment

Scores the sentiment of posts and comments for a specific date:

```bash
python -m reddit_analysis.inference.score --date 2025-04-20
```

This reads from the `hf_raw_dir` HF repository path, scores the sentiment using Replicate, saves to the local `scored_dir` directory, and uploads to the `hf_scored_dir` repository path.

### 3. Summarizing Data

Summarizes the scored data for a specific date:

```bash
python -m reddit_analysis.summarizer.summarize --date 2025-04-20
```

This reads from the `hf_scored_dir` repository path and updates the summary file specified in the configuration.

## Running the Entire Pipeline

To run the entire pipeline for a specific date:

```bash
python -m reddit_analysis.scraper.scrape --date 2025-04-20
python -m reddit_analysis.inference.score --date 2025-04-20
python -m reddit_analysis.summarizer.summarize --date 2025-04-20
```

## Project Structure

```
reddit_analysis/
├── __init__.py
├── config_utils.py
├── scraper/
│   ├── __init__.py
│   └── scrape.py
├── inference/
│   ├── __init__.py
│   └── score.py
└── summarizer/
    ├── __init__.py
    ├── aggregator.py
    └── summarize.py
```

## Development

To set up the development environment:

```bash
pip install -e ".[dev]"
```

This will install the package in development mode with additional development dependencies:
- pytest for testing
- black for code formatting
- isort for import sorting

## Testing

To run the tests:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
