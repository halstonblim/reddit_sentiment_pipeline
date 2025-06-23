### `test_config_utils.py`  
- **Functions under test**  
  - `load_config(path)` — reads settings from a YAML file.  
  - `get_secret(key)` — retrieves a secret first from `os.environ`, then from `streamlit.secrets`, else raises.  
- **Patching & mocking**  
  - Environment variables via `os.environ` or `monkeypatch.setenv()` / `monkeypatch.delenv()`.  
  - `reddit_analysis.config_utils.HAS_STREAMLIT` toggled to simulate presence of Streamlit.  
  - `streamlit.secrets` replaced with a `MockSecrets` object exposing a `.get(key)` method.  
- **Example inputs**  
  - A temporary `config.yaml` with keys like `repo_id: test/repo`, `batch_size: 16`, `replicate_model: test/model`.  
  - Secret key `"TEST_SECRET"` set in `os.environ` or returned by `MockSecrets.get()`.  
  - Missing secret scenario triggers `ValueError("Required secret TEST_SECRET not found…")`.  

---

### `test_scrape.py`  
- **Methods under test**  
  - `RedditScraper.get_posts(subreddit)` — calls PRAW client’s `.subreddit(...).top()` and returns a DataFrame with columns `post_id, title, text, score, subreddit, created_utc, url, num_comments`.  
  - `RedditScraper.upload_to_hf(df, date)` — downloads existing parquet via `hf_hub_download`, deduplicates by `post_id`, then calls `hf_api.upload_file(...)`.  
  - `main(date)` CLI — loads config, checks for Reddit credentials, raises if missing.  
- **Patching & mocking**  
  - A fake PRAW client (`mock_reddit_client`) whose `.subreddit().top()` yields two `Mock` submissions (ids `post0`, `post1`).  
  - `hf_hub_download` patched to return a path for a “previous” parquet file containing `prev_df`.  
  - `mock_hf_api.upload_file` to capture the uploaded parquet path.  
  - Environment via `monkeypatch` and `reddit_analysis.config_utils.HAS_STREAMLIT` + `streamlit.secrets`.  
- **Example inputs**  
  - **`get_posts`** uses two submissions with `id='post0'`, `title='Test Post 0'`, etc., expecting a 2‑row DataFrame.  
  - **`upload_to_hf`** combines `prev_df` (posts 0 & 1) with `new_df` (posts 1 & 2), resulting in only `post1` & `post2` uploaded.  
  - **CLI** invoked with no Reddit env vars, raising `ValueError("Missing required Reddit API credentials")`.  

---

### `test_summarize.py`  
- **Methods under test**  
  - `RedditSummarizer.summarize_date(date)` — downloads scored parquet, groups by `subreddit`, and computes `mean_sentiment`, `count`, `total_score`, `weighted_sentiment`, plus `date`.  
  - `RedditSummarizer.update_summary(df)` — appends to or creates `summary_file`, preserving chronological order.  
  - CLI entrypoint in `main(date)` — validates date format or scored-file existence.  
- **Patching & mocking**  
  - `hf_hub_download` patched to return a temp parquet containing `sample_scored_data` (4 rows for two subreddits).  
  - `reddit_analysis.config_utils.HAS_STREAMLIT` and `streamlit.secrets.get(...)` for missing-file tests.  
- **Example inputs & expectations**  
  - **`summarize_date`**:  
    ```python
    sample_scored_data = pd.DataFrame({
      'subreddit': ['test1','test1','test2','test2'],
      'sentiment': [0.8,0.6,0.4,0.2],
      'score': [10,20,30,40],
      …
    })
    ```  
    – Expect two summary rows:  
    - test1: `mean_sentiment≈0.7`, `count=2`, `total_score=30`, `weighted_sentiment≈0.6667`  
    - test2: `mean_sentiment≈0.3`, `count=2`, `total_score=70`, `weighted_sentiment≈0.2857`  
  - **`update_summary`**: merges an initial 2‑row file for `2025-04-19` with a new 2‑row file for `2025-04-20`, ending with 4 total rows.  
  - **CLI invalid date**: `main('2025-04-20-invalid')` → `ValueError("Invalid date format")`.  
  - **Missing scored file**: patched `hf_hub_download` raises → `ValueError("Failed to download scored file…")`.  

---

### `test_score.py`  
- **Class & functions under test**  
  - `RedditScorer.score_date(date)` — downloads input parquet, asserts required columns (`text, score, post_id, subreddit`), splits into batches, calls `replicate_client.run()`, injects `sentiment` & `confidence`, writes parquet, then calls `hf_api.upload_file()`.  
  - CLI `main(date)` — reads `.env` or `streamlit.secrets`, requires `REPLICATE_API_TOKEN`, else raises.  
- **Patching & mocking**  
  - `hf_hub_download` patched to return a temp parquet for the “input” DataFrame.  
  - `mock_hf_api` supplying a stubbed `upload_file` method.  
  - `mock_replicate_client.run` side‑effect that:  
    ```python
    texts = json.loads(input['texts'])
    sentiments = ['positive' if i%2==0 else 'negative' for i in range(len(texts))]
    confidences = [0.9 if i%2==0 else 0.8 for i in range(len(texts))]
    ```
  - `reddit_analysis.config_utils.HAS_STREAMLIT` + `streamlit.secrets.get(...)` for the CLI missing‑token test.  
- **Example inputs & expectations**  
  - **`test_score_date`**: input DataFrame with two rows (`'Test text 1'`, `'Test text 2'`), expects uploaded parquet to have `sentiment=['positive','negative']`, `confidence=[0.9,0.8]` and all six columns present.  
  - **`test_score_date_missing_columns`**: input missing `post_id`/`subreddit` → `ValueError("missing expected columns")`.  
  - **`test_score_date_batch_processing`**: input of 5 texts, `batch_size=2` → `replicate_client.run` called 3 times, final uploaded file contains all 5 rows.  
  - **`test_cli_missing_token`**: no `REPLICATE_API_TOKEN` in env or secrets → `ValueError("REPLICATE_API_TOKEN is required for scoring")`.  
