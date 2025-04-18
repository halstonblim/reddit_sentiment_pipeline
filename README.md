```
reddit-sentiment-pipeline/         ← root of your GitHub repo
├─ README.md                       ← high‑level project overview + architecture diagram
├─ .github/                        ← CI/CD definitions
│   ├─ workflows/
│   │   ├─ ci.yml
│   │   └─ etl.yml
│   └─ ISSUE_TEMPLATE.md
├─ scraper/                        ← your existing scraping code
│   ├─ Dockerfile
│   └─ scrape.py
├─ inference/                      ← sentiment scoring service
│   ├─ Dockerfile
│   ├─ score.py                    ← “score the raw Parquet with Replicate”
│   └─ test_score.py
├─ summarizer/                     ← computes daily_summary.csv
│   ├─ Dockerfile
│   ├─ summarize.py
│   └─ test_summarize.py
├─ frontend/                       ← Streamlit + optional FastAPI
│   ├─ Dockerfile
│   ├─ app.py
│   └─ test_app.py
├─ monitor/                        ← drift detection & Prometheus exporter
│   ├─ Dockerfile
│   ├─ drift.py
│   └─ test_drift.py
├─ infra/                          ← Terraform/Helm/CloudFormation manifests
│   ├─ terraform/
│   └─ docker-compose.yml
└─ requirements.txt                ← all Python deps for local dev & CI
```
