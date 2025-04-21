from fastapi import FastAPI
from data_utils import load_summary

app = FastAPI(title="Sentiment Trends API")

@app.get("/trends")
def trends():
    df = load_summary()
    return df.to_dict(orient="records")[-60:]  # last 60 days
