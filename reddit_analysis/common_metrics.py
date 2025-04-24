from prometheus_client import CollectorRegistry, Histogram, Gauge, push_to_gateway
import time, os, sys
from reddit_analysis.config_utils import load_environment, get_secret

REGISTRY = CollectorRegistry()
EXEC_DURATION = Gauge(
    "job_duration_seconds",
    "Wall-clock duration of the most recent job run",
    ["job"],
    registry=REGISTRY
)
SUCCESS = Gauge("job_success", "Did the job finish without exception? (1/0)", ["job"], registry=REGISTRY)
load_environment()
GATEWAY = get_secret("PROM_PUSHGW_HOST")

def run_with_metrics(job_name, func, *args, **kwargs):
    start = time.time()
    ok = 0
    try:
        result = func(*args, **kwargs)
        ok = 1
        return result
    finally:
        elapsed = time.time() - start
        EXEC_DURATION.labels(job=job_name).set(elapsed)
        SUCCESS.labels(job=job_name).set(ok)
        if GATEWAY:
            try:
                print("Pushing to gateway")
                push_to_gateway(GATEWAY, job=job_name, registry=REGISTRY)
            except Exception as e:
                print(f"[metrics] WARNING: push to {GATEWAY} failed: {e}")