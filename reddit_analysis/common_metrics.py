from prometheus_client import CollectorRegistry, Histogram, Gauge, push_to_gateway
import time, os, sys

REGISTRY = CollectorRegistry()
DURATION = Histogram(
    "process_duration_seconds",
    "Wall-clock time for a pipeline stage",
    ["job"],
    registry=REGISTRY,
    buckets=(1, 5, 15, 30, 60, 120, 300, 600)
)
SUCCESS = Gauge("job_success", "Did the job finish without exception? (1/0)", ["job"], registry=REGISTRY)

def run_with_metrics(job_name, func, *args, **kwargs):
    start = time.time()
    ok = 0
    try:
        result = func(*args, **kwargs)
        ok = 1
        return result
    finally:
        DURATION.labels(job=job_name).observe(time.time() - start)
        SUCCESS.labels(job=job_name).set(ok)
        push_to_gateway(os.getenv("PROM_PUSHGW_HOST", "localhost:9091"), job=job_name, registry=REGISTRY)
