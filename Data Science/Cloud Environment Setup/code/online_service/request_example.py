"""Send real requests to the running delivery-ETA service and print the transcript.

Assumes the service is already running (see app.py's docstring for the uvicorn command).
This is the Python-`requests` equivalent of a Java integration test that hits a REST
endpoint with a JVM HTTP client (`RestTemplate`/`HttpClient`) instead of curl.

Run:  .venv-serving/Scripts/python request_example.py
"""
from __future__ import annotations

import json

import requests

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    health = requests.get(f"{BASE_URL}/health", timeout=5)
    print("GET /health ->", health.status_code)
    print(json.dumps(health.json(), indent=2))
    print()

    valid_payload = {"distance_km": 4.2, "num_items": 3, "is_peak_hour": True}
    resp = requests.post(f"{BASE_URL}/predict", json=valid_payload, timeout=5)
    print("POST /predict (valid) ->", resp.status_code)
    print("request body:", json.dumps(valid_payload))
    print("response body:", json.dumps(resp.json(), indent=2))
    print()

    # A deliberately invalid request — negative distance — to show pydantic's boundary
    # validation rejecting bad input before it ever reaches the model (Field(gt=0) in
    # app.py's PredictRequest).
    invalid_payload = {"distance_km": -1.0, "num_items": 3, "is_peak_hour": True}
    bad_resp = requests.post(f"{BASE_URL}/predict", json=invalid_payload, timeout=5)
    print("POST /predict (invalid: negative distance_km) ->", bad_resp.status_code)
    print("request body:", json.dumps(invalid_payload))
    print("response body:", json.dumps(bad_resp.json(), indent=2))


if __name__ == "__main__":
    main()
