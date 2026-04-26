"""Quick smoke-test for the Fintech environment server.

Usage:
    python test_client.py

Make sure the environment server is running first:
    python -m uvicorn server.app:app --port 8001
"""
import asyncio
import os
import requests

_ENV_PORT = int(os.getenv("ENV_SERVER_PORT", "8001"))
_BASE_URL = f"http://localhost:{_ENV_PORT}"


def main():
    print(f"Connecting to Fintech Environment Server at {_BASE_URL} ...\n")

    # 1. Reset
    print("--- Resetting Environment ---")
    r = requests.post(f"{_BASE_URL}/reset", timeout=10)
    print(f"Status: {r.status_code}  Body: {r.json()}\n")

    # 2. get_providers
    print("--- Tool: get_providers ---")
    r = requests.post(
        f"{_BASE_URL}/step",
        json={"action": {"tool": "get_providers", "provider_name": "directory"}},
        timeout=10,
    )
    obs = r.json().get("observation", {})
    print(f"Status: {r.status_code}  Reward: {obs.get('reward')}  Data: {obs.get('data')}\n")

    # 3. check_provider — Wise
    print("--- Tool: check_provider (Wise) ---")
    r = requests.post(
        f"{_BASE_URL}/step",
        json={"action": {"tool": "check_provider", "provider_name": "Wise"}},
        timeout=10,
    )
    obs = r.json().get("observation", {})
    print(f"Status: {r.status_code}  Reward: {obs.get('reward')}  Data: {obs.get('data')}\n")

    # 4. execute_transaction — intentionally bad payload (missing required field)
    print("--- Tool: execute_transaction (bad payload — missing fields) ---")
    r = requests.post(
        f"{_BASE_URL}/step",
        json={
            "action": {
                "tool": "execute_transaction",
                "provider_name": "Wise",
                "payload": {"amount": "200"},  # intentionally incomplete
            }
        },
        timeout=10,
    )
    obs = r.json().get("observation", {})
    print(f"Status: {r.status_code}  Reward: {obs.get('reward')}  Data: {obs.get('data')}\n")

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
