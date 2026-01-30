
import requests
import json
import sys
import time

BASE_URL = "http://127.0.0.1:8023"

def trigger_refresh_and_wait():
    print("Triggering model refresh for BTCUSDT_lightgbm v5...")
    try:
        requests.post(f"{BASE_URL}/refresh", json={"model_name": "BTCUSDT_lightgbm", "version": "5"})
    except:
        pass
    time.sleep(2)

def run_test():
    print(f"Testing FastAPI at {BASE_URL}...")
    
    # 1. Health check
    requests.get(f"{BASE_URL}/health", timeout=5)

    # Trigger manual refresh
    trigger_refresh_and_wait()

    # 3. Predict with BTCUSDT_lightgbm v5
    model_name = "BTCUSDT_lightgbm"
    # Pass version as string "5" to match MLflow version, or int if API requires index. 
    # Current codebase seems to support string version in _make_prediction if passed not as query index.
    # But /predict endpoint signature might be picky.
    # Let's try passing '5' as param.
    
    # Feature count guess: 55
    features = [[0.5] * 55]
    
    params = {
        "model_name": model_name,
        "version": "5"
    }
    
    print(f"\nAttempting prediction on {model_name} v5 with 55 features...")
    try:
        resp = requests.post(f"{BASE_URL}/predict", params=params, json=features)
        print(f"Prediction response code: {resp.status_code}")
        if resp.status_code == 200:
            print("Success!")
            print(resp.json())
        else:
            print("Failed:")
            print(resp.text)
    except Exception as e:
        print(f"Prediction request failed: {e}")


if __name__ == "__main__":
    run_test()
