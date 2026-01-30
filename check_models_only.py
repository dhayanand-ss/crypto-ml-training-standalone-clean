
import requests
import time
import json

BASE_URL = "http://127.0.0.1:8025"

def check_models():
    print(f"Checking models at {BASE_URL}...")
    try:
        # Wait a bit for startup
        for i in range(10):
            try:
                requests.get(f"{BASE_URL}/health", timeout=1)
                break
            except:
                time.sleep(1)
        
        resp = requests.get(f"{BASE_URL}/models")
        if resp.status_code == 200:
            models = resp.json()
            print(f"Models visible: {len(models)}")
            print(json.dumps(models, indent=2))
            
            # Verify BTCUSDT_lightgbm v5 match
            found = False
            for m in models:
                if m['model_name'] == 'BTCUSDT_lightgbm' and str(m['version']) == '5':
                    found = True
                    print("\n[SUCCESS] BTCUSDT_lightgbm v5 is visible!")
            
            if not found:
                 print("\n[FAILURE] BTCUSDT_lightgbm v5 NOT found in list.")
        else:
            print(f"Failed to list models: {resp.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_models()
