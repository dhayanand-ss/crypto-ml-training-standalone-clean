
import os
import sys
from pathlib import Path

# Mock app setup
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.artifact_control.model_manager import ModelManager

def debug():
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    print(f"MLFLOW_TRACKING_URI: {uri}")
    
    mm = ModelManager(tracking_uri=uri)
    print(f"Client URI: {mm.client.tracking_uri}")
    
    print("Searching registered models...")
    models = mm.client.search_registered_models()
    print(f"Found {len(models)} models.")
    for m in models:
        print(f" - {m.name}")
        versions = mm.client.get_latest_versions(m.name, stages=["Production"])
        print(f"   Production versions: {[v.version for v in versions]}")
        
    print("\nExplicitly checking BTCUSDT_lightgbm...")
    try:
        model = mm.client.get_registered_model("BTCUSDT_lightgbm")
        print(f"Found model: {model.name}")
        versions = mm.client.get_latest_versions("BTCUSDT_lightgbm", stages=["Production"])
        print(f"Production versions: {[v.version for v in versions]}")
    except Exception as e:
         print(f"Error getting BTCUSDT_lightgbm: {e}")

if __name__ == "__main__":
    debug()
