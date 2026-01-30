
import lightgbm as lgb
import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def check_v5():
    print("Checking v5...")
    try:
        # Get run_id for v5
        v5 = client.get_model_version("BTCUSDT_lightgbm", "5")
        print(f"v5 Run ID: {v5.run_id}")
        
        # Download
        temp_dir = "temp_v5_check"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        local_path = client.download_artifacts(v5.run_id, "BTCUSDT_lightgbm", dst_path=temp_dir)
        print(f"Downloaded to {local_path}")
        
        # Load
        model_path = os.path.join(local_path, "model.lgb")
        if not os.path.exists(model_path):
             # Try listing
             print(f"Listing {local_path}: {os.listdir(local_path)}")
             model_path = os.path.join(local_path, "model.txt")
        
        bst = lgb.Booster(model_file=model_path)
        print(f"SUCCESS: Loaded v5. Features: {bst.num_feature()}")
        
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    check_v5()
