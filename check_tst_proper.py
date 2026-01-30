
import mlflow
import torch
import os
import shutil
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def check_tst():
    print("Checking TST v1...")
    try:
        v1 = client.get_model_version("BTCUSDT_tst", "1")
        print(f"TST Run ID: {v1.run_id}")
        
        temp_dir = "temp_tst_check"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        local_path = client.download_artifacts(v1.run_id, "BTCUSDT_tst", dst_path=temp_dir)
        print(f"Downloaded to {local_path}")
        
        # Determine strict path
        model_uri = local_path
        print(f"Loading from {model_uri}...")
        
        # Load with mlflow pytorch
        model = mlflow.pytorch.load_model(model_uri)
        print(f"SUCCESS: Loaded TST v1. Type: {type(model)}")
        
    except Exception as e:
        print(f"FAILURE TST: {e}")

if __name__ == "__main__":
    check_tst()
