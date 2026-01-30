
import mlflow
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)

# Use the correct tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def check_all_versions(model_name):
    print(f"\n==================================================")
    print(f"EXHAUSTIVE INTEGRITY CHECK: {model_name}")
    print(f"==================================================\n", flush=True)
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        # Sort descending to check newest first
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    except Exception as e:
        print(f"Error listing versions: {e}")
        return

    for v in versions:
        print(f"\n>>>> VERSION {v.version} (Stage: {v.current_stage})")
        
        # Create a temp dir
        temp_dir = f"temp_check_{model_name}_v{v.version}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        try:
            # 1. Download
            print(f"    Downloading artifacts...")
            local_path = client.download_artifacts(v.run_id, model_name, dst_path=temp_dir)
            
            # 2. Find model file
            model_file = None
            for root, dirs, files in os.walk(local_path):
                if 'model.lgb' in files:
                    model_file = os.path.join(root, 'model.lgb')
                    break
                if 'model.txt' in files:
                    model_file = os.path.join(root, 'model.txt')
                    break
            
            if not model_file:
                # Fallback to any file
                for root, dirs, files in os.walk(local_path):
                    for f in files:
                        if f.endswith('.lgb') or f.endswith('.txt'):
                            model_file = os.path.join(root, f)
                            break
                    if model_file: break

            if not model_file:
                print(f"    [SKIP] No LightGBM model file found.")
                continue
                
            # 3. Load
            print(f"    Loading {os.path.basename(model_file)}...")
            try:
                bst = lgb.Booster(model_file=model_file)
                print(f"    [SUCCESS] Loaded v{v.version}! Features: {bst.num_feature()}")
            except Exception as e:
                print(f"    [FAILURE] Load failed for v{v.version}: {e}")
                
        except Exception as e:
            print(f"    [ERROR] Process failed for v{v.version}: {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    print(f"\n==================================================")
    print(f"CHECK COMPLETE")
    print(f"==================================================")

if __name__ == "__main__":
    check_all_versions("BTCUSDT_lightgbm")
