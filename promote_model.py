
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def promote_v5():
    model_name = "BTCUSDT_lightgbm"
    version = "5"
    print(f"Transitioning {model_name} v{version} to Production...")
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        print("Success! v6 should be archived and v5 is now Production.")
        
        # Verify
        model_version = client.get_model_version(model_name, version)
        print(f"Verification: v{model_version.version} is now in stage '{model_version.current_stage}'")
        
    except Exception as e:
        print(f"Failed to transition stage: {e}")

if __name__ == "__main__":
    promote_v5()
