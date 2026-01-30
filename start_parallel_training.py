import runpy
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

# Auto-configure GCS Credentials if not set
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    # Look for the specific key file found in the directory
    key_file = "dhaya123-335710-039eabaad669.json"
    if os.path.exists(key_file):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_file)
        print(f"Set GOOGLE_APPLICATION_CREDENTIALS to {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    else:
        print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not set and key file not found.")

# Disable WandB to prevent login prompts
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "disabled"
    print("Set WANDB_MODE to disabled")

print("Starting parallel training pipeline...")
try:
    # Run the parallel training script as a module
    runpy.run_module("utils.trainer.train_paralelly", run_name="__main__")
except Exception as e:
    print(f"Error running training pipeline: {e}")
    sys.exit(1)
