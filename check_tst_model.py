
from utils.artifact_control.model_manager import ModelManager
import torch
import traceback

def check_tst():
    try:
        mm = ModelManager()
        print('Loading BTCUSDT_tst v1...')
        model, ver = mm.load_model('BTCUSDT_tst', '1', model_type='pytorch')
        print(f'BTCUSDT_tst v1 loaded successfully: {type(model)}')
        
        # Verify it's a pytorch module
        if isinstance(model, torch.nn.Module):
             print("Verified: It is a torch.nn.Module")
        else:
             print(f"Warning: Loaded object is {type(model)}")
             
    except Exception as e:
        print(f"Failed to load BTCUSDT_tst: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    check_tst()
