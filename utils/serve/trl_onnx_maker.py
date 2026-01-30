#!/usr/bin/env python3
"""
TRL ONNX Model Converter
Converts the TRL (FinBERT) model to ONNX format for optimized inference.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_finbert_to_onnx(model_path=None, output_path=None):
    """
    Convert FinBERT model to ONNX format.
    
    Args:
        model_path: Path to the trained FinBERT model (optional)
        output_path: Path to save the ONNX model (optional)
    """
    try:
        from models.finbert_sentiment import FinBERTSentimentAnalyzer
        import torch
    except ImportError as e:
        logger.error(f"Required imports not available: {e}")
        logger.error("Make sure transformers and torch are installed")
        return False
    
    try:
        import onnx
        import onnxruntime
        from transformers import convert_graph_to_onnx
        ONNX_AVAILABLE = True
    except ImportError:
        logger.warning("ONNX conversion tools not fully available")
        logger.info("For full ONNX support, install: pip install onnx onnxruntime transformers[onnx]")
        ONNX_AVAILABLE = False
    
    logger.info("=" * 60)
    logger.info("TRL ONNX Model Converter")
    logger.info("=" * 60)
    
    # Initialize analyzer to get the model
    logger.info("Instantiating FinBERTSentimentAnalyzer...")
    try:
        analyzer = FinBERTSentimentAnalyzer()
        logger.info("FinBERTSentimentAnalyzer instantiated.")
        model = analyzer.model
        tokenizer = analyzer.tokenizer
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check if model is a PEFT model and merge adapters if needed
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            logger.info("Detected PEFT model. Merging adapters with base model for ONNX export...")
            # Merge PEFT adapters with base model
            model = model.merge_and_unload()
            logger.info("PEFT adapters merged successfully")
    except ImportError:
        # PEFT not available, assume model is not a PEFT model
        logger.info("PEFT library not available, assuming standard model")
    except Exception as e:
        logger.warning(f"Could not merge PEFT adapters: {e}. Attempting direct export...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Default paths
    if output_path is None:
        output_dir = "models/onnx"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "finbert_trl.onnx")
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Converting model to ONNX format...")
    logger.info(f"Output path: {output_path}")
    
    try:
        # Create dummy input for tracing
        # FinBERT expects text input, but ONNX needs tensor input
        # We'll use the tokenizer to create a sample input
        dummy_text = "This is a sample financial news article for ONNX conversion."
        dummy_input = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move model and inputs to CPU for ONNX export (ONNX export requires CPU)
        device = next(model.parameters()).device
        logger.info(f"Model device: {device}")
        if device.type != 'cpu':
            logger.info("Moving model to CPU for ONNX export...")
            model = model.cpu()
        
        # Ensure inputs are on CPU
        input_ids = dummy_input['input_ids'].cpu()
        attention_mask = dummy_input['attention_mask'].cpu()
        
        # Export to ONNX
        # Note: This is a simplified conversion - for production, you may need
        # to handle the full pipeline including tokenization
        logger.info("Exporting model to ONNX...")
        
        # Create a wrapper function for ONNX export since transformers models
        # use keyword arguments in forward() and return objects with .logits attribute
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Extract logits from the output (handles both dict and object returns)
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                elif isinstance(outputs, dict):
                    return outputs['logits']
                else:
                    return outputs
        
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        # Use torch.onnx.export for PyTorch models
        torch.onnx.export(
            wrapped_model,
            (input_ids, attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info(f"Model successfully converted to ONNX: {output_path}")
        
        # Verify ONNX model
        if ONNX_AVAILABLE:
            try:
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verification passed")
            except Exception as e:
                logger.warning(f"ONNX model verification warning: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting model to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("TRL ONNX Maker")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/finbert/finbert_grpo.pth"
    if not os.path.exists(model_path):
        logger.warning(f"Trained model not found at {model_path}")
        logger.info("Using base FinBERT model for conversion")
        logger.info("Note: For best results, train the model first")
    
    # Convert to ONNX
    success = convert_finbert_to_onnx()
    
    if success:
        logger.info("=" * 60)
        logger.info("ONNX conversion completed successfully!")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("ONNX conversion failed!")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())

