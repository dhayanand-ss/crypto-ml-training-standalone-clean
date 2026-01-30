"""
FastAPI ML Model Inference API

Serves ML models for predictions with:
- Model management: loads production models from MLflow (ONNX)
- Prometheus metrics: /metrics
- Thread-safe model refresh with locks
- Batch predictions (up to 5000 per request)
"""

import os
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, validator

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
    INSTRUMENTATOR_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    INSTRUMENTATOR_AVAILABLE = False

# Import ModelManager from the project
import sys
from pathlib import Path

# Add parent directory to path to import utils
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.artifact_control.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model cache with thread-safe locks
_model_cache: Dict[str, Any] = {}
_model_locks: Dict[str, threading.RLock] = {}
_cache_lock = threading.RLock()

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    prediction_counter = Counter(
        'ml_predictions_total',
        'Total number of predictions made',
        ['model_name', 'version', 'status']
    )
    prediction_latency = Histogram(
        'ml_predictions_duration_seconds',
        'Prediction latency in seconds',
        ['model_name', 'version']
    )
    model_load_errors = Counter(
        'ml_model_load_errors_total',
        'Total number of model load errors',
        ['model_name', 'version']
    )
    active_models = Gauge(
        'ml_active_models',
        'Number of active models loaded in memory',
        ['model_name', 'version']
    )
else:
    prediction_counter = None
    prediction_latency = None
    model_load_errors = None
    active_models = None

# Check MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Initialize ModelManager
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
model_manager = ModelManager(tracking_uri=mlflow_tracking_uri)


# Pydantic models for request/response
class RefreshRequest(BaseModel):
    """Request model for refreshing models"""
    model_name: Optional[str] = None  # If None, refresh all production models
    version: Optional[Union[str, int]] = None  # If None, load latest production version


class ModelAvailabilityRequest(BaseModel):
    """Request model for checking model availability"""
    model_name: str
    version: Union[str, int]


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[List[float]] = Field(..., description="Feature matrix for predictions")
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features list cannot be empty")
        if len(v) > 5000:
            raise ValueError("Maximum 5000 samples per request")
        # Validate all rows have same length
        if len(set(len(row) for row in v)) > 1:
            raise ValueError("All feature vectors must have the same length")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[List[float]] = Field(..., description="Prediction probabilities")
    model_name: str
    version: str
    num_samples: int


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    version: str
    loaded: bool
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None


def get_model_key(model_name: str, version: Union[str, int]) -> str:
    """Generate a unique key for model cache"""
    return f"{model_name}:{version}"


def load_production_models() -> Dict[str, Any]:
    """
    Load all production models from MLflow.
    Returns a dictionary of model_key -> ONNX InferenceSession
    """
    loaded_models = {}
    
    try:
        # Get all registered models
        registered_models = model_manager.client.search_registered_models()
        logger.info(f"Found {len(registered_models)} registered model(s) in MLflow")
        
        if not registered_models:
            logger.warning("No registered models found in MLflow. Make sure models are registered.")
            return loaded_models
        
        # Ensure specific important models are checked even if search misses them
        known_models = ["BTCUSDT_lightgbm", "BTCUSDT_tst"]
        found_names = {m.name for m in registered_models}
        
        for name in known_models:
            if name not in found_names:
                try:
                    logger.info(f"Explicitly checking for known model: {name}")
                    m = model_manager.client.get_registered_model(name)
                    registered_models.append(m)
                except Exception as e:
                    logger.warning(f"Known model {name} not found: {e}")

        total_production_versions = 0
        for model in registered_models:
            model_name = model.name
            logger.info(f"Checking model: {model_name}")
            
            # Get production versions
            try:
                versions = model_manager.client.get_latest_versions(
                    model_name,
                    stages=["Production"]
                )
                logger.info(f"Found {len(versions)} Production version(s) for {model_name}")
                
                if not versions:
                    logger.warning(f"No Production versions found for {model_name}. Use MLflow UI to transition models to Production stage.")
                    # Also check what stages exist
                    all_versions = model_manager.client.search_model_versions(f"name='{model_name}'")
                    stages = [v.current_stage or "None" for v in all_versions]
                    logger.info(f"Available stages for {model_name}: {set(stages)}")
                    continue
                
                total_production_versions += len(versions)
                
                for version_obj in versions:
                    version = version_obj.version
                    model_key = get_model_key(model_name, version)
                    
                    try:
                        logger.info(f"Loading production model: {model_name} version {version}")
                        
                        # Try ONNX first (as it's the standard for this server)
                        try:
                            logger.info(f"Attempting to load as ONNX model: {model_name}")
                            ort_session = model_manager.load_onnx_model(model_name, version)
                            
                            # Get input/output shapes
                            input_shape = None
                            output_shape = None
                            if ort_session.get_inputs():
                                input_shape = list(ort_session.get_inputs()[0].shape)
                            if ort_session.get_outputs():
                                output_shape = list(ort_session.get_outputs()[0].shape)
                            
                            loaded_models[model_key] = {
                                'session': ort_session,
                                'type': 'onnx',
                                'model_name': model_name,
                                'version': version,
                                'input_shape': input_shape,
                                'output_shape': output_shape,
                                'loaded_at': time.time()
                            }
                        except Exception as onnx_err:
                            logger.warning(f"Could not load {model_name} as ONNX: {onnx_err}. Trying fallback types...")
                            
                            # Fallback: Detect type or try others
                            model_type = "pytorch" # Default fallback
                            if "lightgbm" in model_name.lower():
                                model_type = "lightgbm"
                            
                            logger.info(f"Attempting to load as {model_type} model: {model_name}")
                            model, loaded_version = model_manager.load_model(model_name, version, model_type=model_type)
                            
                            loaded_models[model_key] = {
                                'model': model,
                                'type': model_type,
                                'model_name': model_name,
                                'version': version,
                                'loaded_at': time.time()
                            }
                        
                        logger.info(f"Successfully loaded {model_key}")
                        
                        if active_models:
                            active_models.labels(model_name=model_name, version=str(version)).set(1)
                            
                    except Exception as e:
                        logger.error(f"Failed to load {model_key}: {e}", exc_info=True)
                        if model_load_errors:
                            model_load_errors.labels(model_name=model_name, version=str(version)).inc()
                        
            except Exception as e:
                logger.error(f"Error getting Production versions for {model_name}: {e}", exc_info=True)
        
        logger.info(f"Total Production versions found: {total_production_versions}, Successfully loaded: {len(loaded_models)}")
        
        if total_production_versions > 0 and len(loaded_models) == 0:
            logger.error("Production models exist but none could be loaded. Check ONNX model availability and logs above.")
                    
    except Exception as e:
        logger.error(f"Error loading production models: {e}", exc_info=True)
    
    return loaded_models


def refresh_models(model_name: Optional[str] = None, version: Optional[Union[str, int]] = None):
    """
    Thread-safe model refresh.
    If model_name is None, refresh all production models.
    If version is None, load latest production version.
    """
    global _model_cache
    
    with _cache_lock:
        if model_name is None:
            # Refresh all production models
            logger.info("Refreshing all production models")
            new_cache = load_production_models()
            
            # Update locks for new models
            for model_key in new_cache:
                if model_key not in _model_locks:
                    _model_locks[model_key] = threading.RLock()
            
            # Remove old models from cache and metrics
            old_keys = set(_model_cache.keys()) - set(new_cache.keys())
            for old_key in old_keys:
                old_model = _model_cache[old_key]
                if active_models:
                    active_models.labels(
                        model_name=old_model['model_name'],
                        version=str(old_model['version'])
                    ).set(0)
                del _model_locks[old_key]
            
            _model_cache = new_cache
            logger.info(f"Refreshed {len(_model_cache)} models")
            
        else:
            # Refresh specific model
            if version is None:
                # Load latest production version
                try:
                    versions = model_manager.client.get_latest_versions(
                        model_name,
                        stages=["Production"]
                    )
                    if not versions:
                        raise ValueError(f"No production versions found for {model_name}")
                    version = versions[0].version
                except Exception as e:
                    logger.error(f"Failed to get latest version for {model_name}: {e}")
                    raise
            
            model_key = get_model_key(model_name, version)
            
            # Get lock for this specific model
            if model_key not in _model_locks:
                _model_locks[model_key] = threading.RLock()
            
            with _model_locks[model_key]:
                try:
                    logger.info(f"Refreshing model: {model_key}")
                    
                    # Try ONNX first
                    try:
                        ort_session = model_manager.load_onnx_model(model_name, version)
                        
                        # Get input/output shapes
                        input_shape = None
                        output_shape = None
                        if ort_session.get_inputs():
                            input_shape = list(ort_session.get_inputs()[0].shape)
                        if ort_session.get_outputs():
                            output_shape = list(ort_session.get_outputs()[0].shape)
                        
                        _model_cache[model_key] = {
                            'session': ort_session,
                            'type': 'onnx',
                            'model_name': model_name,
                            'version': version,
                            'input_shape': input_shape,
                            'output_shape': output_shape,
                            'loaded_at': time.time()
                        }
                    except Exception as onnx_err:
                        logger.warning(f"Could not load {model_name} as ONNX during refresh: {onnx_err}. Trying fallback types...")
                        
                        # Fallback: Detect type or try others
                        model_type = "pytorch" # Default fallback
                        if "lightgbm" in model_name.lower():
                            model_type = "lightgbm"
                        
                        logger.info(f"Attempting to load as {model_type} model: {model_name}")
                        model, loaded_version = model_manager.load_model(model_name, version, model_type=model_type)
                        
                        _model_cache[model_key] = {
                            'model': model,
                            'type': model_type,
                            'model_name': model_name,
                            'version': version,
                            'loaded_at': time.time()
                        }
                    
                    logger.info(f"Successfully refreshed {model_key}")
                    
                    if active_models:
                        active_models.labels(model_name=model_name, version=str(version)).set(1)
                        
                except Exception as e:
                    logger.error(f"Failed to refresh {model_key}: {e}")
                    if model_load_errors:
                        model_load_errors.labels(model_name=model_name, version=str(version)).inc()
                    raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup: Load production models
    logger.info("Starting FastAPI ML Inference Service")
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    
    try:
        refresh_models()  # Load all production models
        logger.info(f"Loaded {len(_model_cache)} production models on startup")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down FastAPI ML Inference Service")
    _model_cache.clear()
    _model_locks.clear()


# Create FastAPI app
app = FastAPI(
    title="ML Model Inference API",
    description="FastAPI service for ML model predictions using ONNX models from MLflow",
    version="1.0.0",
    lifespan=lifespan
)

# Prometheus instrumentation (automatic HTTP metrics)
# This automatically exposes /metrics endpoint with HTTP request metrics
if INSTRUMENTATOR_AVAILABLE:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
async def root():
    """Root endpoint - redirects to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(_model_cache),
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "prometheus_available": PROMETHEUS_AVAILABLE
    }


# Custom metrics endpoint (if instrumentator is not available, fallback to manual)
if not INSTRUMENTATOR_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint (fallback)"""
        if not PROMETHEUS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Prometheus client not available")
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


@app.post("/refresh", response_model=Dict[str, Any])
async def refresh(request: RefreshRequest):
    """
    Reload production models from MLflow.
    
    - If model_name is None: refresh all production models
    - If model_name is provided but version is None: load latest production version
    - If both are provided: load specific model version
    
    Returns models in format: {"model_name": [(version, version), ...]}
    """
    try:
        refresh_models(request.model_name, request.version)
        
        # Build response in documented format
        with _cache_lock:
            if request.model_name:
                # Single model refresh - find the loaded version(s) for this model
                model_versions = []
                for model_key, model_info in _model_cache.items():
                    if model_info['model_name'] == request.model_name:
                        version = int(model_info['version'])
                        model_versions.append((version, version))
                
                if model_versions:
                    # Sort and deduplicate
                    model_versions = sorted(set(model_versions))
                    return {
                        "status": "models loaded",
                        "models": {
                            request.model_name: model_versions
                        }
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Model {request.model_name} not found after refresh"
                    }
            else:
                # All models refresh - group by model name
                models_dict = {}
                for model_key, model_info in _model_cache.items():
                    model_name = model_info['model_name']
                    version = int(model_info['version'])
                    if model_name not in models_dict:
                        models_dict[model_name] = []
                    models_dict[model_name].append((version, version))
                
                # Sort versions for each model
                for model_name in models_dict:
                    models_dict[model_name] = sorted(set(models_dict[model_name]))
                
                return {
                    "status": "models loaded",
                    "models": models_dict
                }
            
    except Exception as e:
        logger.error(f"Error refreshing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/is_model_available", response_model=Dict[str, Any])
async def is_model_available(request: ModelAvailabilityRequest):
    """
    Check if a specific model version is available.
    
    Note: version should be integer (0-indexed, where 0 = v1, 1 = v2, etc.)
    but also accepts MLflow version strings directly.
    """
    # Convert 0-indexed version to MLflow version if needed
    if isinstance(request.version, int):
        mlflow_version = str(request.version + 1)  # 0 -> v1, 1 -> v2, etc.
    else:
        mlflow_version = str(request.version)
    
    model_key = get_model_key(request.model_name, mlflow_version)
    
    with _cache_lock:
        if model_key in _model_cache:
            return {
                "available": True
            }
        else:
            # Check if model exists in MLflow
            try:
                version_obj = model_manager.client.get_model_version(
                    request.model_name,
                    mlflow_version
                )
                return {
                    "available": True
                }
            except Exception as e:
                return {
                    "available": False
                }


@app.post("/predict", response_model=Dict[str, Any])
async def predict(
    features: List[List[float]],
    model_name: str = Query(..., description="Model name"),
    version: Union[str, int] = Query(..., description="Model version (0-indexed: 0=v1, 1=v2, etc.)")
):
    """
    Make batch predictions using the specified model.
    
    - Maximum 5000 samples per request
    - Features should be a 2D array (list of feature vectors) in JSON body
    - model_name and version are query parameters
    - Returns prediction probabilities
    
    Request body: [[0.1, 0.2, ...], [0.5, 0.6, ...]]
    """
    if not ONNXRUNTIME_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ONNX Runtime not available. Please install onnxruntime."
        )
    
    # Validate input
    if not features:
        raise HTTPException(status_code=400, detail="Features list cannot be empty")
    
    if len(features) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5000 samples per request"
        )
    
    # Check feature consistency
    if len(set(len(row) for row in features)) > 1:
        raise HTTPException(
            status_code=400,
            detail="All feature vectors must have the same length"
        )
    
    # Convert 0-indexed version to MLflow version if needed
    if isinstance(version, int):
        mlflow_version = str(version + 1)  # 0 -> v1, 1 -> v2, etc.
    else:
        mlflow_version = str(version)
    
    model_key = get_model_key(model_name, mlflow_version)
    
    # Check if model is in cache (thread-safe)
    with _cache_lock:
        model_in_cache = model_key in _model_cache
        if model_in_cache:
            model_info = _model_cache[model_key]
            model_lock = _model_locks.get(model_key)
        else:
            model_info = None
            model_lock = None
    
    # If not in cache, try to load it (outside the lock to avoid deadlock)
    if not model_in_cache:
        try:
            logger.info(f"Model {model_key} not in cache, loading...")
            refresh_models(model_name, mlflow_version)
            # Re-acquire lock to get the loaded model
            with _cache_lock:
                if model_key in _model_cache:
                    model_info = _model_cache[model_key]
                    model_lock = _model_locks.get(model_key)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {model_name} version {mlflow_version} failed to load"
                    )
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} version {mlflow_version} not found or failed to load: {str(e)}"
            )
    
    # Make predictions (thread-safe per model)
    if model_lock:
        with model_lock:
            return _make_predictions(model_info, features, model_name, mlflow_version)
    else:
        return _make_predictions(model_info, features, model_name, mlflow_version)


def _make_predictions(
    model_info: Dict[str, Any],
    features: List[List[float]],
    model_name: str,
    version: str
) -> Dict[str, Any]:
    """Internal function to make predictions"""
    start_time = time.time()
    
    try:

        model_type = model_info.get('type', 'onnx')
        
        # Convert features to numpy array
        features_array = np.array(features, dtype=np.float32)

        if model_type == 'onnx':
            session = model_info['session']
            # Get input name from model
            input_name = session.get_inputs()[0].name
            # Run inference
            outputs = session.run(None, {input_name: features_array})
            predictions = outputs[0].tolist()
            
        elif model_type == 'lightgbm':
            model = model_info['model']
            # LightGBM predict returns array
            preds = model.predict(features_array)
            # Ensure it's list of lists or list of values
            if isinstance(preds, np.ndarray):
                predictions = preds.tolist()
            else:
                predictions = list(preds)
            # If 1D array (regression/binary), make it 2D like ONNX usually is for consistency if needed
            # But standard is fine.
            
        elif model_type == 'pytorch':
            import torch
            model = model_info['model']
            device = next(model.parameters()).device
            
            with torch.no_grad():
                inputs = torch.tensor(features_array, dtype=torch.float32).to(device)
                outputs = model(inputs)
                # If tuple (TST), get first
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                predictions = outputs.cpu().numpy().tolist()
        
        else:
             raise ValueError(f"Unsupported model type for prediction: {model_type}")

        # Common post-processing (logging, metrics) happens below

        

        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        if prediction_counter:
            prediction_counter.labels(
                model_name=model_name,
                version=str(version),
                status="success"
            ).inc()
        
        if prediction_latency:
            prediction_latency.labels(
                model_name=model_name,
                version=str(version)
            ).observe(latency)
        
        logger.info(
            f"Made predictions for {len(features)} samples using {model_name} v{version} "
            f"(latency: {latency:.3f}s)"
        )
        
        # Return in documented format
        return {
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction error for {model_name} v{version}: {e}")
        
        # Update error metrics
        if prediction_counter:
            prediction_counter.labels(
                model_name=model_name,
                version=str(version),
                status="error"
            ).inc()
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all loaded models"""
    with _cache_lock:
        models = []
        for model_key, model_info in _model_cache.items():
            models.append(ModelInfo(
                model_name=model_info['model_name'],
                version=str(model_info['version']),
                loaded=True,
                input_shape=model_info.get('input_shape'),
                output_shape=model_info.get('output_shape')
            ))
        return models


@app.get("/debug/mlflow")
async def debug_mlflow():
    """
    Debug endpoint to check MLflow connection and available models.
    Helps diagnose why no models are loaded.
    
    Shows ALL registered models and their stages, not just Production ones.
    """
    debug_info = {
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_available": MLFLOW_AVAILABLE,
        "connection_status": "unknown",
        "registered_models": [],
        "all_model_versions": [],
        "production_models": [],
        "loaded_models_count": len(_model_cache),
        "errors": []
    }
    
    if not MLFLOW_AVAILABLE:
        debug_info["errors"].append("MLflow is not available. Install with: pip install mlflow")
        return debug_info
    
    try:
        # Test connection
        registered_models = model_manager.client.search_registered_models()
        debug_info["connection_status"] = "connected"
        debug_info["registered_models"] = [{"name": m.name, "latest_versions": len(m.latest_versions)} for m in registered_models]
        
        # Get ALL model versions (not just Production)
        for model in registered_models:
            model_name = model.name
            try:
                # Get all versions
                all_versions = model_manager.client.search_model_versions(f"name='{model_name}'")
                
                for v in all_versions:
                    version_info = {
                        "name": model_name,
                        "version": v.version,
                        "stage": v.current_stage or "None",
                        "source": v.source,
                        "onnx_available": None,
                        "onnx_error": None
                    }
                    
                    # Check ONNX availability
                    try:
                        model_manager.load_onnx_model(model_name, v.version)
                        version_info["onnx_available"] = True
                    except Exception as e:
                        version_info["onnx_available"] = False
                        version_info["onnx_error"] = str(e)
                    
                    debug_info["all_model_versions"].append(version_info)
                    
                    # Track production models separately
                    if v.current_stage == "Production":
                        debug_info["production_models"].append(version_info)
                        
            except Exception as e:
                debug_info["errors"].append(f"Error checking {model_name}: {str(e)}")
        
    except Exception as e:
        debug_info["connection_status"] = "failed"
        debug_info["errors"].append(f"MLflow connection error: {str(e)}")
    
    return debug_info


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

