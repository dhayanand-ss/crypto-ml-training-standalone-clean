"""
Model Manager for MLflow and GCS Integration
Handles model registration, versioning, and storage with MLflow and Google Cloud Storage.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from urllib.parse import urlparse

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Model registration will be disabled.")

from .gcs_manager import GCSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model registration, versioning, and storage with MLflow and Google Cloud Storage.
    
    Features:
    - MLflow model registration and versioning
    - GCS storage integration
    - Model retention policies
    - Production stage management
    - Local artifact caching
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        bucket: str = "mlops-new",
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        cache_dir: str = "models_cache"
    ):
        """
        Initialize ModelManager.
        
        Args:
            tracking_uri: MLflow tracking URI (default: from MLFLOW_TRACKING_URI env var)
            bucket: GCS bucket name (default: "mlops-new")
            credentials_path: Path to GCS service account JSON file (optional)
            project_id: GCP project ID (optional)
            cache_dir: Directory for local model cache
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for ModelManager. Install with: pip install mlflow")
        
        if tracking_uri or os.getenv("MLFLOW_TRACKING_URI"):
            try:
                self.client = MlflowClient(tracking_uri)
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow client: {e}. MLflow features will be limited.")
                self.client = None
        else:
            logger.info("MLflow tracking URI not provided. MLflow features will be disabled.")
            self.client = None
        self.bucket = bucket
        self.cache_dir = os.path.join(os.getcwd(), cache_dir)
        
        # Initialize GCS client for cleanup operations
        try:
            self.gcs_manager = GCSManager(
                bucket=bucket,
                credentials_path=credentials_path,
                project_id=project_id
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GCS client: {e}")
            self.gcs_manager = None
    
    # -------------------
    # Model Version Management
    # -------------------
    def list_versions(self, name: str) -> list:
        """Return all versions sorted by creation time ascending."""
        if not self.client:
            return []
        try:
            versions = self.client.search_model_versions(f"name='{name}'")
            versions = sorted(versions, key=lambda v: v.creation_timestamp)
            logger.info(f"Found {len(versions)} versions for model {name}")
            return versions
        except Exception as e:
            logger.error(f"Error listing versions for {name}: {e}")
            return []
    
    def register(self, model_uri: str, name: str):
        """Register a new version from run artifact path."""
        if not self.client:
            logger.warning("MLflow client not initialized. Cannot register model.")
            return None
        try:
            mv = self.client.create_model_version(name=name, source=model_uri, run_id=None)
            logger.info(f"Registered {name} v{mv.version}")
            return mv
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")
            raise
    
    def enforce_retention(
        self,
        name: str,
        max_versions: int = 10,
        delete_gcs: bool = True
    ):
        """
        Keep version 1 and the last (max_versions - 1). Delete older ones.
        
        Args:
            name: Model name
            max_versions: Maximum number of versions to keep
            delete_gcs: Whether to delete GCS artifacts
        """
        versions = self.list_versions(name)
        if len(versions) <= max_versions:
            logger.info(f"Only {len(versions)} versions found, no pruning needed")
            return
        
        logger.info("Initial version stages before pruning:")
        for v in versions:
            logger.info(f"Version {v.version}: stage={v.current_stage}")
        
        # Versions to keep: v1 and last max_versions-1
        keep_versions = ['1']
        for v in versions[-(max_versions - 1):]:
            keep_versions.append(v.version)
        
        for v in versions:
            if v.version not in keep_versions:
                logger.info(f"Pruning {name} v{v.version} -- {v.source}")
                path = v.source.split("models:/")[-1]
                
                if delete_gcs and self.gcs_manager:
                    # Find and delete GCS artifacts
                    try:
                        available = self.gcs_manager.list_objects(prefix="ml-flow/")
                        
                        for obj in available:
                            if path in obj:
                                obj_prefix = obj.split('/artifacts/')[0]
                                gcs_uri = f"gs://{self.bucket}/{obj_prefix}"
                                logger.info(f"Deleting GCS prefix: {gcs_uri}")
                                self.gcs_manager._delete_gcs_prefix(gcs_uri)
                                break
                    except Exception as e:
                        logger.warning(f"Failed to delete GCS artifacts: {e}")
                
                try:
                    if self.client:
                        self.client.delete_model_version(name, str(v.version))
                        logger.info(f"Deleted model version {name} v{v.version}")
                except Exception as e:
                    logger.error(f"Failed to delete model version: {e}")
        
        versions = self.list_versions(name)
        logger.info("Final version stages after pruning:")
        for v in versions:
            logger.info(f"Version {v.version}: stage={v.current_stage}")
    
    def set_production(self, name: str, keep_latest_n: int = 2) -> list:
        """
        Mark v1 and last N as Production, rest as Archived.
        
        Args:
            name: Model name
            keep_latest_n: Number of latest versions to mark as Production
            
        Returns:
            List of production versions
        """
        versions = self.list_versions(name)
        if not versions:
            logger.warning(f"No versions found for {name}")
            return []
        
        versions = sorted(versions, key=lambda v: int(v.version))
        
        logger.info("Current version stages:")
        for v in versions:
            logger.info(f"Version {v.version}: stage={v.current_stage}")
        
        # Production = version 1 + last keep_latest_n
        prod_versions = ['1']
        for v in versions[-keep_latest_n:]:
            prod_versions.append(v.version)
        
        logger.info(f"Setting production versions for {name}: {prod_versions}")
        
        result = []
        for v in versions:
            if v.version in prod_versions:
                stage = "Production"
            else:
                stage = "Archived"
            
            if v.current_stage != stage and self.client:
                logger.info(f"Transitioning {name} v{v.version} to {stage}")
                try:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=str(v.version),
                        stage=stage,
                        archive_existing_versions=False,
                    )
                except Exception as e:
                    logger.error(f"Failed to transition version {v.version}: {e}")
            
            if v.version in prod_versions:
                result.append(v)
        
        logger.info(f"Set {name} production versions: {[v.version for v in result]}")
        
        versions = self.list_versions(name)
        versions = sorted(versions, key=lambda v: int(v.version))
        production_versions = [v for v in versions if v.current_stage == "Production"]
        
        logger.info("Final version stages:")
        for v in versions:
            logger.info(f"Version {v.version}: stage={v.current_stage}")
        
        return production_versions
    
    # -------------------
    # Model Saving
    # -------------------
    def save_model(
        self,
        model_type: str,
        model: Any,
        name: str,
        onnx_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Save model artifacts during training.
        
        Args:
            model_type: Type of model ('lightgbm', 'tst', 'trl')
            model: Model object
            name: Model name for registration
            onnx_model: Optional ONNX model
            tokenizer: Optional tokenizer (for TRL models)
            
        Returns:
            Registered model version
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for save_model")
        
        artifact_path = name
        
        # Save original model based on type
        if model_type.lower() == "lightgbm":
            import mlflow.lightgbm
            mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
        elif model_type.lower() == "tst":
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, artifact_path=artifact_path)
        elif model_type.lower() == "trl":
            import mlflow.transformers
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model,
                    "tokenizer": tokenizer,
                },
                artifact_path=artifact_path
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Save ONNX if provided
        if onnx_model is not None:
            try:
                import onnx
                onnx_path = f"{name}_tmp.onnx"
                onnx.save(onnx_model, onnx_path)
                mlflow.log_artifact(onnx_path, artifact_path=f"{artifact_path}/onnx")
                os.remove(onnx_path)
                logger.info(f"Saved ONNX model as artifact for {name}")
            except Exception as e:
                logger.warning(f"Failed to save ONNX model: {e}")
        
        # Register model
        try:
            if not self.client:
                logger.warning("MLflow client not initialized. Skipping model registration.")
                return None

            result = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}",
                name=name
            )
            logger.info(f"Registered {name} v{result.version}")
            return result
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    # -------------------
    # Model Loading & Caching
    # -------------------
    def _resolve_model_path(self, name: str, version: Union[str, int]) -> str:
        """
        Resolve local path for model artifacts, using cache if available.
        Downloads from MLflow/GCS if not cached.
        """
        version_str = str(version)
        cache_path = os.path.join(self.cache_dir, name, version_str)
        
        # Check if valid cache exists
        if os.path.exists(cache_path) and os.listdir(cache_path):
            logger.info(f"Using cached artifacts from: {cache_path}")
            return cache_path
            
        logger.info(f"Artifacts not cached for {name} v{version}. Downloading...")
        
        try:
            # Get model version info to find source
            version_obj = self.client.get_model_version(name, version_str)
            run_id = version_obj.run_id
            
            import mlflow
            downloaded_path = None
            
            # 1. Try downloading from source URI (GCS)
            try:
                logger.info(f"Downloading from source URI: {version_obj.source}")
                downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=version_obj.source)
            except Exception as e:
                logger.warning(f"Download from source failed: {e}. Trying run_id...")
                # 2. Fallback to run_id
                downloaded_path = self.client.download_artifacts(run_id, name)
            
            if not downloaded_path:
                raise FileNotFoundError("Failed to download artifacts")
                
            logger.info(f"Downloaded raw artifacts to: {downloaded_path}")
            
            # Find actual model root (sometimes nested)
            actual_model_path = downloaded_path
            if os.path.basename(downloaded_path) != name:
                for root, dirs, files in os.walk(downloaded_path):
                    if os.path.basename(root) == name:
                        actual_model_path = root
                        break
                else:
                    potential = os.path.join(downloaded_path, name)
                    if os.path.exists(potential):
                        actual_model_path = potential
            
            # Move/Copy to cache
            import shutil
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
            
            shutil.copytree(actual_model_path, cache_path)
            logger.info(f"Cached artifacts to: {cache_path}")
            return cache_path
            
        except Exception as e:
            logger.error(f"Failed to resolve/cache model path: {e}")
            raise

    def load_model(
        self,
        name: str,
        version: Union[str, int],
        model_type: str = "pytorch"
    ) -> Tuple[Any, str]:
        """
        Load a specific version of a registered model from MLflow (using local cache).
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for load_model")
        
        import mlflow
        
        # Resolve local path (checks cache or downloads)
        local_path = self._resolve_model_path(name, version)
        # URI for local loading is just the path
        # Note: mlflow.pytorch.load_model works with local paths
        from pathlib import Path
        model_uri = Path(local_path).as_uri()
        
        logger.info(f"Loading {model_type} model from local path: {model_uri}")
        
        if model_type.lower() == "pytorch":
            import mlflow.pytorch
            model = mlflow.pytorch.load_model(model_uri)
        elif model_type.lower() == "lightgbm":
            import mlflow.lightgbm
            model = mlflow.lightgbm.load_model(model_uri)
        elif model_type.lower() == "onnx":
            import mlflow.onnx
            import onnx
            path = mlflow.onnx.load_model(model_uri)
            model = onnx.load(path)
        elif model_type.lower() == "trl":
            import mlflow.transformers
            model_dict = mlflow.transformers.load_model(model_uri, return_type='components')
            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]
            return (model, tokenizer), version
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        return model, version
    
    def load_latest_model(self, name: str, model_type: str = "pytorch") -> Tuple[Any, Optional[str]]:
        """
        Load the latest version of a registered model from MLflow.
        
        Args:
            name: Registered model name
            model_type: Model type ('pytorch', 'lightgbm', 'onnx', 'trl')
            
        Returns:
            Tuple of (model, latest_version) or ((model, tokenizer), latest_version) for TRL
        """
        try:
            versions = self.client.get_latest_versions(name)
        except Exception as e:
            logger.error(f"Error fetching versions for model {name}: {e}")
            return None, None
        
        if not versions:
            logger.warning(f"No versions found for {name}")
            return None, None
        
        latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
        
        logger.info(f"All versions: {[v.version for v in versions]}")
        logger.info(f"Latest version: {latest_version.version}, stage: {latest_version.current_stage}")
        logger.info(f"Model source: {latest_version.source}")
        logger.info(f"Run ID: {latest_version.run_id}")
        
        return self.load_model(name, latest_version.version, model_type=model_type)
    
    def load_onnx_model(self, name: str, version: Union[str, int]):
        """
        Load an ONNX model from a registered MLflow model version.
        
        Args:
            name: Registered model name
            version: Model version to load
            
        Returns:
            ONNX Runtime InferenceSession
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required for load_onnx_model")
        
        # Ensure mlflow is available locally to avoid UnboundLocalError
        import mlflow
        
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX model loading")
            
        # Resolve local path via cache
        local_path = self._resolve_model_path(name, version)
        
        # Look for ONNX model in the expected location
        onnx_path = os.path.join(local_path, f'onnx/{name}_tmp.onnx')
        
        # Also try alternative paths if the expected path doesn't exist
        if not os.path.exists(onnx_path):
            alt_paths = [
                os.path.join(local_path, 'onnx', f'{name}_tmp.onnx'),
                os.path.join(local_path, 'onnx', 'model.onnx'),
                os.path.join(local_path, f'{name}_tmp.onnx'),
            ]
            
            onnx_dir = os.path.join(local_path, 'onnx')
            if os.path.exists(onnx_dir):
                for file in os.listdir(onnx_dir):
                    if file.endswith('.onnx'):
                        alt_paths.append(os.path.join(onnx_dir, file))
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    onnx_path = alt_path
                    logger.info(f"Found ONNX model at alternative path: {onnx_path}")
                    break
        
        if not os.path.exists(onnx_path):
             # List what's actually in the directory for debugging
            logger.warning(f"ONNX model not found in {local_path}. Listing contents...")
            try:
                logger.warning(f"Root: {os.listdir(local_path)}")
                if os.path.exists(os.path.join(local_path, 'onnx')):
                    logger.warning(f"ONNX dir: {os.listdir(os.path.join(local_path, 'onnx'))}")
            except:
                pass
                
            raise FileNotFoundError(f"ONNX model not found in cached path")
        
        logger.info(f"ONNX model '{name}' version {version} loaded successfully!")
        ort_session = ort.InferenceSession(onnx_path)
        
        return ort_session
