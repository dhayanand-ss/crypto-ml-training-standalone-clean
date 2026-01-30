"""
GCS Manager for Crypto ML Training
Handles Google Cloud Storage for models, data, and predictions with versioning support.
"""

try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None
    NotFound = Exception

import pandas as pd
from io import BytesIO
import hashlib
import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCSManager:
    """
    Manages Google Cloud Storage operations for the crypto ML training project.
    
    Supports:
    - Model storage with versioning (v1, v2, v3)
    - Data storage (prices, articles, predictions)
    - Hash-based deduplication
    - MLflow integration
    """
    
    def __init__(
        self,
        bucket: str = "mlops-new",
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize GCSManager.
        
        Args:
            bucket: GCS bucket name (default: "mlops-new")
            credentials_path: Path to GCS service account JSON file (from GOOGLE_APPLICATION_CREDENTIALS env var if not provided)
            project_id: GCP project ID (optional, can be inferred from credentials)
        """
        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required for GCSManager. "
                "Install with: pip install google-cloud-storage"
            )
        
        self.bucket_name = bucket
        self.credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        try:
            # Initialize GCS client
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.client = storage.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project_id
                )
            else:
                # Try default credentials (e.g., from environment or gcloud auth)
                self.client = storage.Client(project=self.project_id)
            
            # Get bucket
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"GCSManager initialized with bucket: {bucket}")
        except Exception as e:
            logger.warning(f"GCS client initialization failed: {e}. GCS operations will be disabled.")
            self.client = None
            self.bucket = None
    
    # -------------------
    # Hash helpers
    # -------------------
    @staticmethod
    def compute_hash_bytes(data_bytes: bytes) -> str:
        """Compute SHA256 hash of bytes."""
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _get_gcs_hash(self, hash_key: str) -> Optional[str]:
        """Get hash from GCS if it exists."""
        if not self.bucket:
            return None
        try:
            blob = self.bucket.blob(hash_key)
            if blob.exists():
                return blob.download_as_text()
            return None
        except Exception as e:
            logger.debug(f"Hash file {hash_key} not found: {e}")
            return None
    
    # -------------------
    # DataFrame operations
    # -------------------
    def upload_df(
        self,
        path_or_df: Union[str, Path, pd.DataFrame],
        key: str,
        skip_if_exists: bool = True
    ) -> bool:
        """
        Upload DataFrame to GCS as Parquet with hash-based deduplication.
        
        Args:
            path_or_df: Path to CSV file or DataFrame
            key: GCS blob name (e.g., 'prices/BTCUSDT.parquet')
            skip_if_exists: Skip upload if hash matches existing file
            
        Returns:
            True if uploaded, False if skipped
        """
        if not self.bucket:
            logger.warning(f"Skipping GCS upload for {key}: GCS client not initialized.")
            return False

        # Load DataFrame
        if isinstance(path_or_df, pd.DataFrame):
            df = path_or_df
        elif isinstance(path_or_df, (str, Path)):
            df = pd.read_csv(path_or_df)
        else:
            raise ValueError(f"Unsupported type: {type(path_or_df)}")
        
        # Convert to Parquet in memory
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        data_bytes = buffer.getvalue()
        
        # Compute hash
        file_hash = self.compute_hash_bytes(data_bytes)
        hash_key = key.replace('.parquet', '.hash')
        
        # Check if hash exists and matches
        if skip_if_exists:
            existing_hash = self._get_gcs_hash(hash_key)
            if existing_hash == file_hash:
                logger.info(f"Skipping upload for {key}, hash matches GCS.")
                return False
        
        # Upload file and hash
        try:
            # Upload parquet file
            blob = self.bucket.blob(key)
            blob.upload_from_string(data_bytes, content_type='application/octet-stream')
            
            # Upload hash file
            hash_blob = self.bucket.blob(hash_key)
            hash_blob.upload_from_string(file_hash, content_type='text/plain')
            
            logger.info(f"Uploaded {key} with hash {file_hash[:8]}... to GCS.")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {key}: {e}")
            return False
    
    def download_df(
        self,
        local_file: str,
        key: str,
        skip_if_exists: bool = True
    ) -> pd.DataFrame:
        """
        Download DataFrame from GCS and save locally as CSV.
        
        Args:
            local_file: Local file path (will be saved as .csv)
            key: GCS blob name (e.g., 'prices/BTCUSDT.parquet')
            skip_if_exists: Skip download if local file exists
            
        Returns:
            Downloaded DataFrame
        """
        # Check if local file exists
        csv_path = local_file.replace('.parquet', '.csv')
        if skip_if_exists and os.path.exists(csv_path):
            logger.info(f"Local file {csv_path} already exists. Loading from disk.")
            return pd.read_csv(csv_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        
        # Download from GCS
        try:
            if not self.bucket:
                 logger.warning(f"GCS client not initialized. Cannot download {key}.")
                 if os.path.exists(csv_path):
                     return pd.read_csv(csv_path)
                 raise FileNotFoundError(f"GCS not available and local file {csv_path} missing.")

            blob = self.bucket.blob(key)
            if not blob.exists():
                raise FileNotFoundError(f"Blob {key} not found in bucket {self.bucket_name}")
            
            # Download to buffer
            buffer = BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)
            
            # Convert to DataFrame
            df = pd.read_parquet(buffer)
            
            # Save as CSV
            df.to_csv(csv_path, index=False)
            
            # Set permissions (for Airflow compatibility)
            try:
                os.chmod(csv_path, 0o777)
                os.chmod(os.path.dirname(csv_path), 0o777)
            except Exception:
                pass  # Ignore permission errors on Windows
            
            logger.info(f"Downloaded {key} to {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            raise
    
    # -------------------
    # Model operations (with versioning)
    # -------------------
    def upload_model(
        self,
        model_path: Union[str, Path],
        model_type: str,
        version: str,
        additional_files: Optional[List[str]] = None
    ) -> bool:
        """
        Upload model files to GCS with versioning.
        
        Args:
            model_path: Path to model file
            model_type: Type of model ('lightgbm', 'tst', 'finbert', 'ensemble')
            version: Version ('v1', 'v2', 'v3')
            additional_files: List of additional files to upload (e.g., scaler, features)
            
        Returns:
            True if uploaded successfully
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Upload main model file
        if not self.bucket:
            logger.warning(f"GCS client not initialized. Skipping model upload for {model_path.name}.")
            return False

        gcs_key = f"models/{model_type}/{version}/{model_path.name}"
        try:
            blob = self.bucket.blob(gcs_key)
            blob.upload_from_filename(str(model_path))
            logger.info(f"Uploaded {model_path.name} to gs://{self.bucket_name}/{gcs_key}")
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise
        
        # Upload additional files (scaler, features, tokenizer, etc.)
        if additional_files:
            for file_path in additional_files:
                file_path = Path(file_path)
                if file_path.exists():
                    file_gcs_key = f"models/{model_type}/{version}/{file_path.name}"
                    try:
                        file_blob = self.bucket.blob(file_gcs_key)
                        file_blob.upload_from_filename(str(file_path))
                        logger.info(f"Uploaded {file_path.name} to gs://{self.bucket_name}/{file_gcs_key}")
                    except Exception as e:
                        logger.warning(f"Failed to upload {file_path.name}: {e}")
        
        return True
    
    def download_model(
        self,
        local_dir: Union[str, Path],
        model_type: str,
        version: str,
        model_filename: str
    ) -> Path:
        """
        Download model files from GCS.
        
        Args:
            local_dir: Local directory to save model
            model_type: Type of model ('lightgbm', 'tst', 'finbert', 'ensemble')
            version: Version ('v1', 'v2', 'v3')
            model_filename: Name of model file (e.g., 'lgb_model.txt', 'tst_model.pth')
            
        Returns:
            Path to downloaded model file
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        gcs_key = f"models/{model_type}/{version}/{model_filename}"
        local_path = local_dir / model_filename
        
        try:
            if not self.bucket:
                logger.warning(f"GCS client not initialized. Cannot download model {gcs_key}.")
                if local_path.exists():
                    return local_path
                raise FileNotFoundError("GCS not available and local model missing.")

            blob = self.bucket.blob(gcs_key)
            if not blob.exists():
                raise FileNotFoundError(f"Model {gcs_key} not found in bucket {self.bucket_name}")
            
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded model from gs://{self.bucket_name}/{gcs_key} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def upload_model_directory(
        self,
        local_dir: Union[str, Path],
        model_type: str,
        version: str
    ):
        """
        Upload entire model directory to GCS.
        
        Args:
            local_dir: Local directory containing model files
            model_type: Type of model
            version: Version ('v1', 'v2', 'v3')
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            raise FileNotFoundError(f"Directory not found: {local_dir}")
        
        gcs_prefix = f"models/{model_type}/{version}/"
        
        if not self.bucket:
            logger.warning(f"GCS client not initialized. Skipping directory upload for {local_dir}")
            return

        for file_path in local_dir.iterdir():
            if file_path.is_file():
                gcs_key = f"{gcs_prefix}{file_path.name}"
                try:
                    blob = self.bucket.blob(gcs_key)
                    blob.upload_from_filename(str(file_path))
                    logger.info(f"Uploaded {file_path.name} to gs://{self.bucket_name}/{gcs_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {file_path.name}: {e}")
    
    # -------------------
    # Data operations
    # -------------------
    def upload_price_data(self, coin: str, price_file: Union[str, Path]):
        """Upload price data for a coin."""
        key = f"prices/{coin}.parquet"
        self.upload_df(price_file, key)
    
    def download_price_data(self, coin: str, local_file: str) -> pd.DataFrame:
        """Download price data for a coin."""
        key = f"prices/{coin}.parquet"
        return self.download_df(local_file, key)
    
    def upload_articles(self, articles_file: Union[str, Path]):
        """Upload articles data."""
        key = "articles/articles.parquet"
        self.upload_df(articles_file, key)
    
    def download_articles(self, local_file: str) -> pd.DataFrame:
        """Download articles data."""
        key = "articles/articles.parquet"
        return self.download_df(local_file, key)
    
    def upload_predictions(
        self,
        predictions_df: pd.DataFrame,
        coin: str,
        model_name: str,
        version: Optional[int] = None
    ):
        """
        Upload predictions to GCS.
        
        Args:
            predictions_df: DataFrame with predictions
            coin: Coin symbol (e.g., 'BTCUSDT')
            model_name: Model name (e.g., 'lightgbm', 'tst', 'trl')
            version: Version number (if None, auto-increments)
        """
        if version is None:
            # Get existing versions and increment
            existing = self.get_existing_prediction_versions(coin, model_name)
            version = len(existing) + 1
        
        key = f"predictions/{coin}/{model_name}/v{version}.parquet"
        self.upload_df(predictions_df, key)
        logger.info(f"Uploaded predictions for {coin} {model_name} as v{version}")
    
    def get_existing_prediction_versions(self, coin: str, model_name: str) -> List[str]:
        """Get list of existing prediction versions."""
        prefix = f"predictions/{coin}/{model_name}/"
        try:
            objects = self.list_objects(prefix)
            versions = [obj for obj in objects if obj.endswith('.parquet')]
            versions = sorted(versions, key=lambda x: int(x.split('/')[-1].split('.')[0][1:]))
            return versions
        except Exception as e:
            logger.warning(f"Error listing prediction versions: {e}")
            return []
    
    # -------------------
    # Utility methods
    # -------------------
    def list_objects(self, prefix: str = '') -> List[str]:
        """List all objects with given prefix."""
        if not self.bucket:
            return []
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, key: str):
        """Delete an object from GCS."""
        if not self.bucket:
            return
        try:
            blob = self.bucket.blob(key)
            blob.delete()
            logger.info(f"Deleted {key} from GCS")
        except NotFound:
            logger.warning(f"Object {key} not found, skipping deletion")
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            raise
    
    def _delete_gcs_prefix(self, gcs_uri: str):
        """
        Delete all objects under a given GCS prefix.
        
        Args:
            gcs_uri: GCS URI (e.g., 'gs://mlops-new/models/lightgbm/v1/')
        """
        parsed = urlparse(gcs_uri, allow_fragments=False)
        bucket_name = parsed.netloc or self.bucket_name
        prefix = parsed.path.lstrip("/")
        
        # If bucket name differs, get that bucket
        if not self.client:
            return

        if bucket_name != self.bucket_name:
            bucket = self.client.bucket(bucket_name)
        else:
            bucket = self.bucket
        
        logger.info(f"Deleting from bucket={bucket_name}, prefix={prefix}")
        
        try:
            blobs = bucket.list_blobs(prefix=prefix)
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            logger.info(f"Deleted {deleted_count} objects under prefix {prefix}")
        except Exception as e:
            logger.error(f"Error deleting prefix {prefix}: {e}")
            raise
    
    def _delete_s3_prefix(self, s3_uri: str):
        """
        Alias for _delete_gcs_prefix for backward compatibility.
        Delete all objects under a given GCS prefix.
        
        Args:
            s3_uri: GCS URI (e.g., 'gs://mlops-new/models/lightgbm/v1/')
        """
        return self._delete_gcs_prefix(s3_uri)
    
    def upload_file(self, local_file: Union[str, Path], key: str):
        """Upload a single file to GCS."""
        local_file = Path(local_file)
        if not local_file.exists():
            raise FileNotFoundError(f"File not found: {local_file}")
        
        try:
            if not self.bucket:
                logger.warning(f"GCS client not initialized. Skipping file upload: {local_file.name}")
                return

            blob = self.bucket.blob(key)
            blob.upload_from_filename(str(local_file))
            logger.info(f"Uploaded {local_file.name} to gs://{self.bucket_name}/{key}")
        except Exception as e:
            logger.error(f"Failed to upload {local_file.name}: {e}")
            raise
    
    def download_file(self, local_file: Union[str, Path], key: str):
        """Download a single file from GCS."""
        local_file = Path(local_file)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if not self.bucket:
                logger.warning(f"GCS client not initialized. Cannot download {key}.")
                if local_file.exists():
                     return
                raise FileNotFoundError("GCS not available and local file missing.")

            blob = self.bucket.blob(key)
            if not blob.exists():
                raise FileNotFoundError(f"Blob {key} not found in bucket {self.bucket_name}")
            
            blob.download_to_filename(str(local_file))
            logger.info(f"Downloaded {key} to {local_file}")
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            raise


# -------------------
# Project-specific convenience functions
# -------------------
def create_data_directories(base_path: str = "data"):
    """Create standard data directories for the project."""
    base_path = Path(base_path)
    directories = [
        base_path / "prices",
        base_path / "articles",
        base_path / "predictions"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    # Initialize GCSManager
    gcs_manager = GCSManager()
    
    # Example: Upload price data
    # gcs_manager.upload_price_data("BTCUSDT", "data/btcusdt.csv")
    
    # Example: Download price data
    # df = gcs_manager.download_price_data("BTCUSDT", "data/btcusdt.csv")
    
    # Example: Upload model
    # gcs_manager.upload_model(
    #     "models/lightgbm/v3/lgb_model.txt",
    #     "lightgbm",
    #     "v3",
    #     additional_files=["models/lightgbm/v3/lgb_model_features.pkl"]
    # )
    
    # Example: Upload predictions
    # predictions_df = pd.DataFrame({"prediction": [0.5, 0.3, 0.2]})
    # gcs_manager.upload_predictions(predictions_df, "BTCUSDT", "lightgbm", version=1)
    
    print("GCSManager initialized successfully")
