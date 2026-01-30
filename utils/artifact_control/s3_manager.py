"""
S3 Manager for Crypto ML Training
Handles S3 storage for models, data, and predictions with versioning support.
"""

import boto3
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


class S3Manager:
    """
    Manages S3 operations for the crypto ML training project.
    
    Supports:
    - Model storage with versioning (v1, v2, v3)
    - Data storage (prices, articles, predictions)
    - Hash-based deduplication
    - MLflow integration
    """
    
    def __init__(
        self,
        bucket: str = "mlops",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "ap-southeast-1"
    ):
        """
        Initialize S3Manager.
        
        Args:
            bucket: S3 bucket name (default: "mlops")
            endpoint_url: S3 endpoint URL (from S3_URL env var if not provided)
            aws_access_key_id: AWS access key (from AWS_ACCESS_KEY_ID env var if not provided)
            aws_secret_access_key: AWS secret key (from AWS_SECRET_ACCESS_KEY env var if not provided)
            region_name: AWS region name (default: "ap-southeast-1")
        """
        self.bucket = bucket
        self.endpoint_url = endpoint_url or os.getenv("S3_URL")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        
        try:
            self.s3 = boto3.client(
                service_name="s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=region_name
            )
            logger.info(f"S3Manager initialized with bucket: {bucket}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    # -------------------
    # Hash helpers
    # -------------------
    @staticmethod
    def compute_hash_bytes(data_bytes: bytes) -> str:
        """Compute SHA256 hash of bytes."""
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _get_s3_hash(self, hash_key: str) -> Optional[str]:
        """Get hash from S3 if it exists."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=hash_key)
            return obj['Body'].read().decode('utf-8')
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
        Upload DataFrame to S3 as Parquet with hash-based deduplication.
        
        Args:
            path_or_df: Path to CSV file or DataFrame
            key: S3 key (e.g., 'prices/BTCUSDT.parquet')
            skip_if_exists: Skip upload if hash matches existing file
            
        Returns:
            True if uploaded, False if skipped
        """
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
            existing_hash = self._get_s3_hash(hash_key)
            if existing_hash == file_hash:
                logger.info(f"Skipping upload for {key}, hash matches S3.")
                return False
        
        # Upload file and hash
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data_bytes)
            self.s3.put_object(Bucket=self.bucket, Key=hash_key, Body=file_hash.encode('utf-8'))
            logger.info(f"Uploaded {key} with hash {file_hash[:8]}... to S3.")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {key}: {e}")
            raise
    
    def download_df(
        self,
        local_file: str,
        key: str,
        skip_if_exists: bool = True
    ) -> pd.DataFrame:
        """
        Download DataFrame from S3 and save locally as CSV.
        
        Args:
            local_file: Local file path (will be saved as .csv)
            key: S3 key (e.g., 'prices/BTCUSDT.parquet')
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
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Download from S3
        try:
            buffer = BytesIO()
            self.s3.download_fileobj(self.bucket, key, buffer)
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
        Upload model files to S3 with versioning.
        
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
        s3_key = f"models/{model_type}/{version}/{model_path.name}"
        try:
            self.s3.upload_file(str(model_path), self.bucket, s3_key)
            logger.info(f"Uploaded {model_path.name} to s3://{self.bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise
        
        # Upload additional files (scaler, features, tokenizer, etc.)
        if additional_files:
            for file_path in additional_files:
                file_path = Path(file_path)
                if file_path.exists():
                    file_s3_key = f"models/{model_type}/{version}/{file_path.name}"
                    try:
                        self.s3.upload_file(str(file_path), self.bucket, file_s3_key)
                        logger.info(f"Uploaded {file_path.name} to s3://{self.bucket}/{file_s3_key}")
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
        Download model files from S3.
        
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
        
        s3_key = f"models/{model_type}/{version}/{model_filename}"
        local_path = local_dir / model_filename
        
        try:
            self.s3.download_file(self.bucket, s3_key, str(local_path))
            logger.info(f"Downloaded model from s3://{self.bucket}/{s3_key} to {local_path}")
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
        Upload entire model directory to S3.
        
        Args:
            local_dir: Local directory containing model files
            model_type: Type of model
            version: Version ('v1', 'v2', 'v3')
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            raise FileNotFoundError(f"Directory not found: {local_dir}")
        
        s3_prefix = f"models/{model_type}/{version}/"
        
        for file_path in local_dir.iterdir():
            if file_path.is_file():
                s3_key = f"{s3_prefix}{file_path.name}"
                try:
                    self.s3.upload_file(str(file_path), self.bucket, s3_key)
                    logger.info(f"Uploaded {file_path.name} to s3://{self.bucket}/{s3_key}")
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
        Upload predictions to S3.
        
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
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, key: str):
        """Delete an object from S3."""
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted {key} from S3")
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            raise
    
    def _delete_s3_prefix(self, s3_uri: str):
        """
        Delete all objects under a given S3 prefix.
        
        Args:
            s3_uri: S3 URI (e.g., 's3://mlops/models/lightgbm/v1/')
        """
        parsed = urlparse(s3_uri, allow_fragments=False)
        bucket = parsed.netloc or self.bucket
        prefix = parsed.path.lstrip("/")
        
        logger.info(f"Deleting from bucket={bucket}, prefix={prefix}")
        
        paginator = self.s3.get_paginator("list_objects_v2")
        
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue
                
                objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
                while objects_to_delete:
                    batch = objects_to_delete[:1000]
                    del objects_to_delete[:1000]
                    
                    self.s3.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": batch, "Quiet": True},
                    )
            logger.info(f"Deleted all objects under prefix {prefix}")
        except Exception as e:
            logger.error(f"Error deleting prefix {prefix}: {e}")
            raise
    
    def upload_file(self, local_file: Union[str, Path], key: str):
        """Upload a single file to S3."""
        local_file = Path(local_file)
        if not local_file.exists():
            raise FileNotFoundError(f"File not found: {local_file}")
        
        try:
            self.s3.upload_file(str(local_file), self.bucket, key)
            logger.info(f"Uploaded {local_file.name} to s3://{self.bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to upload {local_file.name}: {e}")
            raise
    
    def download_file(self, local_file: Union[str, Path], key: str):
        """Download a single file from S3."""
        local_file = Path(local_file)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.s3.download_file(self.bucket, key, str(local_file))
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
    # Initialize S3Manager
    s3_manager = S3Manager()
    
    # Example: Upload price data
    # s3_manager.upload_price_data("BTCUSDT", "data/btcusdt.csv")
    
    # Example: Download price data
    # df = s3_manager.download_price_data("BTCUSDT", "data/btcusdt.csv")
    
    # Example: Upload model
    # s3_manager.upload_model(
    #     "models/lightgbm/v3/lgb_model.txt",
    #     "lightgbm",
    #     "v3",
    #     additional_files=["models/lightgbm/v3/lgb_model_features.pkl"]
    # )
    
    # Example: Upload predictions
    # predictions_df = pd.DataFrame({"prediction": [0.5, 0.3, 0.2]})
    # s3_manager.upload_predictions(predictions_df, "BTCUSDT", "lightgbm", version=1)
    
    print("S3Manager initialized successfully")
