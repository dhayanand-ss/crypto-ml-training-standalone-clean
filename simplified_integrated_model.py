"""
Simplified Integrated Crypto ML Model
Combines LightGBM and Time Series Transformer without FinBERT dependencies
for comprehensive cryptocurrency price prediction
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import our custom models
from trainer.lightgbm_trainer import LightGBMTrainer
from trainer.time_series_transformer import TimeSeriesTransformer, TimeSeriesTransformerTrainer
from models.finbert_sentiment import FinBERTSentimentAnalyzer
from data_fetcher import load_or_fetch_price_data
from utils.model_version_manager import ModelVersionManager, ConsumerManager
from utils.project_output_formatter import ProjectOutputFormatter

# Import training utilities
try:
    from trainer.train_utils import preprocess_common, preprocess_common_batch, save_start_time, load_start_time, convert_to_onnx
    TRAIN_UTILS_AVAILABLE = True
except ImportError:
    TRAIN_UTILS_AVAILABLE = False
    print("Warning: train_utils not available. Some features will be disabled.")

# MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = os.getenv("SKIP_MLFLOW", "0") != "1"
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not available. MLflow logging will be disabled.")

class SimplifiedIntegratedModel:
    """Simplified integrated model that combines LightGBM, Time Series Transformer, and FinBERT"""
    
    def __init__(self, use_versioning=True):
        """
        Initialize the integrated model
        
        Args:
            use_versioning: Whether to use model versioning system (v1/v2/v3)
        """
        self.lgb_trainer = None
        self.transformer_trainer = None
        self.finbert_analyzer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_versioning = use_versioning
        
        # Initialize version manager if versioning is enabled
        if self.use_versioning:
            self.version_manager = ModelVersionManager()
            self.consumer_manager = ConsumerManager(self.version_manager)
        else:
            self.version_manager = None
            self.consumer_manager = None
        
        print(f"Initializing Simplified Integrated Crypto Model on {self.device}")
        print("Models: LightGBM + Time Series Transformer + FinBERT")
        if self.use_versioning:
            print("Model versioning enabled: v1 (baseline), v2 (previous), v3 (latest)")
    
    def load_data(self, symbol="BTCUSDT", interval="1m", start_date="2023-01-01"):
        """
        Load crypto and news data from real sources.
        
        Parameters:
            symbol: Trading pair symbol (default: "BTCUSDT")
            interval: Candle interval (default: "1m")
            start_date: Start date for historical data (default: "2023-01-01")
        
        Returns:
            tuple: (crypto_df, news_df) where news_df is REQUIRED (will raise error if missing)
        
        Raises:
            FileNotFoundError: If data/articles.csv does not exist
            ValueError: If news data file is empty
        """
        print("Loading data...")
        
        try:
            # Load or fetch price data from Binance
            crypto_df = load_or_fetch_price_data(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                data_path="data"
            )
            
            # Parse open_time from string if needed
            if "open_time" in crypto_df.columns:
                if crypto_df["open_time"].dtype == 'object':
                    crypto_df["open_time"] = pd.to_datetime(crypto_df["open_time"], format='%Y-%m-%d %H:%M:%S', utc=True)
            
            # Create date column (YYYY-MM-DD) for merging with daily sentiment
            if "date" not in crypto_df.columns and "open_time" in crypto_df.columns:
                crypto_df["date"] = pd.to_datetime(crypto_df["open_time"]).dt.strftime('%Y-%m-%d')
            elif "date" in crypto_df.columns:
                # Ensure date is string format (YYYY-MM-DD) for merging with daily sentiment
                crypto_df["date"] = pd.to_datetime(crypto_df["date"]).dt.strftime('%Y-%m-%d')
            
            # Add price_change column (calculated from close prices)
            if "price_change" not in crypto_df.columns:
                crypto_df['price_change'] = crypto_df['close'].pct_change()
            
            # Sort by open_time
            crypto_df = crypto_df.sort_values("open_time").reset_index(drop=True)
            
            print(f"Loaded {len(crypto_df)} crypto data points")
            if "date" in crypto_df.columns:
                print(f"Date range: {crypto_df['date'].min()} to {crypto_df['date'].max()}")
            elif "open_time" in crypto_df.columns:
                print(f"Time range: {crypto_df['open_time'].min()} to {crypto_df['open_time'].max()}")
        except Exception as e:
            print(f"Error loading price data: {e}")
            print("Please check your internet connection and ensure the symbol is correct.")
            return None, None
        
        # News data is REQUIRED - load articles.csv
        news_path = os.path.join("data", "articles.csv")
        if not os.path.exists(news_path):
            raise FileNotFoundError(
                f"News data file not found: {news_path}\n"
                "News data is required for this pipeline. Please ensure data/articles.csv exists."
            )
        
        try:
            news_df = pd.read_csv(news_path)
            if len(news_df) == 0:
                raise ValueError("News data file is empty. Please add news articles to data/articles.csv")
            print(f"Loaded {len(news_df)} news articles")
        except pd.errors.EmptyDataError:
            raise ValueError(f"News data file is empty: {news_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading news data from {news_path}: {e}")
        
        return crypto_df, news_df
    
    def train_finbert_sentiment(self, news_df):
        """Train FinBERT sentiment analysis - REQUIRES news data"""
        if news_df is None or len(news_df) == 0:
            raise ValueError(
                "News data is required but not provided. "
                "FinBERT sentiment analysis requires news articles in data/articles.csv"
            )
        
        print("\n" + "="*50)
        print("Training FinBERT Sentiment Analysis")
        print("="*50)
        
        try:
            # FinBERTSentimentAnalyzer now requires LoRA (compulsory)
            self.finbert_analyzer = FinBERTSentimentAnalyzer()
            daily_sentiment = self.finbert_analyzer.get_daily_sentiment_features(news_df)
            
            # Save sentiment features
            os.makedirs("results", exist_ok=True)
            daily_sentiment.to_csv("results/daily_sentiment_features.csv", index=False)
            print("Daily sentiment features saved to results/daily_sentiment_features.csv")
            
            return daily_sentiment
            
        except Exception as e:
            raise RuntimeError(
                f"Error training FinBERT: {e}\n"
                "FinBERT requires transformers library and internet connection."
            )
    
    def train_lightgbm(self, crypto_df, sentiment_df=None, force_retrain=False, initialize_baseline=False):
        """
        Train LightGBM model (or load existing model if available)
        
        Args:
            crypto_df: Crypto price data
            sentiment_df: Sentiment features
            force_retrain: Force retraining even if model exists
            initialize_baseline: If True and v1 doesn't exist, initialize v1 baseline
        """
        print("\n" + "="*50)
        print("Training LightGBM Model")
        print("="*50)
        
        # Determine model path based on versioning
        if self.use_versioning:
            # Check if v1 baseline exists
            v1_path = self.version_manager.get_model_path("lightgbm", "1")
            
            if initialize_baseline and v1_path is None:
                # First time training - will initialize v1 baseline
                temp_model_path = 'models/lgb_model_temp.txt'
            elif not force_retrain and v1_path:
                # Load v1 baseline
                model_path = v1_path
            else:
                # Training new model - will register as v3
                temp_model_path = 'models/lgb_model_temp.txt'
                model_path = temp_model_path
        else:
            model_path = 'models/lgb_model.txt'
        
        # Check if model already exists (non-versioned or v1)
        if not force_retrain and os.path.exists(model_path):
            print(f"Found existing LightGBM model. Loading from {model_path}...")
            try:
                self.lgb_trainer = LightGBMTrainer()
                self.lgb_trainer.load_model(model_path)
                
                # Prepare features for evaluation
                X, y, feature_cols = self.lgb_trainer.prepare_features(crypto_df, sentiment_df)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Make predictions for evaluation
                y_pred_proba = self.lgb_trainer.model.predict(X_test, num_iteration=self.lgb_trainer.model.best_iteration)
                y_pred = np.argmax(y_pred_proba, axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"[OK] LightGBM model loaded successfully (skipped training)")
                print(f"Model accuracy: {accuracy:.3f}")
                return X_test, y_test, y_pred, y_pred, accuracy
                
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will retrain the model...")
                # Fall through to training
        
        # Train new model (or retrain if loading failed)
        try:
            self.lgb_trainer = LightGBMTrainer()
            X, y, feature_cols = self.lgb_trainer.prepare_features(crypto_df, sentiment_df)
            
            # Train model
            X_test, y_test, y_pred, y_pred_binary, accuracy = self.lgb_trainer.train(X, y)
            
            # Save model with versioning
            if self.use_versioning:
                # Save to temp path first
                temp_path = 'models/lgb_model_temp.txt'
                os.makedirs("models", exist_ok=True)
                self.lgb_trainer.save_model(temp_path)
                
                if initialize_baseline and v1_path is None:
                    # Initialize v1 baseline
                    self.version_manager.initialize_baseline("lightgbm", temp_path, {
                        "accuracy": float(accuracy),
                        "trained_at": datetime.now().isoformat()
                    })
                    print(f"[OK] Initialized LightGBM v1 baseline")
                else:
                    # Stop consumers before version shift
                    if self.consumer_manager:
                        self.consumer_manager.stop_consumers("lightgbm", ["2", "3"])
                    
                    # Register new model (shifts versions)
                    new_version = self.version_manager.register_new_model("lightgbm", temp_path, {
                        "accuracy": float(accuracy),
                        "trained_at": datetime.now().isoformat()
                    })
                    print(f"[OK] Registered new LightGBM model as v{new_version}")
                    
                    # Restart consumers
                    if self.consumer_manager:
                        self.consumer_manager.start_consumers("lightgbm", ["2", "3"])
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                # Non-versioned save
                os.makedirs("models", exist_ok=True)
                self.lgb_trainer.save_model(model_path)
            
            return X_test, y_test, y_pred, y_pred_binary, accuracy
            
        except Exception as e:
            print(f"Error training LightGBM: {e}")
            print("LightGBM requires lightgbm library.")
            return None, None, None, None, None
    
    def train_transformer(self, crypto_df, force_retrain=False, initialize_baseline=False):
        """
        Train Time Series Transformer (or load existing model if available)
        
        Args:
            crypto_df: Crypto price data
            force_retrain: Force retraining even if model exists
            initialize_baseline: If True and v1 doesn't exist, initialize v1 baseline
        """
        print("\n" + "="*50)
        print("Time Series Transformer")
        print("="*50)
        
        # Determine model paths based on versioning
        if self.use_versioning:
            v1_path = self.version_manager.get_model_path("tst", "1")
            
            if initialize_baseline and v1_path is None:
                model_path = 'models/tst_model_temp.pth'
                scaler_path = 'models/tst_scaler_temp.pkl'
            elif not force_retrain and v1_path:
                model_path = v1_path
                # Find scaler path
                model_dir = os.path.dirname(v1_path)
                scaler_path = os.path.join(model_dir, 'tst_scaler.pkl')
            else:
                model_path = 'models/tst_model_temp.pth'
                scaler_path = 'models/tst_scaler_temp.pkl'
        else:
            model_path = 'models/tst_model.pth'
            scaler_path = 'models/tst_scaler.pkl'
        
        # Check if model already exists
        if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Found existing TST model. Loading from {model_path}...")
            try:
                # Initialize model with reduced parameters for faster CPU training
                input_dim = 7  # open, high, low, close, volume, taker_base, taker_quote
                model = TimeSeriesTransformer(
                    input_dim=input_dim,
                    hidden_dim=32,      # Reduced from 64 for faster CPU training
                    num_heads=2,        # Keep at 2
                    ff_dim=64,          # Reduced from 128 for faster CPU training
                    num_layers=1,       # Reduced from 2 for faster CPU training
                    dropout=0.1,
                    num_classes=3       # Sell, Hold, Buy
                )
                
                # Load model weights
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                
                # Initialize trainer with loaded model
                self.transformer_trainer = TimeSeriesTransformerTrainer(model, self.device)
                
                # Load scaler
                self.transformer_trainer.scaler = joblib.load(scaler_path)
                
                # Prepare data for evaluation (needed for predictions)
                X_train, y_train, X_test, y_test = self.transformer_trainer.prepare_data(
                    crypto_df, 
                    sequence_length=15,  # Reduced from 30 for faster CPU training
                    test_size=0.2
                )
                
                # Evaluate model
                predictions, metrics = self.transformer_trainer.evaluate(X_test, y_test)
                
                print("[OK] TST model loaded successfully (skipped training)")
                return X_test, y_test, predictions, metrics
                
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will retrain the model...")
                # Fall through to training
        
        # Train new model (or retrain if loading failed)
        print("Training Time Series Transformer...")
        try:
            # Initialize model with reduced parameters for faster CPU training
            input_dim = 7  # open, high, low, close, volume, taker_base, taker_quote
            model = TimeSeriesTransformer(
                input_dim=input_dim,
                hidden_dim=32,      # Reduced from 64 for faster CPU training
                num_heads=2,        # Keep at 2
                ff_dim=64,          # Reduced from 128 for faster CPU training
                num_layers=1,       # Reduced from 2 for faster CPU training
                dropout=0.1,
                num_classes=3       # Sell, Hold, Buy
            )
            
            # Initialize trainer
            self.transformer_trainer = TimeSeriesTransformerTrainer(model, self.device)
            
            # Prepare data with reduced sequence length for faster CPU training
            X_train, y_train, X_test, y_test = self.transformer_trainer.prepare_data(
                crypto_df, 
                sequence_length=15,  # Reduced from 30 for faster CPU training
                test_size=0.2
            )
            
            # Split training data for validation
            val_size = int(len(X_train) * 0.2)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            # Train model with reduced epochs and batch size for faster CPU training
            self.transformer_trainer.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=16)
            
            # Evaluate model
            predictions, metrics = self.transformer_trainer.evaluate(X_test, y_test)
            
            # Save model with versioning
            if self.use_versioning:
                # Save to temp paths first
                temp_model_path = 'models/tst_model_temp.pth'
                temp_scaler_path = 'models/tst_scaler_temp.pkl'
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), temp_model_path)
                joblib.dump(self.transformer_trainer.scaler, temp_scaler_path)
                
                if initialize_baseline and v1_path is None:
                    # Initialize v1 baseline
                    self.version_manager.initialize_baseline("tst", temp_model_path, {
                        "metrics": metrics,
                        "trained_at": datetime.now().isoformat()
                    })
                    print(f"[OK] Initialized TST v1 baseline")
                else:
                    # Stop consumers before version shift
                    if self.consumer_manager:
                        self.consumer_manager.stop_consumers("tst", ["2", "3"])
                    
                    # Register new model (shifts versions)
                    new_version = self.version_manager.register_new_model("tst", temp_model_path, {
                        "metrics": metrics,
                        "trained_at": datetime.now().isoformat()
                    })
                    print(f"[OK] Registered new TST model as v{new_version}")
                    
                    # Restart consumers
                    if self.consumer_manager:
                        self.consumer_manager.start_consumers("tst", ["2", "3"])
                
                # Clean up temp files
                for temp_file in [temp_model_path, temp_scaler_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # Non-versioned save
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                joblib.dump(self.transformer_trainer.scaler, scaler_path)
            
            print("[OK] TST model trained and saved")
            return X_test, y_test, predictions, metrics
            
        except Exception as e:
            print(f"Error training Transformer: {e}")
            print("Transformer requires PyTorch and proper GPU setup.")
            return None, None, None, None
    
    def create_ensemble_features(self, crypto_df, sentiment_df, lgb_predictions, transformer_predictions):
        """Create ensemble features from all models"""
        print("\nCreating ensemble features...")
        
        # Start with basic crypto features
        ensemble_df = crypto_df.copy()
        
        # Add sentiment features (REQUIRED - sentiment_df must exist)
        if sentiment_df is None:
            raise ValueError("Sentiment features are required but not provided. Cannot create ensemble features.")
        
        # Normalize dates to string format (YYYY-MM-DD) to avoid timezone issues
        if ensemble_df['date'].dtype == 'object':
            # Already string, ensure format
            ensemble_df['date'] = pd.to_datetime(ensemble_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            # Convert datetime to string
            ensemble_df['date'] = pd.to_datetime(ensemble_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        if sentiment_df['date'].dtype == 'object':
            # Already string, ensure format
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            # Convert datetime to string (normalize timezone if present)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
        
        ensemble_df = ensemble_df.merge(sentiment_df, on='date', how='left')
        
        # Add technical indicators
        ensemble_df['sma_5'] = ensemble_df['close'].rolling(window=5).mean()
        ensemble_df['sma_20'] = ensemble_df['close'].rolling(window=20).mean()
        ensemble_df['rsi'] = self.calculate_rsi(ensemble_df['close'])
        ensemble_df['volatility'] = ensemble_df['price_change'].rolling(window=10).std()
        
        # Add model predictions as features
        if lgb_predictions is not None:
            # Pad LGB predictions to match crypto data length
            lgb_padded = np.full(len(ensemble_df), np.nan)
            lgb_padded[-len(lgb_predictions):] = lgb_predictions
            ensemble_df['lgb_prediction'] = lgb_padded
        
        if transformer_predictions is not None:
            # Pad transformer predictions to match crypto data length
            transformer_padded = np.full(len(ensemble_df), np.nan)
            transformer_padded[-len(transformer_predictions):] = transformer_predictions.flatten()
            ensemble_df['transformer_prediction'] = transformer_padded
        
        # Create target variable
        ensemble_df['target'] = (ensemble_df['close'].shift(-1) > ensemble_df['close']).astype(int)
        
        # Remove NaN values
        ensemble_df = ensemble_df.dropna()
        
        return ensemble_df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def preprocess_for_inference(self, crypto_df, model_type="lightgbm", seq_len=30, horizon=1, threshold=0.00015):
        """
        Preprocess crypto data for single prediction inference using train_utils.
        
        Parameters:
            crypto_df: DataFrame with OHLCV data
            model_type: "lightgbm" or "tst"
            seq_len: Sequence length for TST (default: 30)
            horizon: Prediction horizon (default: 1)
            threshold: Classification threshold (default: 0.00015)
        
        Returns:
            Preprocessed features ready for model prediction
        """
        if not TRAIN_UTILS_AVAILABLE:
            raise ImportError("train_utils not available. Cannot use preprocess_for_inference.")
        
        return preprocess_common(model_type, crypto_df, seq_len=seq_len, 
                                horizon=horizon, threshold=threshold, 
                                return_first=True, inference=True)
    
    def preprocess_batch_for_inference(self, crypto_df, model_type="lightgbm", seq_len=30, horizon=1, threshold=0.00015):
        """
        Preprocess crypto data for batch prediction inference using train_utils.
        
        Parameters:
            crypto_df: DataFrame with OHLCV data
            model_type: "lightgbm" or "tst"
            seq_len: Sequence length for TST (default: 30)
            horizon: Prediction horizon (default: 1)
            threshold: Classification threshold (default: 0.00015)
        
        Returns:
            Preprocessed features ready for batch model prediction
        """
        if not TRAIN_UTILS_AVAILABLE:
            raise ImportError("train_utils not available. Cannot use preprocess_batch_for_inference.")
        
        return preprocess_common_batch(model_type, crypto_df, seq_len=seq_len, 
                                      horizon=horizon, threshold=threshold, 
                                      return_first=True, inference=True)
    
    def get_predictions_from_all_versions(self, model_type: str, X_test, crypto_df=None):
        """
        Get predictions from all versions (v1, v2, v3) of a model.
        
        Args:
            model_type: Type of model ('lightgbm', 'tst', 'finbert')
            X_test: Test data for predictions
            crypto_df: Crypto dataframe (needed for TST predictions)
        
        Returns:
            Dictionary with 'v1', 'v2', 'v3' keys containing predictions or None
        """
        if not self.use_versioning or self.version_manager is None:
            return {"v1": None, "v2": None, "v3": None}
        
        predictions = {"v1": None, "v2": None, "v3": None}
        
        for version in ["1", "2", "3"]:
            model_path = self.version_manager.get_model_path(model_type, version)
            if model_path is None:
                continue
            
            try:
                if model_type == "lightgbm":
                    trainer = LightGBMTrainer()
                    trainer.load_model(model_path)
                    pred = trainer.predict_proba(X_test)
                    predictions[f"v{version}"] = pred
                elif model_type == "tst":
                    # Load model and scaler
                    model_dir = os.path.dirname(model_path)
                    scaler_path = os.path.join(model_dir, 'tst_scaler.pkl')
                    
                    if not os.path.exists(scaler_path):
                        continue
                    
                    input_dim = 7
                    model = TimeSeriesTransformer(
                        input_dim=input_dim,
                        hidden_dim=32,
                        num_heads=2,
                        ff_dim=64,
                        num_layers=1,
                        dropout=0.1,
                        num_classes=3
                    )
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    
                    trainer = TimeSeriesTransformerTrainer(model, self.device)
                    trainer.scaler = joblib.load(scaler_path)
                    
                    # Get current prices if needed
                    current_prices = None
                    if crypto_df is not None:
                        test_start_idx = len(crypto_df) - len(X_test)
                        current_prices = crypto_df['close'].values[test_start_idx:test_start_idx+len(X_test)]
                    
                    pred = trainer.predict_three_class(X_test, current_prices)
                    predictions[f"v{version}"] = pred
                elif model_type == "finbert":
                    # FinBERT doesn't have versioned models in the same way
                    # This would need to be implemented based on your FinBERT versioning strategy
                    pass
            except Exception as e:
                logger.warning(f"Error loading {model_type} v{version}: {e}")
                continue
        
        return predictions
    
    def combine_three_class_predictions(self, lgb_3class=None, tst_3class=None, finbert_3class=None,
                                       lgb_versions=None, tst_versions=None, finbert_versions=None,
                                       weights=None, return_format='dict', use_latest_only=False):
        """
        Combine 3-class predictions from all models and versions into final ensemble output.
        
        Args:
            lgb_3class: LightGBM 3-class predictions (latest) - array of shape (n, 3) or None
            tst_3class: TST 3-class predictions (latest) - array of shape (n, 3) or None
            finbert_3class: FinBERT 3-class predictions (latest) - array of shape (n, 3) or None
            lgb_versions: Dict with 'v1', 'v2', 'v3' keys containing predictions
            tst_versions: Dict with 'v1', 'v2', 'v3' keys containing predictions
            finbert_versions: Dict with 'v1', 'v2', 'v3' keys containing predictions
            weights: Dictionary with model weights (default: {'lgb': 0.35, 'tst': 0.33, 'finbert': 0.32})
            return_format: 'dict' for dictionary format, 'array' for numpy array
            use_latest_only: If True, only use latest (v3) predictions. If False, include all versions.
        
        Returns:
            Dictionary or array with combined predictions and final decision
        """
        # Helper function for getting action from probabilities
        def get_action(probs):
            actions = ['SELL', 'HOLD', 'BUY']
            idx = np.argmax(probs)
            return actions[idx], float(probs[idx])
        
        if weights is None:
            weights = {
                'lgb': 0.35,
                'tst': 0.33,
                'finbert': 0.32
            }
        
        # Determine which predictions to use
        # If versions dicts are provided, use them; otherwise use single predictions
        if use_latest_only or (lgb_versions is None and tst_versions is None):
            # Use latest only (backward compatible)
            if lgb_3class is not None:
                lgb_v1 = lgb_v2 = lgb_v3 = np.asarray(lgb_3class)
            else:
                lgb_v1 = lgb_v2 = lgb_v3 = None
            
            if tst_3class is not None:
                tst_v1 = tst_v2 = tst_v3 = np.asarray(tst_3class)
            else:
                tst_v1 = tst_v2 = tst_v3 = None
            
            if finbert_3class is not None:
                finbert_v1 = finbert_v2 = finbert_v3 = np.asarray(finbert_3class)
            else:
                finbert_v1 = finbert_v2 = finbert_v3 = None
        else:
            # Use versioned predictions
            lgb_v1 = np.asarray(lgb_versions['v1']) if lgb_versions and lgb_versions['v1'] is not None else None
            lgb_v2 = np.asarray(lgb_versions['v2']) if lgb_versions and lgb_versions['v2'] is not None else None
            lgb_v3 = np.asarray(lgb_versions['v3']) if lgb_versions and lgb_versions['v3'] is not None else None
            
            tst_v1 = np.asarray(tst_versions['v1']) if tst_versions and tst_versions['v1'] is not None else None
            tst_v2 = np.asarray(tst_versions['v2']) if tst_versions and tst_versions['v2'] is not None else None
            tst_v3 = np.asarray(tst_versions['v3']) if tst_versions and tst_versions['v3'] is not None else None
            
            finbert_v1 = np.asarray(finbert_versions['v1']) if finbert_versions and finbert_versions['v1'] is not None else None
            finbert_v2 = np.asarray(finbert_versions['v2']) if finbert_versions and finbert_versions['v2'] is not None else None
            finbert_v3 = np.asarray(finbert_versions['v3']) if finbert_versions and finbert_versions['v3'] is not None else None
        
        # Determine minimum length across all available predictions
        lengths = []
        for pred in [lgb_v1, lgb_v2, lgb_v3, tst_v1, tst_v2, tst_v3, finbert_v1, finbert_v2, finbert_v3]:
            if pred is not None:
                lengths.append(len(pred))
        
        if not lengths:
            raise ValueError("No predictions provided")
        
        min_len = min(lengths)
        
        # Truncate all predictions to same length
        def truncate(pred):
            return pred[:min_len] if pred is not None else None
        
        lgb_v1, lgb_v2, lgb_v3 = truncate(lgb_v1), truncate(lgb_v2), truncate(lgb_v3)
        tst_v1, tst_v2, tst_v3 = truncate(tst_v1), truncate(tst_v2), truncate(tst_v3)
        finbert_v1, finbert_v2, finbert_v3 = truncate(finbert_v1), truncate(finbert_v2), truncate(finbert_v3)
        
        # Use v3 (latest) for ensemble calculation, fallback to v2 or v1
        lgb_for_ensemble = lgb_v3 if lgb_v3 is not None else (lgb_v2 if lgb_v2 is not None else lgb_v1)
        tst_for_ensemble = tst_v3 if tst_v3 is not None else (tst_v2 if tst_v2 is not None else tst_v1)
        finbert_for_ensemble = finbert_v3 if finbert_v3 is not None else (finbert_v2 if finbert_v2 is not None else finbert_v1)
        
        # Combine predictions (weighted average) - use latest versions
        ensemble_probs = np.zeros((min_len, 3))
        total_weight = 0
        
        if lgb_for_ensemble is not None:
            ensemble_probs += lgb_for_ensemble * weights['lgb']
            total_weight += weights['lgb']
        
        if tst_for_ensemble is not None:
            ensemble_probs += tst_for_ensemble * weights['tst']
            total_weight += weights['tst']
        
        if finbert_for_ensemble is not None:
            ensemble_probs += finbert_for_ensemble * weights['finbert']
            total_weight += weights['finbert']
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        # Get final decisions using argmax
        decisions = np.argmax(ensemble_probs, axis=1)  # 0=Sell, 1=Hold, 2=Buy
        confidences = ensemble_probs[np.arange(len(ensemble_probs)), decisions]
        
        # Map decisions to action strings
        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        actions = [action_map[d] for d in decisions]
        
        if return_format == 'dict':
            # Return as dictionary with per-sample results
            results = []
            for i in range(min_len):
                result = {
                    'ensemble': ensemble_probs[i].tolist(),
                    'action': actions[i],
                    'confidence': float(confidences[i]),
                    'probabilities': {
                        'sell': float(ensemble_probs[i][0]),
                        'hold': float(ensemble_probs[i][1]),
                        'buy': float(ensemble_probs[i][2])
                    }
                }
                
                # Add versioned predictions per OUTPUT_FORMAT_SPECIFICATION.md
                if lgb_v1 is not None:
                    result['lightgbm_1'] = lgb_v1[i].tolist()
                if lgb_v2 is not None:
                    result['lightgbm_2'] = lgb_v2[i].tolist()
                if lgb_v3 is not None:
                    result['lightgbm_3'] = lgb_v3[i].tolist()
                
                if tst_v1 is not None:
                    result['tst_1'] = tst_v1[i].tolist()
                if tst_v2 is not None:
                    result['tst_2'] = tst_v2[i].tolist()
                if tst_v3 is not None:
                    result['tst_3'] = tst_v3[i].tolist()
                
                if finbert_v1 is not None:
                    result['trl_1'] = finbert_v1[i].tolist()
                if finbert_v2 is not None:
                    result['trl_2'] = finbert_v2[i].tolist()
                if finbert_v3 is not None:
                    result['trl_3'] = finbert_v3[i].tolist()
                
                results.append(result)
            
            # Build model versions list
            model_versions = []
            if lgb_v1 is not None or lgb_v2 is not None or lgb_v3 is not None:
                versions = []
                if lgb_v1 is not None:
                    versions.append('lightgbm_v1')
                if lgb_v2 is not None:
                    versions.append('lightgbm_v2')
                if lgb_v3 is not None:
                    versions.append('lightgbm_v3')
                model_versions.extend(versions)
            
            if tst_v1 is not None or tst_v2 is not None or tst_v3 is not None:
                versions = []
                if tst_v1 is not None:
                    versions.append('tst_v1')
                if tst_v2 is not None:
                    versions.append('tst_v2')
                if tst_v3 is not None:
                    versions.append('tst_v3')
                model_versions.extend(versions)
            
            if finbert_v1 is not None or finbert_v2 is not None or finbert_v3 is not None:
                versions = []
                if finbert_v1 is not None:
                    versions.append('trl_v1')
                if finbert_v2 is not None:
                    versions.append('trl_v2')
                if finbert_v3 is not None:
                    versions.append('trl_v3')
                model_versions.extend(versions)
            
            # Also return summary statistics
            return {
                'results': results,
                'summary': {
                    'total_samples': min_len,
                    'actions_distribution': {
                        'SELL': actions.count('SELL'),
                        'HOLD': actions.count('HOLD'),
                        'BUY': actions.count('BUY')
                    },
                    'average_confidence': float(np.mean(confidences)),
                    'model_versions': model_versions
                }
            }
        else:
            # Return as arrays
            return {
                'lightgbm_1': lgb_v1,
                'lightgbm_2': lgb_v2,
                'lightgbm_3': lgb_v3,
                'tst_1': tst_v1,
                'tst_2': tst_v2,
                'tst_3': tst_v3,
                'trl_1': finbert_v1,
                'trl_2': finbert_v2,
                'trl_3': finbert_v3,
                'ensemble': ensemble_probs,
                'actions': np.array(actions),
                'confidences': np.array(confidences)
            }
    
    def train_ensemble(self, ensemble_df, force_retrain=False):
        """Train final ensemble model combining LightGBM and Transformer outputs (or load existing model if available)"""
        print("\n" + "="*50)
        print("Training Ensemble Model")
        print("="*50)
        
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        model_path = 'models/ensemble_model.pkl'
        
        # Check if model already exists
        if not force_retrain and os.path.exists(model_path):
            print("Found existing Ensemble model. Loading from disk...")
            try:
                ensemble = joblib.load(model_path)
                
                # Prepare features for evaluation
                exclude_cols = ['date', 'target', 'open_time']
                datetime_cols = [col for col in ensemble_df.columns if pd.api.types.is_datetime64_any_dtype(ensemble_df[col])]
                exclude_cols.extend(datetime_cols)
                feature_cols = [col for col in ensemble_df.columns if col not in exclude_cols]
                X = ensemble_df[feature_cols].values
                y = ensemble_df['target'].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Make predictions
                y_pred = ensemble.predict(X_test)
                y_pred_proba = ensemble.predict_proba(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"[OK] Ensemble model loaded successfully (skipped training)")
                print(f"Model accuracy: {accuracy:.3f}")
                return ensemble, X_test, y_test, y_pred, y_pred_proba, accuracy
                
            except Exception as e:
                print(f"Error loading existing model: {e}")
                print("Will retrain the model...")
                # Fall through to training
        
        # Prepare features - exclude datetime columns and target
        # Exclude columns that are datetime types or explicitly named datetime columns
        exclude_cols = ['date', 'target', 'open_time']
        # Also exclude any columns that are datetime dtype
        datetime_cols = [col for col in ensemble_df.columns if pd.api.types.is_datetime64_any_dtype(ensemble_df[col])]
        exclude_cols.extend(datetime_cols)
        
        feature_cols = [col for col in ensemble_df.columns if col not in exclude_cols]
        X = ensemble_df[feature_cols].values
        y = ensemble_df['target'].values
        
        print(f"Ensemble features: {feature_cols}")
        print(f"Feature matrix shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create ensemble of classifiers
        ensemble = VotingClassifier([
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ], voting='soft')
        
        # Train ensemble
        print("Training ensemble with voting classifier...")
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Ensemble Accuracy: {accuracy:.3f}")
        
        # Print classification report
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Price Down', 'Price Up']))
        
        # Save ensemble model
        os.makedirs("models", exist_ok=True)
        joblib.dump(ensemble, model_path)
        print("Ensemble model saved to models/ensemble_model.pkl")
        
        return ensemble, X_test, y_test, y_pred, y_pred_proba, accuracy
    
    def create_visualizations(self, crypto_df, ensemble_df, lgb_importance=None):
        """Create visualizations"""
        print("\nCreating visualizations...")
        
        os.makedirs("results", exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Price chart with moving averages
        axes[0, 0].plot(crypto_df.index, crypto_df['close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(crypto_df.index, crypto_df['close'].rolling(5).mean(), label='SMA 5', alpha=0.7)
        axes[0, 0].plot(crypto_df.index, crypto_df['close'].rolling(20).mean(), label='SMA 20', alpha=0.7)
        axes[0, 0].set_title('Bitcoin Price and Moving Averages')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Model predictions comparison
        if 'lgb_prediction' in ensemble_df.columns and 'transformer_prediction' in ensemble_df.columns:
            axes[0, 1].plot(ensemble_df['lgb_prediction'], label='LightGBM', alpha=0.7)
            axes[0, 1].plot(ensemble_df['transformer_prediction'], label='Transformer', alpha=0.7)
            axes[0, 1].set_title('Model Predictions Comparison')
            axes[0, 1].set_xlabel('Days')
            axes[0, 1].set_ylabel('Prediction Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Model predictions not available', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Model Predictions')
        
        # 3. Feature importance (if available)
        if lgb_importance is not None:
            top_features = lgb_importance.head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_title('LightGBM Feature Importance (Top 10)')
            axes[1, 0].set_xlabel('Importance')
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance not available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
        
        # 4. Price change distribution
        axes[1, 1].hist(crypto_df['price_change'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Price Change Distribution')
        axes[1, 1].set_xlabel('Price Change')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/simplified_integrated_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Analysis saved to results/simplified_integrated_analysis.png")
    
    def run_complete_pipeline(self):
        """Run the complete simplified integrated model pipeline"""
        print("=" * 60)
        print("Simplified Integrated Crypto ML Model Pipeline")
        print("=" * 60)
        print("This pipeline combines LightGBM, Time Series Transformer, and FinBERT")
        print("for cryptocurrency price prediction.")
        print()
        
        # Initialize MLflow
        mlflow_run = None
        mlflow_enabled = False
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow_uri = os.getenv("MLFLOW_URI", "http://localhost:5000")
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment("crypto-ml-pipeline")
                mlflow_run = mlflow.start_run()
                mlflow_enabled = True
                
                # Log pipeline parameters
                mlflow.log_param("use_versioning", self.use_versioning)
                mlflow.log_param("device", str(self.device))
                mlflow.set_tag("pipeline_type", "simplified_integrated")
                mlflow.set_tag("models", "LightGBM+TST+FinBERT+Ensemble")
                
                print(f"[MLFLOW] Started run: {mlflow_run.info.run_id}")
                print(f"[MLFLOW] Experiment: crypto-ml-pipeline")
                print(f"[MLFLOW] Tracking URI: {mlflow_uri}")
            except Exception as e:
                print(f"[MLFLOW] Warning: Could not initialize MLflow: {e}")
                print("[MLFLOW] Continuing without MLflow logging.")
                mlflow_run = None
                mlflow_enabled = False
        else:
            print("[MLFLOW] MLflow not available. Skipping MLflow logging.")
        
        # Track training start time
        if TRAIN_UTILS_AVAILABLE:
            try:
                save_start_time()
                start_time = load_start_time()
                print(f"Training started at: {datetime.fromtimestamp(start_time)}")
                if mlflow_enabled:
                    mlflow.log_param("training_start_time", datetime.fromtimestamp(start_time).isoformat())
            except Exception as e:
                print(f"Warning: Failed to track start time: {e}")
        
        # Load data (news data is REQUIRED)
        crypto_df, news_df = self.load_data()
        if crypto_df is None:
            raise ValueError("Failed to load crypto price data. Cannot continue.")
        
        # Train FinBERT sentiment analysis (REQUIRED - will fail if news data missing)
        sentiment_df = self.train_finbert_sentiment(news_df)
        if sentiment_df is None:
            raise RuntimeError("FinBERT sentiment analysis failed. Cannot continue without sentiment features.")
        
        # Train LightGBM
        lgb_results = self.train_lightgbm(crypto_df, sentiment_df)
        if lgb_results[0] is not None:
            X_test_lgb, y_test_lgb, y_pred_lgb, y_pred_binary_lgb, lgb_accuracy = lgb_results
        else:
            y_pred_lgb = None
        
        # Train Time Series Transformer
        transformer_results = self.train_transformer(crypto_df)
        if transformer_results[0] is not None:
            X_test_tst, y_test_tst, transformer_predictions, transformer_metrics = transformer_results
        else:
            transformer_predictions = None
            X_test_tst = None
        
        # Get 3-class predictions from all versions (v1, v2, v3) of each model
        print("\n" + "="*50)
        print("Generating 3-Class Predictions from All Versions")
        print("="*50)
        
        # Get predictions from all versions - these will be DIFFERENT models
        lgb_versions = None
        tst_versions = None
        finbert_versions = None
        
        if self.use_versioning:
            # Get predictions from all versions (v1, v2, v3) - each loads a DIFFERENT model
            if lgb_results[0] is not None:
                try:
                    print("Getting LightGBM predictions from all versions...")
                    lgb_versions = self.get_predictions_from_all_versions("lightgbm", X_test_lgb)
                    print(f"LightGBM v1: {'Available' if lgb_versions['v1'] is not None else 'Not available'}")
                    print(f"LightGBM v2: {'Available' if lgb_versions['v2'] is not None else 'Not available'}")
                    print(f"LightGBM v3: {'Available' if lgb_versions['v3'] is not None else 'Not available'}")
                except Exception as e:
                    print(f"Error getting LightGBM versioned predictions: {e}")
            
            if transformer_results[0] is not None:
                try:
                    print("Getting TST predictions from all versions...")
                    tst_versions = self.get_predictions_from_all_versions("tst", X_test_tst, crypto_df)
                    print(f"TST v1: {'Available' if tst_versions['v1'] is not None else 'Not available'}")
                    print(f"TST v2: {'Available' if tst_versions['v2'] is not None else 'Not available'}")
                    print(f"TST v3: {'Available' if tst_versions['v3'] is not None else 'Not available'}")
                except Exception as e:
                    print(f"Error getting TST versioned predictions: {e}")
        
        # Fallback: Use latest model if versioning not enabled or versions not available
        lgb_3class = None
        tst_3class = None
        finbert_3class = None
        
        if not self.use_versioning or (lgb_versions and all(v is None for v in lgb_versions.values())):
            if lgb_results[0] is not None and self.lgb_trainer is not None:
                try:
                    print("Getting LightGBM 3-class predictions (latest)...")
                    lgb_3class = self.lgb_trainer.predict_proba(X_test_lgb)
                    print(f"LightGBM 3-class predictions shape: {lgb_3class.shape}")
                except Exception as e:
                    print(f"Error getting LightGBM 3-class predictions: {e}")
        
        if not self.use_versioning or (tst_versions and all(v is None for v in tst_versions.values())):
            if transformer_results[0] is not None and self.transformer_trainer is not None:
                try:
                    print("Getting TST 3-class predictions (latest)...")
                    test_start_idx = len(crypto_df) - len(X_test_tst)
                    current_prices = crypto_df['close'].values[test_start_idx:test_start_idx+len(X_test_tst)]
                    tst_3class = self.transformer_trainer.predict_three_class(X_test_tst, current_prices)
                    print(f"TST 3-class predictions shape: {tst_3class.shape}")
                except Exception as e:
                    print(f"Error getting TST 3-class predictions: {e}")
        
        # FinBERT/TRL 3-class predictions (REQUIRED - sentiment_df must exist)
        if sentiment_df is None:
            raise ValueError("Sentiment features are required for FinBERT predictions.")
        
        if self.finbert_analyzer is not None:
            try:
                print("Getting FinBERT/TRL 3-class predictions...")
                if news_df is not None and len(news_df) > 0:
                    sample_texts = news_df['text'].head(min(100, len(news_df))).tolist()
                    finbert_3class = self.finbert_analyzer.predict_three_class(sample_texts)
                    print(f"FinBERT/TRL 3-class predictions shape: {finbert_3class.shape}")
            except Exception as e:
                print(f"Error getting FinBERT 3-class predictions: {e}")
        
        # Combine 3-class predictions with versioned predictions
        if (lgb_versions or lgb_3class is not None) and (tst_versions or tst_3class is not None):
            print("\n" + "="*50)
            print("Combining 3-Class Predictions")
            print("="*50)
            
            combined_results = self.combine_three_class_predictions(
                lgb_3class=lgb_3class, 
                tst_3class=tst_3class, 
                finbert_3class=finbert_3class,
                lgb_versions=lgb_versions,
                tst_versions=tst_versions,
                finbert_versions=finbert_versions,
                return_format='dict',
                use_latest_only=False  # Use all versions
            )
            
            # Save combined results (with ensemble/action/confidence for wrapper service)
            os.makedirs("results", exist_ok=True)
            with open("results/combined_three_class_predictions.json", "w") as f:
                json.dump(combined_results, f, indent=2)
            
            print(f"\nCombined predictions saved to results/combined_three_class_predictions.json")
            print(f"Total samples: {combined_results['summary']['total_samples']}")
            print(f"Actions distribution:")
            for action, count in combined_results['summary']['actions_distribution'].items():
                print(f"  {action}: {count}")
            print(f"Average confidence: {combined_results['summary']['average_confidence']:.3f}")
            
            # Generate project output format (without ensemble/action/confidence)
            if self.use_versioning and (lgb_versions or tst_versions):
                print("\n" + "="*50)
                print("Generating Project Output Format")
                print("="*50)
                
                # Format /prices/{coin} endpoint output
                prices_output = ProjectOutputFormatter.format_prices_output(
                    crypto_df, 
                    lgb_versions=lgb_versions,
                    tst_versions=tst_versions
                )
                
                # Format /trl endpoint output
                trl_output = None
                if finbert_versions or finbert_3class is not None:
                    # Create finbert_versions dict if needed
                    if finbert_versions is None and finbert_3class is not None:
                        finbert_versions = {'v1': None, 'v2': None, 'v3': np.asarray(finbert_3class)}
                    
                    trl_output = ProjectOutputFormatter.format_trl_output(
                        news_df,
                        trl_versions=finbert_versions
                    )
                
                # Save project format outputs
                with open("results/prices_output.json", "w") as f:
                    json.dump(prices_output, f, indent=2)
                print(f"[OK] Saved /prices/{{coin}} format to results/prices_output.json")
                
                if trl_output:
                    with open("results/trl_output.json", "w") as f:
                        json.dump(trl_output, f, indent=2)
                    print(f"[OK] Saved /trl format to results/trl_output.json")
            
            # Show first few predictions
            print("\nFirst 3 predictions:")
            for i, result in enumerate(combined_results['results'][:3]):
                print(f"\nSample {i+1}:")
                print(f"  LightGBM: {result.get('lightgbm_1', 'N/A')}")
                print(f"  TST:      {result.get('tst_1', 'N/A')}")
                if 'trl_1' in result and result['trl_1'] is not None:
                    print(f"  TRL:      {result['trl_1']}")
                if 'ensemble' in result:
                    print(f"  Ensemble: {result['ensemble']}")
                    print(f"  Action:   {result['action']} (confidence: {result['confidence']:.3f})")
        
        # Create ensemble features (for backward compatibility)
        ensemble_df = self.create_ensemble_features(
            crypto_df, sentiment_df, y_pred_lgb, transformer_predictions
        )
        
        # Train ensemble model
        ensemble_results = self.train_ensemble(ensemble_df)
        if ensemble_results[0] is not None:
            ensemble_model, X_test_ensemble, y_test_ensemble, y_pred_ensemble, y_pred_proba_ensemble, ensemble_accuracy = ensemble_results
        
        # Create visualizations
        lgb_importance = self.lgb_trainer.feature_importance if self.lgb_trainer else None
        self.create_visualizations(crypto_df, ensemble_df, lgb_importance)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SIMPLIFIED INTEGRATED MODEL TRAINING COMPLETED")
        print("=" * 60)
        print("Models trained:")
        finbert_success = sentiment_df is not None
        lgb_success = lgb_results[0] is not None
        tst_success = transformer_results[0] is not None
        ensemble_success = ensemble_results[0] is not None
        
        print(f"[OK] FinBERT Sentiment Analysis: {'Success' if finbert_success else 'FAILED (Required!)'}")
        print(f"[OK] LightGBM: {'Success' if lgb_success else 'Failed'}")
        print(f"[OK] Time Series Transformer: {'Success' if tst_success else 'Failed'}")
        print(f"[OK] Ensemble Model: {'Success' if ensemble_success else 'Failed'}")
        
        # Log metrics to MLflow
        if mlflow_enabled:
            try:
                mlflow.log_metric("finbert_success", 1 if finbert_success else 0)
                mlflow.log_metric("lightgbm_success", 1 if lgb_success else 0)
                mlflow.log_metric("tst_success", 1 if tst_success else 0)
                mlflow.log_metric("ensemble_success", 1 if ensemble_success else 0)
                
                # Log model accuracies if available
                if lgb_results[0] is not None:
                    _, _, _, _, lgb_accuracy = lgb_results
                    mlflow.log_metric("lightgbm_accuracy", lgb_accuracy)
                
                if transformer_results[0] is not None:
                    _, _, _, transformer_metrics = transformer_results
                    if transformer_metrics and 'accuracy' in transformer_metrics:
                        mlflow.log_metric("tst_accuracy", transformer_metrics['accuracy'])
                
                if ensemble_results[0] is not None:
                    _, _, _, _, _, ensemble_accuracy = ensemble_results
                    mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)
                
                # Log artifacts
                if os.path.exists("results/simplified_integrated_analysis.png"):
                    mlflow.log_artifact("results/simplified_integrated_analysis.png", "visualizations")
                if os.path.exists("results/daily_sentiment_features.csv"):
                    mlflow.log_artifact("results/daily_sentiment_features.csv", "data")
                
                print(f"[MLFLOW] Metrics and artifacts logged successfully")
            except Exception as e:
                print(f"[MLFLOW] Warning: Failed to log metrics: {e}")
        
        # Validate that all required components are present
        if sentiment_df is None:
            raise RuntimeError("Pipeline failed: FinBERT sentiment analysis is required but failed.")
        # Track training end time
        if TRAIN_UTILS_AVAILABLE:
            try:
                end_time = time.time()
                start_time = load_start_time()
                duration = end_time - start_time
                print(f"\nTraining completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
                if mlflow_enabled:
                    mlflow.log_metric("training_duration_seconds", duration)
                    mlflow.log_param("training_end_time", datetime.fromtimestamp(end_time).isoformat())
            except Exception as e:
                print(f"Warning: Failed to track end time: {e}")
        
        # Close MLflow run
        if mlflow_enabled and mlflow_run:
            try:
                run_id = mlflow_run.info.run_id
                mlflow.end_run()
                tracking_uri = str(mlflow.get_tracking_uri())
                print("[MLFLOW] Run completed successfully!")
                print(f"[MLFLOW] Run ID: {run_id}")
                print(f"[MLFLOW] View at: {tracking_uri}")
            except Exception as e:
                # Try to end run even if printing fails
                try:
                    mlflow.end_run()
                except:
                    pass
                print(f"[MLFLOW] Warning: Error during run closure: {e}")
                if mlflow_run:
                    print(f"[MLFLOW] Run ID was: {mlflow_run.info.run_id}")
        
        print()
        print("Files created:")
        print("- results/simplified_integrated_analysis.png")
        print("- results/combined_three_class_predictions.json")
        print("- models/lgb_model.txt")
        print("- models/tst_model.pth")
        print("- models/ensemble_model.pkl")
        print()
        print("You can now use the simplified integrated model for predictions!")
        print("Combined 3-class predictions are available in results/combined_three_class_predictions.json")

def main():
    """Main function"""
    simplified_model = SimplifiedIntegratedModel()
    simplified_model.run_complete_pipeline()

if __name__ == "__main__":
    main()
