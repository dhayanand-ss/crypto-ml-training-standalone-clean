"""
LightGBM Trainer Model
Uses LightGBM for cryptocurrency price prediction with advanced features
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import shutil
warnings.filterwarnings('ignore')

# Import training utilities
try:
    from trainer.train_utils import preprocess_crypto, log_classification_metrics, save_start_time, load_start_time
    TRAIN_UTILS_AVAILABLE = True
except ImportError:
    TRAIN_UTILS_AVAILABLE = False
    print("Warning: train_utils not available. Some features will be disabled.")

class LightGBMTrainer:
    """LightGBM trainer for cryptocurrency prediction"""
    
    def __init__(self, params=None):
        """Initialize LightGBM trainer with default parameters"""
        self.params = params or {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 3,  # 3 classes: Sell (0), Hold (1), Buy (2)
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.evals_result_ = None  # Store eval results to preserve training metadata
        self.best_iteration_ = None  # Store best iteration (also in model, but preserved here)
        self.best_score_ = None  # Store best score
        
    def prepare_features(self, crypto_df, sentiment_df=None):
        """Prepare features for LightGBM training"""
        print("Preparing features for LightGBM...")
        
        # Technical indicators
        crypto_df = crypto_df.copy()
        
        # Ensure date column exists (map open_time to date if needed)
        if 'date' not in crypto_df.columns and 'open_time' in crypto_df.columns:
            crypto_df['date'] = pd.to_datetime(crypto_df['open_time'])
        
        # Price-based features
        crypto_df['price_change'] = crypto_df['close'].pct_change()
        crypto_df['high_low_ratio'] = crypto_df['high'] / crypto_df['low']
        crypto_df['open_close_ratio'] = crypto_df['open'] / crypto_df['close']
        
        # Moving averages
        crypto_df['sma_5'] = crypto_df['close'].rolling(window=5).mean()
        crypto_df['sma_10'] = crypto_df['close'].rolling(window=10).mean()
        crypto_df['sma_20'] = crypto_df['close'].rolling(window=20).mean()
        crypto_df['sma_50'] = crypto_df['close'].rolling(window=50).mean()
        
        # Moving average ratios
        crypto_df['sma_5_ratio'] = crypto_df['close'] / crypto_df['sma_5']
        crypto_df['sma_20_ratio'] = crypto_df['close'] / crypto_df['sma_20']
        
        # Volatility features
        crypto_df['volatility_5'] = crypto_df['price_change'].rolling(window=5).std()
        crypto_df['volatility_10'] = crypto_df['price_change'].rolling(window=10).std()
        crypto_df['volatility_20'] = crypto_df['price_change'].rolling(window=20).std()
        
        # Volume features
        crypto_df['volume_sma_5'] = crypto_df['volume'].rolling(window=5).mean()
        crypto_df['volume_ratio'] = crypto_df['volume'] / crypto_df['volume_sma_5']
        
        # RSI calculation
        crypto_df['rsi'] = self.calculate_rsi(crypto_df['close'])
        
        # Bollinger Bands
        crypto_df['bb_upper'] = crypto_df['sma_20'] + (crypto_df['close'].rolling(window=20).std() * 2)
        crypto_df['bb_lower'] = crypto_df['sma_20'] - (crypto_df['close'].rolling(window=20).std() * 2)
        crypto_df['bb_position'] = (crypto_df['close'] - crypto_df['bb_lower']) / (crypto_df['bb_upper'] - crypto_df['bb_lower'])
        
        # Add sentiment features (REQUIRED)
        if sentiment_df is None:
            raise ValueError("Sentiment features are required for LightGBM training. Please provide sentiment_df.")
        
        # Normalize dates to string format (YYYY-MM-DD) to avoid timezone issues
        if crypto_df['date'].dtype == 'object':
            # Already string, ensure format
            crypto_df['date'] = pd.to_datetime(crypto_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            # Convert datetime to string
            crypto_df['date'] = pd.to_datetime(crypto_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        if sentiment_df['date'].dtype == 'object':
            # Already string, ensure format
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        else:
            # Convert datetime to string (normalize timezone if present)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')
        
        # Merge on date
        crypto_df = crypto_df.merge(sentiment_df, on='date', how='left')
        
        # Fill missing sentiment values
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'news_count', 
                        'sentiment_confidence', 'negative_sentiment', 
                        'neutral_sentiment', 'positive_sentiment']
        for col in sentiment_cols:
            if col in crypto_df.columns:
                col_mean = crypto_df[col].mean()
                fill_val = col_mean if not pd.isna(col_mean) else 0
                crypto_df[col] = crypto_df[col].fillna(fill_val)
        
        # Create 3-class target variable (Sell, Hold, Buy) using threshold
        # Calculate price change percentage
        price_change_pct = (crypto_df['close'].shift(-1) - crypto_df['close']) / crypto_df['close']
        
        # Create 3-class labels: 0=Sell, 1=Hold, 2=Buy
        # Using threshold to determine significant price movements
        threshold = 0.00015  # 0.015% threshold for 1-minute data
        crypto_df['target'] = self._label_price_change(price_change_pct, threshold=threshold)
        
        # Select features
        feature_cols = [col for col in crypto_df.columns if col not in ['date', 'target']]
        feature_cols = [col for col in feature_cols if not col.startswith('open_time')]
        
        # Remove rows with NaN values (technical indicators, last target, etc.)
        # Log before dropping to debug
        initial_len = len(crypto_df)
        print(f"DEBUG: Before dropna - Shape: {crypto_df.shape}")
        print("DEBUG: First 5 rows:")
        print(crypto_df.head())
        print("DEBUG: NaN counts:")
        print(crypto_df.isna().sum())
        
        crypto_df = crypto_df.dropna()
        dropped_count = initial_len - len(crypto_df)
        if dropped_count > 0:
             print(f"Dropped {dropped_count} rows containing NaNs (mostly indicator warm-up)")
        
        if len(crypto_df) == 0:
            print("WARNING: All samples dropped after preparing features. Check for NaNs in input data.")
            # Last-ditch attempt: if dropna made it zero, try to see which columns were NaN
            # This is just for debugging on VastAI
            print("NaN counts per column:")
            print(crypto_df.isna().sum())
            
        X = crypto_df[feature_cols].values
        y = crypto_df['target'].values
        
        self.feature_names = feature_cols
        
        print(f"Created {len(feature_cols)} features for {len(X)} samples")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _label_price_change(self, price_change_pct, threshold=0.00015):
        """
        Label price changes using threshold (for training data labeling only)
        
        This function is used during training data preparation to create labels from
        historical price movements. NOT used during inference.
        
        Args:
            price_change_pct: Price change percentage (can be scalar or array)
                            Calculated as: (future_price - current_price) / current_price
            threshold: Threshold for significant price change (default: 0.015% = 0.00015)
                      For 1-minute price data
        
        Returns:
            Label: 0=Sell, 1=Hold, 2=Buy (can be scalar or array)
        """
        price_change_pct = np.asarray(price_change_pct)
        labels = np.zeros_like(price_change_pct, dtype=int)
        
        # Label based on threshold
        labels[price_change_pct > threshold] = 2   # Buy
        labels[price_change_pct < -threshold] = 0  # Sell
        labels[(price_change_pct >= -threshold) & (price_change_pct <= threshold)] = 1  # Hold
        
        if price_change_pct.ndim == 0:  # Scalar
            return int(labels.item())
        else:
            return labels
    
    def train(self, X, y, test_size=0.2, cv_folds=5, use_mlflow=False, use_wandb=False):
        """Train LightGBM model"""
        print("Training LightGBM model...")

        # Setup MLflow if enabled
        if use_mlflow:
            try:
                import mlflow
                import mlflow.lightgbm
                # mlflow.lightgbm.autolog()
                print("MLflow autologging disabled (using manual logging)")
            except ImportError:
                print("Warning: MLflow not installed. logging disabled.")
                use_mlflow = False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Preserve training metadata BEFORE saving/reloading
        # These are lost when reloading with lgb.Booster(model_file=...)
        # evals_result is a dict: {'train': {'multi_logloss': [...]}, 'valid': {'multi_logloss': [...]}}
        self.evals_result_ = getattr(self.model, 'evals_result', None)
        self.best_iteration_ = getattr(self.model, 'best_iteration', None)
        # best_score may not always be available, but best_iteration is preserved in saved model
        self.best_score_ = getattr(self.model, 'best_score', None)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)  # Get class predictions from probabilities
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"LightGBM Accuracy: {accuracy:.3f}")
        
        # Print classification report
        print("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=['Sell', 'Hold', 'Buy'])
        print(report)
        
        # Log metrics using train_utils if available
        if TRAIN_UTILS_AVAILABLE:
            try:
                log_classification_metrics(y_pred, y_test, name="lightgbm_val", 
                                         class_labels=['0', '1', '2'], 
                                         use_mlflow=use_mlflow, use_wandb=use_wandb)
            except Exception as e:
                print(f"Warning: Failed to log metrics: {e}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(self.feature_importance.head(10))
        
        # Cross-validation
        cv_scores = cross_val_score(
            lgb.LGBMClassifier(**self.params), 
            X, y, cv=cv_folds, scoring='accuracy'
        )
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return X_test, y_test, y_pred, y_pred_proba, accuracy
    
    def predict(self, X):
        """Make predictions - returns class indices"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Use preserved best_iteration if available, otherwise use model's
        best_iter = self.best_iteration_ if self.best_iteration_ is not None else self.model.best_iteration
        predictions_proba = self.model.predict(X, num_iteration=best_iter)
        predictions = np.argmax(predictions_proba, axis=1)
        return predictions
    
    def predict_proba(self, X):
        """Make probability predictions - returns probabilities for all 3 classes"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Use preserved best_iteration if available, otherwise use model's
        best_iter = self.best_iteration_ if self.best_iteration_ is not None else self.model.best_iteration
        predictions_proba = self.model.predict(X, num_iteration=best_iter)
        return predictions_proba
    
    def get_training_metadata(self):
        """
        Get preserved training metadata (eval results, best iteration, etc.)
        
        Returns:
            dict with keys: evals_result, best_iteration, best_score, params
        """
        return {
            'evals_result': self.evals_result_,
            'best_iteration': self.best_iteration_,
            'best_score': self.best_score_,
            'params': self.params
        }
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("No feature importance data available. Train model first.")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'LightGBM Feature Importance (Top {top_n})')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path=None):
        """
        Save trained model with 3-model baseline versioning strategy.
        
        Versioning strategy:
        - v1 = baseline (first ever trained model, never overwritten)
        - v2 = previous model (stored before each new training run)
        - v3 = latest model (model produced by the current training run)
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # ============================================================
        # 3-Model Baseline Versioning Strategy
        # ============================================================
        base_dir = Path("models/lightgbm")
        v1_dir = base_dir / "v1"
        v2_dir = base_dir / "v2"
        v3_dir = base_dir / "v3"
        
        # Ensure directories exist
        v1_dir.mkdir(parents=True, exist_ok=True)
        v2_dir.mkdir(parents=True, exist_ok=True)
        v3_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        v1_model_path = v1_dir / "model.txt"
        v1_features_path = v1_dir / "model_features.pkl"
        v2_model_path = v2_dir / "model.txt"
        v2_features_path = v2_dir / "model_features.pkl"
        v3_model_path = v3_dir / "model.txt"
        v3_features_path = v3_dir / "model_features.pkl"
        
        print("\n" + "="*60)
        print("Saving LightGBM model with 3-model baseline versioning")
        print("="*60)
        
        # Before saving a new trained model:
        # Move the existing v3 → v2 (if v3 exists)
        if v3_model_path.exists() and v3_features_path.exists():
            print("[SAVE] Moving old v3 to v2")
            try:
                # Remove old v2 if it exists
                if v2_model_path.exists():
                    v2_model_path.unlink()
                if v2_features_path.exists():
                    v2_features_path.unlink()
                # Move v3 to v2
                v3_model_path.rename(v2_model_path)
                v3_features_path.rename(v2_features_path)
                print(f"[SAVE] Successfully moved v3 → v2")
            except Exception as e:
                print(f"[SAVE] Warning: Failed to move v3 → v2: {e}. Continuing...")
        
        # Keep v1 unchanged (v1 is only created once, never overwritten)
        # Create v1 baseline if it doesn't exist (first training run)
        if not v1_model_path.exists() or not v1_features_path.exists():
            print("[SAVE] Creating v1 baseline (first training run)")
            try:
                # Save model
                self.model.save_model(str(v1_model_path))
                # Save feature info and training metadata
                feature_info = {
                    'feature_names': self.feature_names,
                    'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
                    'evals_result': self.evals_result_,
                    'best_iteration': self.best_iteration_,
                    'best_score': self.best_score_,
                    'params': self.params
                }
                joblib.dump(feature_info, str(v1_features_path))
                print(f"[SAVE] Successfully created v1 baseline at {v1_dir}")
            except Exception as e:
                print(f"[SAVE] Warning: Failed to create v1 baseline: {e}")
        
        # Save the new trained model as v3
        print("[SAVE] Saving new v3 latest model")
        try:
            # Save LightGBM model (preserves trees and best_iteration)
            self.model.save_model(str(v3_model_path))
            
            # Save feature names AND training metadata
            # This preserves eval results that are lost when reloading with lgb.Booster()
            feature_info = {
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
                # Preserve training metadata
                'evals_result': self.evals_result_,
                'best_iteration': self.best_iteration_,
                'best_score': self.best_score_,
                'params': self.params
            }
            
            joblib.dump(feature_info, str(v3_features_path))
            print(f"[SAVE] Successfully saved new v3 latest model at {v3_dir}")
            
            # Print preserved metadata summary
            if self.best_iteration_ is not None:
                print(f"  Best iteration: {self.best_iteration_}")
            if self.best_score_ is not None:
                print(f"  Best score: {self.best_score_:.6f}")
            if self.evals_result_:
                print(f"  Eval results preserved for {len(self.evals_result_)} validation sets")
        except Exception as e:
            print(f"[SAVE] ERROR: Failed to save v3 model: {e}")
            raise
        
        # If model_path was provided (legacy support), also save there
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save_model(model_path)
            joblib.dump(feature_info, model_path.replace('.txt', '_features.pkl'))
            print(f"LightGBM model also saved to {model_path} (legacy path)")
    
    def load_model(self, model_path=None):
        """
        Load trained model with 3-model baseline versioning strategy.
        
        Loading priority:
        - Check if v3 exists → load v3
        - Else if v2 exists → load v2
        - Else → return False (no model found, need to train)
        
        Note: lgb.Booster(model_file=...) resets internal states except trees.
        This method restores eval results and training metadata from saved file.
        
        Returns:
            bool: True if model was loaded successfully, False if no model found
        """
        # ============================================================
        # 3-Model Baseline Versioning Strategy
        # ============================================================
        base_dir = Path("models/lightgbm")
        v1_dir = base_dir / "v1"
        v2_dir = base_dir / "v2"
        v3_dir = base_dir / "v3"
        
        # Define file paths
        v1_model_path = v1_dir / "model.txt"
        v1_features_path = v1_dir / "model_features.pkl"
        v2_model_path = v2_dir / "model.txt"
        v2_features_path = v2_dir / "model_features.pkl"
        v3_model_path = v3_dir / "model.txt"
        v3_features_path = v3_dir / "model_features.pkl"
        
        loaded_version = None
        model_path_to_load = None
        features_path_to_load = None
        
        # Check if v3 exists → load it
        if v3_model_path.exists() and v3_features_path.exists():
            print("[LOAD] Loading v3 latest model")
            model_path_to_load = v3_model_path
            features_path_to_load = v3_features_path
            loaded_version = "v3"
        # Else if v2 exists → load v2
        elif v2_model_path.exists() and v2_features_path.exists():
            print("[LOAD] Loading v2 previous model")
            model_path_to_load = v2_model_path
            features_path_to_load = v2_features_path
            loaded_version = "v2"
        # Else if v1 exists → load v1 (baseline)
        elif v1_model_path.exists() and v1_features_path.exists():
            print("[LOAD] Loading v1 baseline model")
            model_path_to_load = v1_model_path
            features_path_to_load = v1_features_path
            loaded_version = "v1"
        # Legacy support: if model_path provided, try to load from there
        elif model_path and os.path.exists(model_path):
            print(f"[LOAD] Loading from legacy path: {model_path}")
            model_path_to_load = Path(model_path)
            features_path_to_load = Path(model_path.replace('.txt', '_features.pkl'))
            loaded_version = "legacy"
        else:
            print("[LOAD] No versions found, need to train new model")
            return False
        
        # Load model (best_iteration is preserved in the model file)
        try:
            self.model = lgb.Booster(model_file=str(model_path_to_load))
            
            # Load feature info and training metadata
            if features_path_to_load.exists():
                feature_info = joblib.load(str(features_path_to_load))
                self.feature_names = feature_info.get('feature_names')
                
                # Restore feature importance
                if feature_info.get('feature_importance'):
                    self.feature_importance = pd.DataFrame(feature_info['feature_importance'])
                
                # Restore training metadata (preserved from training)
                self.evals_result_ = feature_info.get('evals_result')
                self.best_iteration_ = feature_info.get('best_iteration', self.model.best_iteration)
                self.best_score_ = feature_info.get('best_score')
                
                # Restore params if available
                if 'params' in feature_info:
                    self.params = feature_info['params']
                
                # Verify best_iteration matches
                if self.best_iteration_ != self.model.best_iteration:
                    print(f"Warning: Saved best_iteration ({self.best_iteration_}) != model.best_iteration ({self.model.best_iteration})")
                    # Use model's best_iteration as source of truth
                    self.best_iteration_ = self.model.best_iteration
            else:
                # Fallback: use model's best_iteration if metadata file doesn't exist
                self.best_iteration_ = self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
                print(f"Warning: Feature info file not found. Training metadata may be incomplete.")
            
            print(f"[LOAD] Successfully loaded {loaded_version} model from {model_path_to_load}")
            if self.best_iteration_ is not None:
                print(f"  Best iteration: {self.best_iteration_}")
            if self.best_score_ is not None:
                print(f"  Best score: {self.best_score_:.6f}")
            if self.evals_result_:
                print(f"  Eval results restored for {len(self.evals_result_)} validation sets")
            
            return True
        except Exception as e:
            print(f"[LOAD] Failed to load model: {e}")
            return False

def main():
    """Main function to demonstrate LightGBM training"""
    import argparse
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--use_mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    print("=" * 60)
    print("LightGBM Training")
    print("=" * 60)
    
    # Load data
    crypto_path = "data/btcusdt.csv"
    if not os.path.exists(crypto_path):
        print(f"Error: Crypto data not found at {crypto_path}")
        print("Please fetch price data using data_fetcher.py")
        print("Example: python data_fetcher.py --symbol BTCUSDT --interval 1h --start-date 2024-01-01")
        return
    
    crypto_df = pd.read_csv(crypto_path)
    print(f"Loaded {len(crypto_df)} crypto data points")
    
    # Load sentiment data if available
    sentiment_df = None
    if os.path.exists("data/articles.csv"):
        # For demo purposes, we can process articles here or assume pre-processed
        pass
    
    if os.path.exists("results/daily_sentiment_features.csv"):
        sentiment_df = pd.read_csv("results/daily_sentiment_features.csv")
        print(f"Loaded {len(sentiment_df)} sentiment features")
    elif os.path.exists("data/articles.csv") and os.path.exists("data/btcusdt.csv"):
        # Fallback: Create mock sentiment features for training demo
        print("Creating mock sentiment features for demonstration...")
        
        # Ensure date column exists
        if 'date' not in crypto_df.columns and 'open_time' in crypto_df.columns:
            crypto_df['date'] = pd.to_datetime(crypto_df['open_time'])
            
        dates = pd.to_datetime(crypto_df['date']).dt.strftime('%Y-%m-%d').unique()
        sentiment_df = pd.DataFrame({
            'date': dates,
            'sentiment_mean': np.random.normal(0.1, 0.5, len(dates)),
            'sentiment_std': np.random.uniform(0.1, 0.3, len(dates)),
            'news_count': np.random.randint(5, 50, len(dates)),
            'sentiment_confidence': np.random.uniform(0.8, 0.99, len(dates)),
            'negative_sentiment': np.random.uniform(0, 0.3, len(dates)),
            'neutral_sentiment': np.random.uniform(0.3, 0.7, len(dates)),
            'positive_sentiment': np.random.uniform(0, 0.3, len(dates))
        })
    
    # Initialize trainer
    trainer = LightGBMTrainer()
    
    if args.use_mlflow:
        import mlflow
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("lightgbm_training")
        
        with mlflow.start_run(run_name="manual_lightgbm"):
             # Prepare features
            X, y, feature_cols = trainer.prepare_features(crypto_df, sentiment_df)
            
            # Train model
            X_test, y_test, y_pred, y_pred_proba, accuracy = trainer.train(
                X, y, use_mlflow=True, use_wandb=args.use_wandb
            )
            
            # Log params
            mlflow.log_params(trainer.params)
            mlflow.log_metric("final_accuracy", accuracy)
            
            # Save model to disk
            os.makedirs("models", exist_ok=True)
            trainer.save_model("models/lgb_model.txt")
            
            # Log model to MLflow and register
            print("Logging model to MLflow...")
            # Log as artifact (safer than log_model for custom loading)
            mlflow.log_artifact("models/lgb_model.txt", artifact_path="model")
            
            # Register model (requires a run URI)
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            try:
                mlflow.register_model(model_uri, "lightgbm_local")
                print("Model registered as 'lightgbm_local'")
            except Exception as e:
                print(f"Warning: Failed to register model: {e}")
            
            print("\nLightGBM training completed!")
            print("Model saved to models/lgb_model.txt")
            
    else:
        # Prepare features
        X, y, feature_cols = trainer.prepare_features(crypto_df, sentiment_df)
        
        # Train model
        X_test, y_test, y_pred, y_pred_proba, accuracy = trainer.train(
            X, y, use_mlflow=False, use_wandb=args.use_wandb
        )
        
        # Save model
        os.makedirs("models", exist_ok=True)
        trainer.save_model("models/lgb_model.txt")
        
        print("\nLightGBM training completed!")
        print("Model saved to models/lgb_model.txt")

if __name__ == "__main__":
    main()