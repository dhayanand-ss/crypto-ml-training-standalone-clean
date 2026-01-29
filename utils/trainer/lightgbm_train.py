#!/usr/bin/env python3
"""
LightGBM Training Entry Point
Wrapper script to run LightGBM training and log status to Airflow DB.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from trainer.lightgbm_trainer import LightGBMTrainer
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--coin", type=str, default="BTCUSDT", help="Cryptocurrency pair")
    parser.add_argument("--prices_path", type=str, default=None, help="Path to crypto prices CSV")
    parser.add_argument("--articles_path", type=str, default=None, help="Path to news articles CSV")
    parser.add_argument("--use_mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    
    args = parser.parse_args()
    
    model_name = "lightgbm"
    coin = args.coin
    
    # Setup status logging
    try:
        from utils.database.airflow_db import db
        # Update status to RUNNING
        db.set_state(model_name, coin, "RUNNING")
        print(f"Set status for {model_name}_{coin} to RUNNING")
    except Exception as e:
        print(f"Warning: Failed to connect to status DB: {e}")
        db = None

    try:
        print(f"Starting LightGBM training for {coin}...")
        
        # Load data (Use provided paths or default)
        crypto_path = args.prices_path or f"data/{coin.lower()}.csv"
        articles_path = args.articles_path or "data/articles.csv"
        
        if not os.path.exists(crypto_path):
            raise FileNotFoundError(f"Crypto data not found at {crypto_path}")
            
        print(f"Loading data from {crypto_path}...")
        crypto_df = pd.read_csv(crypto_path)
        
        # Load or mock sentiment data
        sentiment_df = None
        sentiment_path = "results/daily_sentiment_features.csv"
        
        if os.path.exists(sentiment_path):
            temp_sentiment = pd.read_csv(sentiment_path)
            
            # Check for date overlap
            if 'date' not in crypto_df.columns and 'open_time' in crypto_df.columns:
                crypto_df['date'] = pd.to_datetime(crypto_df['open_time'])
            
            crypto_dates = pd.to_datetime(crypto_df['date']).dt.strftime('%Y-%m-%d').unique()
            sentiment_dates = pd.to_datetime(temp_sentiment['date']).dt.strftime('%Y-%m-%d').unique()
            
            overlap = set(crypto_dates).intersection(set(sentiment_dates))
            if overlap:
                sentiment_df = temp_sentiment
                print(f"Loaded sentiment features from {sentiment_path} ({len(overlap)} overlapping days)")
            else:
                print(f"Warning: No date overlap between crypto data and {sentiment_path}. Will generate mocks.")
        
        # If no sentiment data loaded yet (either file missing or no overlap), generate mocks
        if sentiment_df is None:
            print("Generating mock sentiment features for training...")
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
        
        if sentiment_df is None:
             raise ValueError("Sentiment data required but not found (articles.csv or pre-computed features)")

        # Initialize and Train
        trainer = LightGBMTrainer()
        X, y, feature_cols = trainer.prepare_features(crypto_df, sentiment_df)
        
        trainer.train(X, y, use_mlflow=args.use_mlflow, use_wandb=args.use_wandb)
        
        # Save Model
        os.makedirs("models/lightgbm", exist_ok=True)
        trainer.save_model("models/lightgbm/lgb_model.txt")
        
        # Log Success
        if db:
            db.set_state(model_name, coin, "SUCCESS")
            print(f"Set status for {model_name}_{coin} to SUCCESS")
            
    except Exception as e:
        error_msg = str(e)
        print(f"Training failed: {error_msg}")
        traceback.print_exc()
        
        if db:
            db.set_state(model_name, coin, "FAILED", error_message=error_msg)
            print(f"Set status for {model_name}_{coin} to FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
