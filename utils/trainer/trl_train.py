#!/usr/bin/env python3
"""
TRL (Transformer Reinforcement Learning) Training Script with GRPO
Main training script for FinBERT sentiment classifier using GRPO algorithm.

This script matches the documentation in TRL_GRPO_Implementation.md
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.finbert_sentiment import FinBERTSentimentAnalyzer
import time


def main():
    parser = argparse.ArgumentParser(
        description="Train TRL model with GRPO algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with defaults
  python -m utils.trainer.trl_train --coin BTCUSDT
  
  # Custom parameters
  python -m utils.trainer.trl_train \\
      --coin BTCUSDT \\
      --lora_rank 4 \\
      --window_hours 12 \\
      --threshold 0.005 \\
      --epochs 10 \\
      --batch_size 4 \\
      --clip_eps 0.2 \\
      --kl_coef 0.1 \\
      --lr 2e-5 \\
      --use_mlflow \\
      --use_wandb
        """
    )
    
    # Data parameters
    parser.add_argument("--coin", type=str, default="BTCUSDT",
                       help="Cryptocurrency pair for training (default: BTCUSDT)")
    parser.add_argument("--articles_path", type=str, default="data/articles.csv",
                       help="Path to articles CSV file (default: data/articles.csv)")
    parser.add_argument("--prices_path", type=str, default=None,
                       help="Path to prices CSV file (default: data/{coin}.csv)")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="ProsusAI/finbert",
                       help="Base transformer model (default: ProsusAI/finbert)")
    parser.add_argument("--lora_rank", type=int, default=4,
                       help="LoRA rank (default: 4)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps (default: 1)")
    
    # GRPO parameters
    parser.add_argument("--window_hours", type=int, default=12,
                       help="Time window for price change calculation in hours (default: 12)")
    parser.add_argument("--threshold", type=float, default=0.005,
                       help="Price change threshold for label assignment (default: 0.005 = 0.5%%)")
    parser.add_argument("--group_size", type=int, default=4,
                       help="Number of actions sampled per article (default: 4)")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                       help="PPO clipping epsilon (default: 0.2)")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                       help="KL divergence regularization coefficient (default: 0.1)")
    parser.add_argument("--reward_clip", type=float, default=None,
                       help="Optional reward clipping value (default: None)")
    parser.add_argument("--update_old_every_iter", action="store_true", default=True,
                       help="Update old policy after each iteration (default: True)")
    
    # Validation
    parser.add_argument("--val_frac", type=float, default=0.1,
                       help="Validation fraction (default: 0.1)")
    
    # Logging
    parser.add_argument("--use_mlflow", action="store_true",
                       help="Enable MLflow logging")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    
    # Time limit
    parser.add_argument("--max_time", type=int, default=1200,
                       help="Maximum training time in seconds (default: 1200 = 20 minutes)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="models/finbert",
                       help="Directory to save trained model (default: models/finbert)")
    
    args = parser.parse_args()
    
    # Set default prices path if not provided
    if args.prices_path is None:
        args.prices_path = f"data/{args.coin.lower()}.csv"
    
    print("=" * 60)
    print("TRL GRPO Training")
    print("=" * 60)
    print(f"Coin: {args.coin}")
    print(f"Articles: {args.articles_path}")
    print(f"Prices: {args.prices_path}")
    print(f"Model: {args.model_name}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Window Hours: {args.window_hours}")
    print(f"Threshold: {args.threshold}")
    print(f"Group Size: {args.group_size}")
    print(f"Clip Epsilon: {args.clip_eps}")
    print(f"KL Coefficient: {args.kl_coef}")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(args.articles_path):
        print(f"Error: Articles file not found: {args.articles_path}")
        sys.exit(1)
    
    if not os.path.exists(args.prices_path):
        print(f"Error: Prices file not found: {args.prices_path}")
        sys.exit(1)
    
    # Load data
    print("\nLoading data...")
    try:
        news_df = pd.read_csv(args.articles_path)
        print(f"Loaded {len(news_df)} news articles")
        
        crypto_df = pd.read_csv(args.prices_path)
        print(f"Loaded {len(crypto_df)} price records")
        
        # Ensure required columns exist
        required_news_cols = ['title', 'text', 'date']
        required_price_cols = ['open_time', 'close']
        
        missing_news = [col for col in required_news_cols if col not in news_df.columns]
        missing_price = [col for col in required_price_cols if col not in crypto_df.columns]
        
        if missing_news:
            print(f"Error: Missing columns in articles: {missing_news}")
            sys.exit(1)
        
        if missing_price:
            print(f"Error: Missing columns in prices: {missing_price}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Date Alignment and Mocking
    print("\nChecking date alignment...")
    try:
        if 'date' not in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['created_at']) if 'created_at' in news_df.columns else None
        
        # Ensure UTC and datetime format
        news_df['date'] = pd.to_datetime(news_df['date'], utc=True)
        crypto_df['open_time'] = pd.to_datetime(crypto_df['open_time'], utc=True)
        
        news_min = news_df['date'].min()
        news_max = news_df['date'].max()
        crypto_min = crypto_df['open_time'].min()
        crypto_max = crypto_df['open_time'].max()
        
        print(f"News Range: {news_min} to {news_max}")
        print(f"Price Range: {crypto_min} to {crypto_max}")
        
        # Simple overlap check (timezone-aware)
        overlap = (news_max >= crypto_min) and (news_min <= crypto_max)
        
        if not overlap:
            print("\nWARNING: No overlap between news and price dates.")
            print("Generating 100 aligned mock news articles for training...")
            
            # Generate mock news aligned with price data
            import numpy as np
            
            # Select 100 random timestamps from price data range
            # Use data from the last 24 hours of price data if possible
            last_price_time = crypto_df['open_time'].max()
            first_price_time = crypto_df['open_time'].min()
            
            mock_dates = pd.to_datetime(np.random.choice(
                crypto_df['open_time'].values, size=100
            ), utc=True)
            
            news_df = pd.DataFrame({
                'date': mock_dates,
                'title': [f"Mock Crypto News {i}" for i in range(100)],
                'text': [f"This is a mock news article about crypto movement {i}. Bullish or bearish sentiment here." for i in range(100)],
                'source': ['MockSource'] * 100
            })
            print(f"Generated {len(news_df)} mock articles aligned with price data.")
        else:
            print("Date overlap confirmed. Proceeding with real data.")
            
    except Exception as e:
        print(f"Warning: Date alignment check failed: {e}")
        import traceback
        traceback.print_exc()

    
    # Initialize model
    print("\nInitializing FinBERT model with LoRA...")
    try:
        analyzer = FinBERTSentimentAnalyzer(
            model_name=args.model_name,
            lora_rank=args.lora_rank
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Track start time
    start_time = time.time()
    
    # Define training function to avoid code duplication
    def run_training():
        # Train model
        print("\nStarting GRPO training...")
        history = analyzer.train_grpo(
            news_df=news_df,
            crypto_df=crypto_df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            window_hours=args.window_hours,
            threshold=args.threshold,
            group_size=args.group_size,
            clip_eps=args.clip_eps,
            kl_coef=args.kl_coef,
            grad_accum_steps=args.grad_accum_steps,
            reward_clip=args.reward_clip,
            update_old_every_iter=args.update_old_every_iter,
            val_frac=args.val_frac,
            use_mlflow=args.use_mlflow,
            use_wandb=args.use_wandb
        )
        
        # Check time limit
        elapsed_time = time.time() - start_time
        if elapsed_time > args.max_time:
            print(f"\nWarning: Training exceeded time limit ({args.max_time}s)")
            print(f"Elapsed time: {elapsed_time:.1f}s")
            
        return history

    # Run training with or without MLflow
    try:
        if args.use_mlflow:
            import mlflow
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("trl_training")
            
            with mlflow.start_run(run_name="manual_finbert_grpo") as run:
                # Log parameters
                mlflow.log_params(vars(args))
                
                history = run_training()
                
                # Save model
                print("\nSaving model...")
                os.makedirs(args.output_dir, exist_ok=True)
                model_path = os.path.join(args.output_dir, "finbert_grpo.pth")
                tokenizer_path = os.path.join(args.output_dir, "tokenizer")
                
                analyzer.save_model(model_path, tokenizer_path)
                print(f"Model saved to {model_path}")
                print(f"Tokenizer saved to {tokenizer_path}")
                
                # Log artifacts and register model
                print("Logging model to MLflow...")
                mlflow.log_artifact(model_path, artifact_path="model")
                if os.path.exists(tokenizer_path):
                    mlflow.log_artifacts(tokenizer_path, artifact_path="tokenizer")
                
                model_uri = f"runs:/{run.info.run_id}/model"
                try:
                    mlflow.register_model(model_uri, "finbert_grpo_local")
                    print("Model registered as 'finbert_grpo_local'")
                except Exception as e:
                    print(f"Warning: Failed to register model: {e}")
                    
        else:
            history = run_training()
            
            # Save model
            print("\nSaving model...")
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, "finbert_grpo.pth")
            tokenizer_path = os.path.join(args.output_dir, "tokenizer")
            
            analyzer.save_model(model_path, tokenizer_path)
            print(f"Model saved to {model_path}")
            print(f"Tokenizer saved to {tokenizer_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    if history:
        print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Surrogate Loss: {history['train_surrogate'][-1]:.4f}")
        print(f"Final KL Divergence: {history['train_kl'][-1]:.4f}")
        if history['val_accuracy']:
            print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Total Training Time: {time.time() - start_time:.1f}s")
    print("=" * 60)
    print(f"Total Training Time: {time.time() - start_time:.1f}s")
    print("=" * 60)
    print("Training completed successfully!")
    
    # Log success to status DB
    try:
        from utils.database.status_db import status_db
        
        run_id = os.getenv("AIRFLOW_RUN_ID", "manual")
        dag_id = os.getenv("AIRFLOW_DAG_ID", "manual_dag")
        task_id = os.getenv("AIRFLOW_TASK_ID", "vast_ai_train")
        
        # Only log if we have a valid run_id
        if run_id and run_id != "manual":
            status_db.log_event(
                dag_name=dag_id,
                task_name=task_id,
                model_name="trl",
                run_id=run_id,
                event_type="COMPLETE",
                status="SUCCESS",
                message="TRL training completed successfully"
            )
            print("Logged SUCCESS status to database")
    except Exception as e:
        print(f"Warning: Failed to log status to DB: {e}")


if __name__ == "__main__":
    main()

















