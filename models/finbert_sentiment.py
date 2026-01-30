"""
FinBERT Sentiment Analysis Model with GRPO Training
Uses FinBERT (Financial BERT) with GRPO (Group Relative Policy Optimization) 
for training and inference on financial news sentiment
"""

import os
import copy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score

# Optional: PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError as e:
    PEFT_AVAILABLE = False
    print(f"Warning: peft library not available. LoRA training will be disabled. Error: {e}")
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import training utilities
try:
    from trainer.train_utils import annotate_news, log_classification_metrics, save_start_time, load_start_time
    TRAIN_UTILS_AVAILABLE = True
except ImportError:
    TRAIN_UTILS_AVAILABLE = False
    print("Warning: train_utils not available. Training features will be disabled.")

class NewsDataset(Dataset):
    """Dataset for financial news with price change labels"""
    
    def __init__(self, df):
        # Combine title and text
        df = df.copy()
        df['news_text'] = df['title'] + ":\n" + df['text']
        self.texts = df['news_text'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.price_changes = df['price_change'].astype(float).tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
            "price_change": self.price_changes[idx]
        }

class NewsSentimentDataset(Dataset):
    """Dataset for financial news sentiment analysis (for inference)"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer with GRPO training support
    
    Uses LoRA (Low-Rank Adaptation) adapters for efficient fine-tuning.
    LoRA is COMPULSORY - all instances use LoRA adapters.
    
    Supports both:
    - Inference: Using pre-trained or fine-tuned FinBERT model with LoRA
    - Training: GRPO (Group Relative Policy Optimization) with LoRA
    """
    
    def __init__(self, model_name="ProsusAI/finbert", max_length=512, lora_config=None, lora_rank=4):
        """
        Initialize FinBERT analyzer with LoRA adapters (COMPULSORY)
        
        Args:
            model_name: HuggingFace model name (default: "ProsusAI/finbert")
            max_length: Maximum sequence length for tokenization
            lora_config: LoRA configuration (if None, uses default with lora_rank)
            lora_rank: Rank of LoRA matrices (default: 4, used if lora_config is None)
        
        Raises:
            ImportError: If peft library is not available
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft library is REQUIRED for FinBERTSentimentAnalyzer. "
                "LoRA adapters are compulsory. Install with: pip install peft"
            )
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lora = True  # Always True - LoRA is compulsory
        
        print(f"Loading FinBERT model: {model_name}")
        print(f"Using device: {self.device}")
        print("LoRA adapters: COMPULSORY (always enabled)")
        
        # Load tokenizer
        print(f"Step 1: Loading Tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Step 1 Complete: Tokenizer loaded.")
        
        # Load model
        print(f"Step 2: Loading Base Model {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # Negative, Neutral, Positive (maps to Sell, Hold, Buy)
        )
        print("Step 2 Complete: Base model loaded.")
        
        # Apply LoRA adapters (COMPULSORY)
        if lora_config is None:
            lora_config = LoraConfig(
                r=lora_rank,  # rank of the low-rank matrices
                lora_alpha=32,  # scaling factor
                target_modules=["query", "key", "value"],  # layers to apply LoRA
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS"  # sequence classification
            )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA adapters applied to model (rank={lora_config.r})")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path, tokenizer_path=None):
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to saved model
            tokenizer_path: Path to tokenizer (if None, uses model_name)
        """
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()
    
    def save_model(self, model_path, tokenizer_path=None):
        """
        Save model to disk (LoRA adapters are saved)
        
        Args:
            model_path: Path to save model (LoRA adapters will be saved)
            tokenizer_path: Path to save tokenizer (if None, saves to same directory)
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save LoRA adapters (PEFT models can be saved with state_dict)
        torch.save(self.model.state_dict(), model_path)
        if tokenizer_path:
            self.tokenizer.save_pretrained(tokenizer_path)
        print(f"Model with LoRA adapters saved to {model_path}")
    
    @staticmethod
    def compute_normalizer(df, eps=1e-9):
        """Compute normalizer for price changes"""
        mu = np.mean(np.abs(df['price_change'].values))
        return max(mu, eps)
    
    @staticmethod
    def get_label(price_change, threshold):
        """
        Get label from price change
        
        Args:
            price_change: Price change percentage
            threshold: Threshold for classification
        
        Returns:
            0=Sell, 1=Hold, 2=Buy
        """
        if price_change > threshold:
            return 2  # Buy
        elif price_change < -threshold:
            return 0  # Sell
        else:
            return 1  # Hold
    
    @staticmethod
    def log_probs_from_logits(logits, actions):
        """Extract log probabilities for specific actions"""
        logp_all = F.log_softmax(logits, dim=-1)
        return logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    @staticmethod
    def kl_divergence_from_logits(logits_p, logits_q):
        """Compute KL divergence between two logits"""
        p = F.log_softmax(logits_p, dim=-1)
        q = F.log_softmax(logits_q, dim=-1)
        sp = torch.softmax(p, dim=-1)
        kl = torch.sum(sp * (p - q), dim=-1)
        return kl.mean()
    
    @torch.no_grad()
    def get_predictions(self, dataloader):
        """Get predictions from model"""
        pred = []
        self.model.eval()
        for batch in dataloader:
            texts = batch["text"]
            enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            logits = logits.cpu().numpy()
            pred.extend(logits.tolist())
        return pred
    
    def train_grpo(self, news_df, crypto_df, 
                   epochs=10, batch_size=4, lr=2e-5,
                   window_hours=12, threshold=0.005,
                   group_size=4, clip_eps=0.2, kl_coef=0.1,
                   grad_accum_steps=1, reward_clip=None,
                   update_old_every_iter=True, val_frac=0.1,
                   use_mlflow=False, use_wandb=False):
        """
        Train FinBERT using GRPO (Group Relative Policy Optimization)
        
        Args:
            news_df: DataFrame with news articles (must have 'title', 'text', 'date')
            crypto_df: DataFrame with price data (must have 'open_time', 'close')
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            window_hours: Hours after news to measure price change
            threshold: Threshold for Buy/Sell classification
            group_size: Group size for GRPO
            clip_eps: Clipping epsilon for PPO
            kl_coef: KL divergence coefficient
            grad_accum_steps: Gradient accumulation steps
            reward_clip: Clip rewards to this value (if None, no clipping)
            update_old_every_iter: Update old policy every iteration
            val_frac: Validation fraction
            use_mlflow: Whether to log to MLflow
            use_wandb: Whether to log to WandB
        
        Returns:
            Training history dictionary
        """
        if not TRAIN_UTILS_AVAILABLE:
            raise ImportError("train_utils not available. Cannot train GRPO model.")
        
        print("\n" + "="*60)
        print("Training FinBERT with GRPO")
        print("="*60)
        
        # Annotate news with price changes
        print("Annotating news with price changes...")
        df_annotated = annotate_news(crypto_df, news_df.copy(), 
                                     window_hours=window_hours, threshold=threshold)
        
        # Compute normalizer
        normalizer = self.compute_normalizer(df_annotated)
        print(f"Price change normalizer: {normalizer:.6f}")
        
        # Create dataset
        ds = NewsDataset(df_annotated)
        print(f"Total dataset size: {len(ds)}")
        print(f"Label distribution:\n{df_annotated['label'].value_counts()}")
        
        # Split into train/val
        n_total = len(ds)
        n_val = int(n_total * val_frac)
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(ds, [n_train, n_val])
        
        print(f"Train samples: {n_train}, Val samples: {n_val}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Create reference model (frozen)
        reference = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        ).to(self.device)
        for p in reference.parameters():
            p.requires_grad = False
        reference.eval()
        
        # Create old policy (theta_old)
        theta_old = copy.deepcopy(self.model).to(self.device)
        for p in theta_old.parameters():
            p.requires_grad = False
        theta_old.eval()
        
        # Training history
        history = {
            'train_loss': [],
            'train_surrogate': [],
            'train_kl': [],
            'val_accuracy': []
        }
        
        # Set model to training mode
        self.model.train()
        self.model.enable_adapter_layers()  # LoRA is always enabled
        
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Track start time
        if TRAIN_UTILS_AVAILABLE:
            save_start_time()
            start_time = load_start_time()
        
        # Training loop
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Validation
            val_preds = self.get_predictions(val_loader)
            val_labels = [item['label'] for item in val_dataset]
            val_preds = np.array(val_preds).argmax(axis=1).tolist()
            val_accuracy = accuracy_score(val_labels, val_preds)
            history['val_accuracy'].append(val_accuracy)
            
            if TRAIN_UTILS_AVAILABLE:
                try:
                    log_classification_metrics(val_preds, val_labels, name="finbert_val", 
                                             step=epoch, use_mlflow=use_mlflow, use_wandb=use_wandb)
                except Exception as e:
                    print(f"Warning: Failed to log metrics: {e}")
            
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Training
            epoch_loss = 0
            epoch_surrogate = 0
            epoch_kl = 0
            
            optimizer.zero_grad()
            
            for batch_idx, batch_items in enumerate(tqdm(train_loader, desc=f"Training")):
                texts = batch_items["text"]
                true_labels = batch_items["label"].to(self.device)
                price_changes = batch_items["price_change"].numpy()
                
                # Encode with old policy (theta_old)
                with torch.no_grad():
                    enc_old = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    logits_old = theta_old(**enc_old).logits
                
                # Sample actions from old policy
                all_texts, all_actions, all_old_logps, all_rewards, all_group_ids = [], [], [], [], []
                probs_old = F.softmax(logits_old, dim=-1)
                
                for i in range(len(texts)):
                    dist = torch.distributions.Categorical(probs_old[i])
                    sampled_actions = dist.sample((group_size,))
                    logp_all = F.log_softmax(logits_old[i], dim=-1)
                    sampled_logps = logp_all[sampled_actions]
                    
                    pc = price_changes[i]
                    for g in range(group_size):
                        action = int(sampled_actions[g].item())
                        old_logp = sampled_logps[g]
                        reward_mag = abs(pc) / normalizer
                        if reward_clip is not None:
                            reward_mag = np.clip(reward_mag, -reward_clip, reward_clip)
                        reward = reward_mag if action == true_labels[i].item() else -reward_mag
                        
                        all_texts.append(texts[i])
                        all_actions.append(action)
                        all_old_logps.append(old_logp)
                        all_rewards.append(float(reward))
                        all_group_ids.append(i)
                
                # Compute new logits and log_probs
                enc_new = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits_new = self.model(**enc_new).logits
                actions_tensor = torch.tensor(all_actions, dtype=torch.long, device=self.device)
                new_logps = self.log_probs_from_logits(logits_new, actions_tensor)
                old_logps_tensor = torch.stack(all_old_logps).to(self.device)
                
                # Compute advantages (group-relative)
                rewards_np = np.array(all_rewards)
                group_ids = np.array(all_group_ids)
                adv = np.zeros_like(rewards_np)
                for gid in np.unique(group_ids):
                    mask = group_ids == gid
                    adv[mask] = rewards_np[mask] - rewards_np[mask].mean()
                advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
                
                # PPO loss
                logratio = new_logps - old_logps_tensor
                ratio = torch.exp(logratio)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
                surrogate = -torch.min(unclipped, clipped)
                loss_surrogate = surrogate.mean()
                
                # KL divergence term
                kl_term = torch.tensor(0.0, device=self.device)
                if kl_coef > 0.0:
                    with torch.no_grad():
                        enc_ref = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                        logits_ref = reference(**enc_ref).logits
                    logits_pol = self.model(**enc_ref).logits
                    kl_term = self.kl_divergence_from_logits(logits_pol, logits_ref)
                
                loss = loss_surrogate + kl_coef * kl_term
                
                epoch_loss += loss.item()
                epoch_surrogate += loss_surrogate.item()
                epoch_kl += kl_term.item()
                
                # Backward pass
                (loss / grad_accum_steps).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if update_old_every_iter:
                        theta_old.load_state_dict(self.model.state_dict())
            
            # Final gradient step if needed
            if len(train_loader) % grad_accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            avg_surrogate = epoch_surrogate / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)
            
            history['train_loss'].append(avg_loss)
            history['train_surrogate'].append(avg_surrogate)
            history['train_kl'].append(avg_kl)
            
            print(f"Loss: {avg_loss:.4f}, Surrogate: {avg_surrogate:.4f}, KL: {avg_kl:.4f}")
        
        # Set model back to eval mode
        self.model.eval()
        
        print("\nTraining completed!")
        return history
    
    def predict_sentiment(self, texts, batch_size=16):
        """Predict sentiment for a list of texts"""
        print(f"Predicting sentiment for {len(texts)} texts...")
        
        # Create dataset
        dataset = NewsSentimentDataset(
            texts=texts,
            labels=[0] * len(texts),  # Dummy labels for prediction
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting sentiment"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions and probabilities
                batch_predictions = torch.argmax(logits, dim=1)
                batch_probabilities = torch.softmax(logits, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probabilities.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def predict_three_class(self, texts, batch_size=16):
        """
        Predict 3-class probabilities [Sell, Hold, Buy] from news sentiment
        
        Args:
            texts: List of news texts to analyze
            batch_size: Batch size for processing
        
        Returns:
            Array of shape (n_samples, 3) with [Sell, Hold, Buy] probabilities
            Format: [[prob_sell, prob_hold, prob_buy], ...]
            Use np.argmax(probs, axis=1) to get decisions (0=Sell, 1=Hold, 2=Buy)
        """
        _, probabilities = self.predict_sentiment(texts, batch_size=batch_size)
        return np.array(probabilities)
    
    def analyze_news_sentiment(self, news_df):
        """Analyze sentiment of news data"""
        print("Analyzing news sentiment with FinBERT...")
        
        # Combine title and text
        news_df = news_df.copy()
        news_df['combined_text'] = news_df['title'] + ". " + news_df['text']
        
        # Get predictions
        texts = news_df['combined_text'].tolist()
        predictions, probabilities = self.predict_sentiment(texts)
        
        # Add results to dataframe
        news_df['finbert_prediction'] = predictions
        news_df['finbert_confidence'] = np.max(probabilities, axis=1)
        news_df['finbert_negative_prob'] = probabilities[:, 0]
        news_df['finbert_neutral_prob'] = probabilities[:, 1]
        news_df['finbert_positive_prob'] = probabilities[:, 2]
        
        return news_df
    
    def get_daily_sentiment_features(self, news_df):
        """Extract daily sentiment features from news data"""
        print("Extracting daily sentiment features...")
        
        # Analyze sentiment
        sentiment_df = self.analyze_news_sentiment(news_df)
        
        # Ensure date column exists and is in proper format
        if 'date' not in sentiment_df.columns:
            raise ValueError("No 'date' column found in news data. Required for daily aggregation.")
        
        # Convert date to datetime and extract date part for grouping
        if sentiment_df['date'].dtype == 'object':
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce', utc=True)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        
        # Group by date and calculate daily sentiment metrics
        daily_sentiment = sentiment_df.groupby('date').agg({
            'finbert_prediction': ['mean', 'std', 'count'],
            'finbert_confidence': 'mean',
            'finbert_negative_prob': 'mean',
            'finbert_neutral_prob': 'mean',
            'finbert_positive_prob': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date',
            'sentiment_mean',
            'sentiment_std', 
            'news_count',
            'sentiment_confidence',
            'negative_sentiment',
            'neutral_sentiment',
            'positive_sentiment'
        ]
        
        # Fill NaN values
        daily_sentiment = daily_sentiment.fillna(0)
        
        return daily_sentiment

def main():
    """Main function to demonstrate FinBERT sentiment analysis"""
    print("=" * 60)
    print("FinBERT Sentiment Analysis")
    print("=" * 60)
    
    # Load news data
    news_path = "data/articles.csv"
    if not os.path.exists(news_path):
        print(f"Error: News data not found at {news_path}")
        print("Please create news dataset using create_news_dataset.py")
        return
    
    news_df = pd.read_csv(news_path)
    print(f"Loaded {len(news_df)} news articles")
    
    # Initialize FinBERT analyzer (LoRA is compulsory)
    try:
        analyzer = FinBERTSentimentAnalyzer()  # LoRA adapters automatically applied
        
        # Analyze sentiment
        daily_sentiment = analyzer.get_daily_sentiment_features(news_df)
        
        # Save results
        os.makedirs("results", exist_ok=True)
        daily_sentiment.to_csv("results/daily_sentiment_features.csv", index=False)
        print("Daily sentiment features saved to results/daily_sentiment_features.csv")
        
        # Print sample results
        print("\nSample daily sentiment features:")
        print(daily_sentiment.head())
        
        print("\nFinBERT sentiment analysis completed successfully!")
        
    except Exception as e:
        print(f"Error running FinBERT: {e}")
        print("This might be due to missing transformers library or model download issues.")
        print("FinBERT requires internet connection to download the pre-trained model.")

if __name__ == "__main__":
    main()
