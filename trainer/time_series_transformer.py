"""
Time Series Transformer Model
Uses Transformer architecture for cryptocurrency price prediction
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

# Import training utilities
try:
    from trainer.train_utils import preprocess_sequences, log_classification_metrics, save_start_time, load_start_time
    TRAIN_UTILS_AVAILABLE = True
except ImportError:
    TRAIN_UTILS_AVAILABLE = False
    print("Warning: train_utils not available. Some features will be disabled.")

class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(self.targets[idx], dtype=torch.long)

class TSTBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super(TSTBlock, self).__init__()
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Pre-norm architecture: normalize before attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out  # Residual connection
        
        # Pre-norm architecture: normalize before FFN
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out  # Residual connection
        
        return x


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series prediction (TSTClassifier)"""
    
    def __init__(self, input_dim, hidden_dim=32, num_heads=2, ff_dim=64, num_layers=1, dropout=0.1, num_classes=3):
        super(TimeSeriesTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers (TSTBlocks)
        self.layers = nn.ModuleList([
            TSTBlock(hidden_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Input projection: (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        x = self.input_proj(x)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Use last timestep for classification (as per specifications)
        out = self.classifier(x[:, -1, :])  # (batch_size, num_classes)
        
        return out

class TimeSeriesTransformerTrainer:
    """Trainer for Time Series Transformer"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        # Force CPU to avoid CUDA kernel errors on Vast.ai with incompatible drivers/images
        self.device = 'cpu' 
        self.model.to(self.device)
        
        # Loss and optimizer - CrossEntropyLoss for 3-class classification
        self.criterion = nn.CrossEntropyLoss()
        # Learning rate as per specifications: 1e-3
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
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
    
    def create_sequences(self, data, close_prices, sequence_length=30):
        """Create sequences for time series prediction with 3-class labels"""
        
        sequences = []
        targets = []
        
        # Threshold for creating 3-class labels
        threshold = 0.00015  # 0.015% threshold for 1-minute data
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
            
            # Calculate price change percentage
            if i < len(close_prices):
                price_change_pct = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                # Create 3-class label: 0=Sell, 1=Hold, 2=Buy
                label = self._label_price_change(price_change_pct, threshold=threshold)
                targets.append(label)
            else:
                targets.append(1)  # Default to Hold if no future price available
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, crypto_df, sequence_length=30, test_size=0.2):
        """Prepare data for training"""
        print("Preparing time series data...")
        
        # Select features for time series (7 features as per specifications)
        # Try to use taker_base and taker_quote if available, otherwise use price_change
        required_features = ['open', 'high', 'low', 'close', 'volume']
        optional_features = ['taker_base', 'taker_quote']
        
        available_features = required_features.copy()
        for feat in optional_features:
            if feat in crypto_df.columns:
                available_features.append(feat)
        
        # If taker_base/taker_quote not available, use price_change as fallback
        if 'taker_base' not in crypto_df.columns or 'taker_quote' not in crypto_df.columns:
            if 'price_change' in crypto_df.columns:
                available_features.append('price_change')
            else:
                # Calculate price_change if not present
                crypto_df['price_change'] = crypto_df['close'].pct_change()
                available_features.append('price_change')
        
        # Ensure we have exactly 7 features (pad with zeros if needed)
        while len(available_features) < 7:
            available_features.append('price_change')  # Duplicate if needed
        
        # Take first 7 features (or whatever we have)
        features = available_features[:7]
        
        # KEY FIX: Check if selected features are actually populated. 
        # If a feature column is all NaNs (e.g. taker_base), drop the feature, not the rows.
        # This prevents "0 samples" error.
        valid_features = []
        for feat in features:
            if crypto_df[feat].isna().all():
                print(f"WARNING: Feature '{feat}' is completely empty. Dropping feature.")
            else:
                valid_features.append(feat)
        
        if len(valid_features) == 0:
            raise ValueError("All selected features are empty! Cannot train.")
            
        features = valid_features
        print(f"Using valid features: {features}")
        initial_len = len(crypto_df)
        crypto_df = crypto_df.dropna(subset=features)
        if len(crypto_df) < initial_len:
            print(f"Dropped {initial_len - len(crypto_df)} rows with missing values")
            
        data = crypto_df[features].values
        
        # Get close prices before scaling (needed for label creation)
        close_prices = crypto_df['close'].values
        
        # Normalize data
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences with 3-class labels
        sequences, targets = self.create_sequences(data_scaled, close_prices, sequence_length)
        
        print(f"Target distribution: {np.bincount(targets)}")
        
        # Split data
        split_idx = int(len(sequences) * (1 - test_size))
        
        X_train = sequences[:split_idx]
        y_train = targets[:split_idx]
        X_test = sequences[split_idx:]
        y_test = targets[split_idx:]
        
        print(f"Created {len(sequences)} sequences")
        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        # Use tqdm for progress bar if available, otherwise silent
        try:
            loader = tqdm(train_loader, desc='  Training', leave=False, ncols=80)
        except:
            loader = train_loader
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar if using tqdm
            if hasattr(loader, 'set_postfix'):
                loader.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with 3-model baseline versioning strategy.
        
        Versioning strategy:
        - v1 = baseline (first ever trained model, never overwritten)
        - v2 = previous model (stored before each new training run)
        - v3 = latest model (model produced by the current training run)
        """
        # ============================================================
        # STARTUP: Load existing model (v3 → v2 → new baseline)
        # ============================================================
        base_dir = Path("models/tst")
        v1_dir = base_dir / "v1"
        v2_dir = base_dir / "v2"
        v3_dir = base_dir / "v3"
        
        # Ensure directories exist
        v1_dir.mkdir(parents=True, exist_ok=True)
        v2_dir.mkdir(parents=True, exist_ok=True)
        v3_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        v1_model_path = v1_dir / "model.pth"
        v1_scaler_path = v1_dir / "scaler.pkl"
        v2_model_path = v2_dir / "model.pth"
        v2_scaler_path = v2_dir / "scaler.pkl"
        v3_model_path = v3_dir / "model.pth"
        v3_scaler_path = v3_dir / "scaler.pkl"
        
        loaded_version = None
        
        # Check if v3 exists → load it for continued training
        if v3_model_path.exists() and v3_scaler_path.exists():
            try:
                print("[LOAD] Loading v3 latest model for continued training")
                self.model.load_state_dict(torch.load(str(v3_model_path), map_location=self.device))
                if hasattr(self, 'scaler'):
                    self.scaler = joblib.load(str(v3_scaler_path))
                loaded_version = "v3"
                print(f"[LOAD] Successfully loaded v3 latest model from {v3_model_path}")
            except Exception as e:
                print(f"[LOAD] Failed to load v3: {e}. Trying v2...")
                loaded_version = None
        
        # Else if v2 exists → load v2
        if loaded_version is None and v2_model_path.exists() and v2_scaler_path.exists():
            try:
                print("[LOAD] Loading v2 previous model for continued training")
                self.model.load_state_dict(torch.load(str(v2_model_path), map_location=self.device))
                if hasattr(self, 'scaler'):
                    self.scaler = joblib.load(str(v2_scaler_path))
                loaded_version = "v2"
                print(f"[LOAD] Successfully loaded v2 previous model from {v2_model_path}")
            except Exception as e:
                print(f"[LOAD] Failed to load v2: {e}. Starting fresh training...")
                loaded_version = None
        
        # Else → start fresh training (new baseline will be created)
        if loaded_version is None:
            print("[LOAD] No versions found, starting fresh training (will create v1 baseline)")
        
        print("Training Time Series Transformer...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("-" * 60)
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        # Temporary path for best model during training
        temp_best_model_path = Path('models/temp_best_tst_model.pth')
        temp_best_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            # Train with progress bar
            print(f'Epoch {epoch+1}/{epochs}...', end=' ', flush=True)
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model to temporary location
                torch.save(self.model.state_dict(), str(temp_best_model_path))
                print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} ✓ (best)')
            else:
                patience_counter += 1
                print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} (patience: {patience_counter}/{patience})')
            
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch+1}')
                break
        
        # Load best model if it was saved (if training improved at least once)
        if temp_best_model_path.exists():
            print(f"[FINAL] Loading best model from {temp_best_model_path}")
            self.model.load_state_dict(torch.load(str(temp_best_model_path), map_location=self.device))
        else:
            print("[WARNING] No improvement during training (no best model saved). Using final weights.")
        
        # ============================================================
        # SAVE: Before saving new model, move v3 → v2, save as v3
        # ============================================================
        print("\n" + "="*60)
        print("Saving Time Series Transformer model with 3-model baseline versioning")
        print("="*60)
        
        # Before saving a new trained model:
        # Move the existing v3 → v2 (if v3 exists)
        if v3_model_path.exists() and v3_scaler_path.exists():
            print("[SAVE] Moving old v3 to v2")
            try:
                # Remove old v2 if it exists
                if v2_model_path.exists():
                    v2_model_path.unlink()
                if v2_scaler_path.exists():
                    v2_scaler_path.unlink()
                # Move v3 to v2
                v3_model_path.rename(v2_model_path)
                v3_scaler_path.rename(v2_scaler_path)
                print(f"[SAVE] Successfully moved v3 → v2")
            except Exception as e:
                print(f"[SAVE] Warning: Failed to move v3 → v2: {e}. Continuing...")
        
        # Keep v1 unchanged (v1 is only created once, never overwritten)
        # Create v1 baseline if it doesn't exist (first training run)
        if not v1_model_path.exists() or not v1_scaler_path.exists():
            print("[SAVE] Creating v1 baseline (first training run)")
            try:
                torch.save(self.model.state_dict(), str(v1_model_path))
                if hasattr(self, 'scaler'):
                    joblib.dump(self.scaler, str(v1_scaler_path))
                print(f"[SAVE] Successfully created v1 baseline at {v1_dir}")
            except Exception as e:
                print(f"[SAVE] Warning: Failed to create v1 baseline: {e}")
        
        # Save the new trained model as v3
        print("[SAVE] Saving new v3 latest model")
        try:
            torch.save(self.model.state_dict(), str(v3_model_path))
            if hasattr(self, 'scaler'):
                joblib.dump(self.scaler, str(v3_scaler_path))
            print(f"[SAVE] Successfully saved new v3 latest model at {v3_dir}")
        except Exception as e:
            print(f"[SAVE] ERROR: Failed to save v3 model: {e}")
            raise
        
        # Clean up temporary best model file
        if temp_best_model_path.exists():
            temp_best_model_path.unlink()
        
        # Clean up legacy best model file if it exists
        legacy_best_path = Path('models/best_tst_model.pth')
        if legacy_best_path.exists():
            try:
                legacy_best_path.unlink()
                print("[CLEANUP] Removed legacy best_tst_model.pth")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not remove legacy file: {e}")
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Total epochs trained: {len(self.train_losses)}")
    
    def predict(self, X_test):
        """Make class predictions - returns class indices"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), 32):  # Batch processing
                batch = torch.FloatTensor(X_test[i:i+32]).to(self.device)
                logits = self.model(batch)
                class_preds = torch.argmax(logits, dim=1)
                predictions.extend(class_preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """Make probability predictions - returns probabilities for all 3 classes"""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), 32):  # Batch processing
                batch = torch.FloatTensor(X_test[i:i+32]).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def predict_three_class(self, X_test, current_prices=None):
        """
        Make 3-class predictions [Sell, Hold, Buy] - returns probabilities
        
        The model is trained as a 3-class classifier, so it directly outputs
        probabilities for all three classes.
        
        Args:
            X_test: Input sequences
            current_prices: Unused parameter (kept for API compatibility)
        
        Returns:
            Array of shape (n_samples, 3) with [Sell, Hold, Buy] probabilities
            Format: [[prob_sell, prob_hold, prob_buy], ...]
            Use np.argmax(probs, axis=1) to get decisions (0=Sell, 1=Hold, 2=Buy)
        """
        # Model already outputs 3-class probabilities directly
        return self.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.predict(X_test)
        
        # Calculate classification metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Test Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        report = classification_report(y_test, predictions, target_names=['Sell', 'Hold', 'Buy'])
        print(report)
        
        # Log metrics using train_utils if available
        if TRAIN_UTILS_AVAILABLE:
            try:
                log_classification_metrics(predictions, y_test, name="tst_val", 
                                         class_labels=['0', '1', '2'], 
                                         use_mlflow=False, use_wandb=False)  # Disable by default
            except Exception as e:
                print(f"Warning: Failed to log metrics: {e}")
        
        return predictions, {'accuracy': accuracy}
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-20:], label='Training Loss (Last 20)')
        plt.plot(self.val_losses[-20:], label='Validation Loss (Last 20)')
        plt.title('Training History (Last 20 Epochs)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/tst_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training history plot saved to results/tst_training_history.png")

def main():
    """Main function to demonstrate Time Series Transformer"""
    print("=" * 60)
    print("Time Series Transformer Training")
    print("=" * 60)
    
    # Load crypto data
    crypto_path = "data/btcusdt.csv"
    if not os.path.exists(crypto_path):
        print(f"Error: Crypto data not found at {crypto_path}")
        print("Please fetch price data using data_fetcher.py")
        print("Example: python data_fetcher.py --symbol BTCUSDT --interval 1h --start-date 2024-01-01")
        return
    
    crypto_df = pd.read_csv(crypto_path)
    print(f"Loaded {len(crypto_df)} crypto data points")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with reduced parameters for faster CPU training
    input_dim = 7  # open, high, low, close, volume, taker_base, taker_quote
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        hidden_dim=32,      # Reduced from 64 for faster CPU training
        num_heads=2,        # Keep at 2
        ff_dim=64,          # Reduced from 128 for faster CPU training
        num_layers=1,       # Reduced from 2 for faster CPU training
        dropout=0.1,
        num_classes=3
    )
    
    # Initialize trainer
    trainer = TimeSeriesTransformerTrainer(model, device)
    
    # Prepare data with reduced sequence length for faster CPU training
    X_train, y_train, X_test, y_test = trainer.prepare_data(
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
    trainer.train(X_train, y_train, X_val, y_val, epochs=2, batch_size=16)
    
    # Evaluate model
    predictions, metrics = trainer.evaluate(X_test, y_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Model is already saved with versioning in train() method
    # No need to save again here - versioning handles it automatically
    
    print("\nTime Series Transformer training completed!")
    print("Model saved with versioning to models/tst/v1/, v2/, v3/")

if __name__ == "__main__":
    main()


