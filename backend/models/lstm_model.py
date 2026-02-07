"""
LSTM Temporal Prediction Model
Predicts slope instability based on time-series sensor data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorDataset(Dataset):
    """PyTorch dataset for sensor time series"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input sequences (num_samples, sequence_length, num_features)
            y: Labels (num_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM model for temporal sequence prediction"""
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: Number of input features (sensors)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary classification output
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Prediction (batch_size, 1)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size*2)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights shape: (batch_size, sequence_length, 1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector shape: (batch_size, hidden_size*2)
        
        # Classification
        output = self.fc(context_vector)
        
        return output


class LSTMTrainer:
    """Trainer for LSTM model"""
    
    def __init__(
        self,
        model: LSTMModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()  # Binary cross-entropy
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        logger.info(f"Model initialized on {device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy()[:, 0])
                all_labels.extend(batch_y.cpu().numpy()[:, 0])
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Binary predictions (threshold = 0.5)
        binary_preds = (all_predictions > 0.5).astype(int)
        
        # Metrics
        tp = np.sum((binary_preds == 1) & (all_labels == 1))
        fp = np.sum((binary_preds == 1) & (all_labels == 0))
        tn = np.sum((binary_preds == 0) & (all_labels == 0))
        fn = np.sum((binary_preds == 0) & (all_labels == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        checkpoint_dir: str = "models/checkpoints"
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum epochs
            early_stopping_patience: Stop if no improvement
            checkpoint_dir: Where to save checkpoints
            
        Returns:
            Training history
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1_score']:.4f}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = Path(checkpoint_dir) / "lstm_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def save_model(self, filepath: str):
        """Save model"""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data.synthetic_generator import SyntheticDataGenerator
    from data.data_preprocessor import DataPreprocessor
    from torch.utils.data import random_split
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(seed=42)
    
    # Create multiple scenarios
    scenarios = [
        generator.generate_normal_scenario(168),
        generator.generate_rainfall_scenario(72),
        generator.generate_seismic_scenario(48),
    ]
    
    # Preprocess
    preprocessor = DataPreprocessor()
    all_X, all_y = [], []
    
    for scenario in scenarios:
        data = preprocessor.prepare_lstm_data(scenario, window_size=60, prediction_horizon=60)
        all_X.append(data.features)
        all_y.append(data.labels)
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"Dataset: {X.shape}, Labels: {y.shape}")
    print(f"Positive rate: {y.mean():.2%}")
    
    # Create dataset
    dataset = SensorDataset(X, y)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = LSTMModel(input_size=4, hidden_size=128, num_layers=3)
    trainer = LSTMTrainer(model, learning_rate=0.001)
    
    # Train
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=50,
        early_stopping_patience=10
    )
