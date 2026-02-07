"""
CNN Spatial Analysis Model
Analyzes spatial stress patterns for slope instability prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialDataset(Dataset):
    """PyTorch dataset for spatial grids"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Spatial grids (num_samples, height, width, channels)
            y: Labels (num_samples,)
        """
        # PyTorch expects (N, C, H, W) format
        if len(X.shape) == 3:
            X = X[:, np.newaxis, :, :]  # Add channel dimension
        elif len(X.shape) == 4:
            X = np.transpose(X, (0, 3, 1, 2))  # (N, H, W, C) → (N, C, H, W)
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    """Residual block for CNN"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class CNNModel(nn.Module):
    """CNN model for spatial analysis"""
    
    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 64,
        dropout: float = 0.3
    ):
        """
        Args:
            in_channels: Number of input channels
            num_filters: Base number of filters
            dropout: Dropout rate
        """
        super(CNNModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(num_filters, num_filters, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(num_filters, num_filters*2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(num_filters*2, num_filters*4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(num_filters*4, num_filters*8, num_blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters*8, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a layer of residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Prediction (batch, 1)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class CNNTrainer:
    """Trainer for CNN model"""
    
    def __init__(
        self,
        model: CNNModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        logger.info(f"CNN Model initialized on {device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
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
        binary_preds = (all_predictions > 0.5).astype(int)
        
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
            "f1_score": f1
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
        """Full training loop"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        
        logger.info(f"Starting CNN training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}"
                )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, Path(checkpoint_dir) / "cnn_best.pth")
                logger.info(f"Saved best CNN model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history


if __name__ == "__main__":
    # Example usage with stress field data
    from digital_twin.stress_simulator import StressSimulator
    from data.dem_ingestion import DEMIngestion
    
    # Generate terrain and stress fields
    dem_ingestor = DEMIngestion()
    stress_sim = StressSimulator()
    
    # Create dataset
    X_spatial = []
    y_labels = []
    
    for i in range(100):
        dem = dem_ingestor.generate_synthetic_dem(128, 128, slope_angle=30+i*0.2)
        slope = dem_ingestor.calculate_slope(dem)
        stress_field = stress_sim.calculate_stress_field(dem.elevation, slope)
        risk = stress_sim.get_risk_map(stress_field)
        
        X_spatial.append(risk)
        y_labels.append(1 if risk.max() > 60 else 0)
    
    X = np.array(X_spatial)
    y = np.array(y_labels)
    
    print(f"Spatial dataset: {X.shape}, Labels: {y.shape}")
    
    # Create model and train
    dataset = SpatialDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    model = CNNModel(in_channels=1, num_filters=32)
    trainer = CNNTrainer(model, learning_rate=0.001)
    history = trainer.train(train_loader, val_loader, num_epochs=30)
