"""
Data Preprocessing Pipeline
Handles normalization, feature engineering, and data preparation for ML models
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container for preprocessed data"""
    features: np.ndarray
    labels: np.ndarray
    feature_names: list
    scaler: Optional[object] = None
    metadata: Optional[Dict] = None


class DataPreprocessor:
    """Preprocessing pipeline for sensor data"""
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Args:
            scaling_method: 'standard' (z-score) or 'minmax' (0-1 range)
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        self.feature_names = []
        
    def create_sliding_windows(
        self,
        data: np.ndarray,
        window_size: int = 60,
        step_size: int = 1,
        prediction_horizon: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows for time-series prediction
        
        Args:
            data: Input time series (samples, features)
            window_size: Length of lookback window (timesteps)
            step_size: Step between windows
            prediction_horizon: How far ahead to predict (timesteps)
            
        Returns:
            X: Input windows (num_windows, window_size, features)
            y: Target values (num_windows,)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        num_samples, num_features = data.shape
        num_windows = (num_samples - window_size - prediction_horizon) // step_size + 1
        
        X = np.zeros((num_windows, window_size, num_features))
        y = np.zeros(num_windows)
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            target_idx = end_idx + prediction_horizon - 1
            
            X[i] = data[start_idx:end_idx]
            
            # Target: binary classification (will instability occur?)
            # For now, use simple threshold on future values
            # In real scenario, this would be based on actual failure events
            future_window = data[end_idx:target_idx+1]
            
            # Check if any significant change occurs (placeholder logic)
            threshold = np.std(data) * 2
            y[i] = 1.0 if np.max(np.abs(future_window - data[end_idx-1])) > threshold else 0.0
        
        logger.info(f"Created {num_windows} windows: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def extract_statistical_features(
        self,
        data: np.ndarray,
        window_size: int = 60
    ) -> np.ndarray:
        """
        Extract statistical features from time series windows
        
        Args:
            data: Input time series (samples, features)
            window_size: Window size for feature extraction
            
        Returns:
            Feature matrix (num_windows, num_statistical_features)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        num_samples, num_features = data.shape
        num_windows = num_samples - window_size + 1
        
        # Features per window: mean, std, min, max, range, slope, skewness
        features_per_channel = 7
        feature_matrix = np.zeros((num_windows, num_features * features_per_channel))
        
        for i in range(num_windows):
            window = data[i:i+window_size]
            
            for feat_idx in range(num_features):
                channel_data = window[:, feat_idx]
                
                base_idx = feat_idx * features_per_channel
                
                # Statistical features
                feature_matrix[i, base_idx] = np.mean(channel_data)
                feature_matrix[i, base_idx + 1] = np.std(channel_data)
                feature_matrix[i, base_idx + 2] = np.min(channel_data)
                feature_matrix[i, base_idx + 3] = np.max(channel_data)
                feature_matrix[i, base_idx + 4] = np.ptp(channel_data)  # Range
                
                # Trend (linear slope)
                x = np.arange(window_size)
                slope = np.polyfit(x, channel_data, 1)[0]
                feature_matrix[i, base_idx + 5] = slope
                
                # Skewness (asymmetry measure)
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                if std_val > 0:
                    skewness = np.mean(((channel_data - mean_val) / std_val) ** 3)
                else:
                    skewness = 0.0
                feature_matrix[i, base_idx + 6] = skewness
        
        logger.info(f"Extracted statistical features: {feature_matrix.shape}")
        
        return feature_matrix
    
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize data using configured scaling method
        
        Args:
            data: Input data
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            Normalized data
        """
        original_shape = data.shape
        
        # Reshape for scaler (expects 2D)
        if len(data.shape) > 2:
            data_2d = data.reshape(-1, data.shape[-1])
        else:
            data_2d = data
            
        if fit:
            normalized = self.scaler.fit_transform(data_2d)
            logger.info(f"Fitted scaler on data with shape {data_2d.shape}")
        else:
            normalized = self.scaler.transform(data_2d)
            
        # Reshape back
        if len(original_shape) > 2:
            normalized = normalized.reshape(original_shape)
            
        return normalized
    
    def combine_sensor_data(
        self,
        geophone: np.ndarray,
        inclinometer: np.ndarray,
        soil_moisture: np.ndarray,
        acoustic: np.ndarray
    ) -> np.ndarray:
        """
        Combine multiple sensor streams into single feature matrix
        
        Args:
            geophone: Vibration data
            inclinometer: Tilt data
            soil_moisture: Moisture data
            acoustic: Acoustic emission data
            
        Returns:
            Combined feature matrix (samples, 4)
        """
        # Ensure all have same length (take minimum)
        min_length = min(len(geophone), len(inclinometer), len(soil_moisture), len(acoustic))
        
        combined = np.column_stack([
            geophone[:min_length],
            inclinometer[:min_length],
            soil_moisture[:min_length],
            acoustic[:min_length]
        ])
        
        self.feature_names = ["geophone", "inclinometer", "soil_moisture", "acoustic"]
        
        logger.info(f"Combined sensor data: {combined.shape}")
        
        return combined
    
    def prepare_lstm_data(
        self,
        sensor_data: Dict[str, np.ndarray],
        window_size: int = 60,
        prediction_horizon: int = 60,
        normalize: bool = True
    ) -> PreprocessedData:
        """
        Full preprocessing pipeline for LSTM model
        
        Args:
            sensor_data: Dictionary of sensor arrays
            window_size: Lookback window size
            prediction_horizon: Prediction horizon
            normalize: Whether to normalize features
            
        Returns:
            PreprocessedData object
        """
        # Combine sensors
        combined = self.combine_sensor_data(
            sensor_data["geophone"],
            sensor_data["inclinometer"],
            sensor_data["soil_moisture"],
            sensor_data["acoustic"]
        )
        
        # Normalize
        if normalize:
            combined = self.normalize_data(combined, fit=True)
        
        # Create windows
        X, y = self.create_sliding_windows(
            combined,
            window_size=window_size,
            step_size=1,
            prediction_horizon=prediction_horizon
        )
        
        return PreprocessedData(
            features=X,
            labels=y,
            feature_names=self.feature_names,
            scaler=self.scaler if normalize else None,
            metadata={
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "scaling_method": self.scaling_method
            }
        )
    
    def prepare_cnn_data(
        self,
        spatial_grid: np.ndarray,
        normalize: bool = True
    ) -> PreprocessedData:
        """
        Prepare spatial grid data for CNN model
        
        Args:
            spatial_grid: 3D array (samples, height, width) or 4D (samples, height, width, channels)
            normalize: Whether to normalize
            
        Returns:
            PreprocessedData object
        """
        if normalize:
            # Normalize each channel independently
            if len(spatial_grid.shape) == 4:
                num_samples, h, w, c = spatial_grid.shape
                normalized = np.zeros_like(spatial_grid)
                
                for ch in range(c):
                    channel_data = spatial_grid[:, :, :, ch].reshape(num_samples, -1)
                    channel_normalized = self.normalize_data(channel_data, fit=True)
                    normalized[:, :, :, ch] = channel_normalized.reshape(num_samples, h, w)
            else:
                num_samples = spatial_grid.shape[0]
                flat = spatial_grid.reshape(num_samples, -1)
                normalized_flat = self.normalize_data(flat, fit=True)
                normalized = normalized_flat.reshape(spatial_grid.shape)
        else:
            normalized = spatial_grid
        
        # For CNN, labels would come from separate source
        # Placeholder: create dummy labels
        labels = np.zeros(len(normalized))
        
        return PreprocessedData(
            features=normalized,
            labels=labels,
            feature_names=["spatial_stress"],
            scaler=self.scaler if normalize else None,
            metadata={"data_type": "spatial_grid"}
        )


if __name__ == "__main__":
    # Example usage
    from synthetic_generator import SyntheticDataGenerator
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(seed=42)
    scenario_data = generator.generate_rainfall_scenario(duration_hours=168)
    
    # Preprocess for LSTM
    preprocessor = DataPreprocessor(scaling_method="standard")
    lstm_data = preprocessor.prepare_lstm_data(
        scenario_data,
        window_size=60,
        prediction_horizon=60
    )
    
    print(f"LSTM Features: {lstm_data.features.shape}")
    print(f"LSTM Labels: {lstm_data.labels.shape}")
    print(f"Positive samples: {lstm_data.labels.sum()}/{len(lstm_data.labels)}")
