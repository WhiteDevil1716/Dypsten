"""
Ensemble Decision Engine
Combines LSTM, CNN, and Physics models with voting and drift detection
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction output"""
    probability: float
    risk_score: float
    risk_level: str
    confidence: float
    model_predictions: Dict[str, float]
    model_confidences: Dict[str, float]
    explanations: List[str]
    drift_detected: bool
    recommendation: str


class DriftDetector:
    """Detects prediction drift (model inconsistency)"""
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.3
    ):
        """
        Args:
            window_size: Number of recent predictions to track
            drift_threshold: Max allowed std deviation in predictions
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.prediction_history = deque(maxlen=window_size)
        
    def update(self, predictions: Dict[str, float]) -> bool:
        """
        Update drift detector with new predictions
        
        Args:
            predictions: Dict of model predictions (e.g., {"lstm": 0.8, "cnn": 0.3})
            
        Returns:
            True if drift detected
        """
        # Calculate variance across models
        pred_values = list(predictions.values())
        variance = np.std(pred_values)
        
        self.prediction_history.append(variance)
        
        # Check if recent variance is consistently high
        if len(self.prediction_history) >= 10:
            recent_variance = np.mean(list(self.prediction_history)[-10:])
            drift_detected = recent_variance > self.drift_threshold
        else:
            drift_detected = False
        
        if drift_detected:
            logger.warning(f"Drift detected! Recent variance: {recent_variance:.3f}")
        
        return drift_detected
    
    def reset(self):
        """Reset drift detector"""
        self.prediction_history.clear()


class EnsembleEngine:
    """Ensemble decision engine combining multiple models"""
    
    def __init__(
        self,
        lstm_model = None,
        cnn_model = None,
        physics_model = None,
        weights: Dict[str, float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            lstm_model: Trained LSTM model
            cnn_model: Trained CNN model
            physics_model: Physics guardrail model
            weights: Model weights for ensemble (should sum to 1.0)
            device: Torch device
        """
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.physics_model = physics_model
        self.device = device
        
        # Default weights if  not provided
        if weights is None:
            weights = {
                "lstm": 0.4,
                "cnn": 0.4,
                "physics": 0.2
            }
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        self.drift_detector = DriftDetector()
        
        logger.info(f"Ensemble initialized with weights: {self.weights}")
        
    def predict(
        self,
        lstm_input: Optional[np.ndarray] = None,
        cnn_input: Optional[np.ndarray] = None,
        sensor_data: Optional[Dict] = None,
        fos: Optional[float] = None
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction
        
        Args:
            lstm_input: Temporal sequence (sequence_length, num_features)
            cnn_input: Spatial grid (height, width) or (height, width, channels)
            sensor_data: Dict of current sensor values for physics model
            fos: Factor of Safety for physics model
            
        Returns:
            EnsemblePrediction object
        """
        model_predictions = {}
        model_confidences = {}
        explanations = []
        
        # LSTM prediction
        if self.lstm_model is not None and lstm_input is not None:
            lstm_prob = self._predict_lstm(lstm_input)
            model_predictions["lstm"] = lstm_prob
            model_confidences["lstm"] = self._calculate_confidence(lstm_prob)
            explanations.append(f"LSTM temporal analysis: {lstm_prob:.1%} probability")
        
        # CNN prediction
        if self.cnn_model is not None and cnn_input is not None:
            cnn_prob = self._predict_cnn(cnn_input)
            model_predictions["cnn"] = cnn_prob
            model_confidences["cnn"] = self._calculate_confidence(cnn_prob)
            explanations.append(f"CNN spatial analysis: {cnn_prob:.1%} probability")
        
        # Physics model prediction
        if self.physics_model is not None and sensor_data is not None:
            physics_result = self.physics_model.predict(sensor_data, fos=fos)
            model_predictions["physics"] = physics_result["probability"]
            model_confidences["physics"] = physics_result["confidence"]
            explanations.extend(physics_result["explanations"])
        
        # Check drift
        drift_detected = self.drift_detector.update(model_predictions)
        
        # Weighted voting
        weighted_prob = self._weighted_vote(model_predictions)
        
        # If drift detected, use conservative approach (max of predictions)
        if drift_detected:
            final_prob = max(model_predictions.values())
            explanations.append("⚠️ Model drift detected - using conservative estimate")
        else:
            final_prob = weighted_prob
        
        # Convert to risk score and level
        risk_score = final_prob * 100
        risk_level = self._prob_to_level(final_prob)
        
        # Overall confidence (average of model confidences)
        avg_confidence = np.mean(list(model_confidences.values()))
        
        # Generate recommendation
        recommendation = self._generate_recommendation(risk_level, final_prob, drift_detected)
        
        return EnsemblePrediction(
            probability=final_prob,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=avg_confidence,
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            explanations=explanations,
            drift_detected=drift_detected,
            recommendation=recommendation
        )
    
    def _predict_lstm(self, lstm_input: np.ndarray) -> float:
        """Make LSTM prediction"""
        self.lstm_model.eval()
        
        # Ensure correct shape (1, sequence_length, num_features)
        if len(lstm_input.shape) == 2:
            lstm_input = lstm_input[np.newaxis, :, :]
        
        with torch.no_grad():
            x = torch.FloatTensor(lstm_input).to(self.device)
            output = self.lstm_model(x)
            probability = output.cpu().numpy()[0, 0]
        
        return float(probability)
    
    def _predict_cnn(self, cnn_input: np.ndarray) -> float:
        """Make CNN prediction"""
        self.cnn_model.eval()
        
        # Ensure correct shape (1, channels, height, width)
        if len(cnn_input.shape) == 2:
            cnn_input = cnn_input[np.newaxis, np.newaxis, :, :]
        elif len(cnn_input.shape) == 3:
            # Assume (height, width, channels) → (1, channels, height, width)
            cnn_input = np.transpose(cnn_input, (2, 0, 1))[np.newaxis, :, :, :]
        
        with torch.no_grad():
            x = torch.FloatTensor(cnn_input).to(self.device)
            output = self.cnn_model(x)
            probability = output.cpu().numpy()[0, 0]
        
        return float(probability)
    
    def _weighted_vote(self, model_predictions: Dict[str, float]) -> float:
        """Calculate weighted average of predictions"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in model_predictions.items():
            weight = self.weights.get(model_name, 0.0)
            weighted_sum += prediction * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback: simple average
            return np.mean(list(model_predictions.values()))
    
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate model confidence based on prediction certainty
        Predictions close to 0.5 have low confidence, close to 0 or 1 have high confidence
        """
        distance_from_50 = abs(probability - 0.5)
        confidence = distance_from_50 * 2  # Scale to 0-1
        return confidence
    
    def _prob_to_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        risk_score = probability * 100
        
        if risk_score >= 85:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 25:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendation(self, risk_level: str, probability: float, drift_detected: bool) -> str:
        """Generate actionable recommendation"""
        recommendations = {
            "CRITICAL": "🚨 IMMEDIATE EVACUATION REQUIRED - Slope failure imminent within 1 hour",
            "HIGH": "⚠️ EVACUATE NON-ESSENTIAL PERSONNEL - High instability detected",
            "MEDIUM": "⚡ INCREASE MONITORING - Elevated risk, prepare evacuation plan",
            "LOW": "✅ CONTINUE NORMAL OPERATIONS - Conditions stable"
        }
        
        rec = recommendations.get(risk_level, "Monitor closely")
        
        if drift_detected and risk_level in ["HIGH", "CRITICAL"]:
            rec += " | Model uncertainty detected - exercise extreme caution"
        
        return rec
    
    def batch_predict(
        self,
        lstm_inputs: Optional[np.ndarray] = None,
        cnn_inputs: Optional[np.ndarray] = None,
        sensor_data_list: Optional[List[Dict]] = None,
        fos_list: Optional[List[float]] = None
    ) -> List[EnsemblePrediction]:
        """
        Batch prediction
        
        Args:
            lstm_inputs: Array of LSTM inputs (num_samples, seq_len, features)
            cnn_inputs: Array of CNN inputs (num_samples, height, width)
            sensor_data_list: List of sensor dicts
            fos_list: List of FoS values
            
        Returns:
            List of EnsemblePrediction objects
        """
        num_samples = len(lstm_inputs) if lstm_inputs is not None else len(cnn_inputs)
        predictions = []
        
        for i in range(num_samples):
            lstm_in = lstm_inputs[i] if lstm_inputs is not None else None
            cnn_in = cnn_inputs[i] if cnn_inputs is not None else None
            sensor = sensor_data_list[i] if sensor_data_list is not None else None
            fos = fos_list[i] if fos_list is not None else None
            
            pred = self.predict(lstm_in, cnn_in, sensor, fos)
            predictions.append(pred)
        
        return predictions


if __name__ == "__main__":
    from models.physics_guardrail import PhysicsGuardrail
    
    # Example usage (without trained LSTM/CNN for demo)
    physics_model = PhysicsGuardrail()
    
    ensemble = EnsembleEngine(
        physics_model=physics_model,
        weights={"physics": 1.0}  # Only physics for demo
    )
    
    # Test prediction
    sensor_data = {
        "geophone": 8.0,
        "inclinometer_current": 0.15,
        "inclinometer_previous": 0.10,
        "soil_moisture": 75.0,
        "acoustic": 80.0
    }
    
    prediction = ensemble.predict(sensor_data=sensor_data, fos=1.3)
    
    print(f"Ensemble Prediction:")
    print(f"  Risk Level: {prediction.risk_level}")
    print(f"  Probability: {prediction.probability:.1%}")
    print(f"  Confidence: {prediction.confidence:.1%}")
    print(f"  Recommendation: {prediction.recommendation}")
    print(f"  Drift Detected: {prediction.drift_detected}")
    print(f"\nExplanations:")
    for exp in prediction.explanations:
        print(f"  - {exp}")
