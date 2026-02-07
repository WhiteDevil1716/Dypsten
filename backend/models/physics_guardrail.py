"""
Physics-Informed Guardrail Model
Rule-based safety model that never fails - provides baseline predictions
"""
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhysicsRules:
    """Physics-based safety thresholds"""
    # Factor of Safety
    fos_critical: float = 1.2
    fos_high: float = 1.5
    fos_medium: float = 2.0
    
    # Ground acceleration (mm/s)
    vibration_critical: float = 10.0
    vibration_high: float = 5.0
    vibration_medium: float = 2.0
    
    # Tilt rate (degrees/hour)
    tilt_rate_critical: float = 0.1
    tilt_rate_high: float = 0.05
    tilt_rate_medium: float = 0.02
    
    # Soil moisture (%)
    moisture_critical: float = 80.0
    moisture_high: float = 60.0
    
    # Acoustic emission rate (events/hour)
    acoustic_critical: float = 100.0
    acoustic_high: float = 50.0


class PhysicsGuardrail:
    """Rule-based model using geotechnical principles"""
    
    def __init__(self, rules: PhysicsRules = None):
        self.rules = rules if rules is not None else PhysicsRules()
        logger.info("Physics guardrail initialized")
        
    def evaluate_factor_of_safety(self, fos: float) -> Tuple[float, str]:
        """
        Evaluate risk based on Factor of Safety
        
        Args:
            fos: Factor of Safety value
            
        Returns:
            (risk_score, risk_level)
        """
        if fos < self.rules.fos_critical:
            return 95.0, "CRITICAL"
        elif fos < self.rules.fos_high:
            return 75.0, "HIGH"
        elif fos < self.rules.fos_medium:
            return 45.0, "MEDIUM"
        else:
            return 15.0, "LOW"
    
    def evaluate_vibration(self, vibration: float) -> Tuple[float, str]:
        """Evaluate risk based on ground vibration"""
        if vibration > self.rules.vibration_critical:
            return 90.0, "CRITICAL"
        elif vibration > self.rules.vibration_high:
            return 70.0, "HIGH"
        elif vibration > self.rules.vibration_medium:
            return 40.0, "MEDIUM"
        else:
            return 10.0, "LOW"
    
    def evaluate_tilt_rate(self, current_tilt: float, previous_tilt: float, dt_hours: float = 1.0) -> Tuple[float, str]:
        """Evaluate risk based on tilt rate of change"""
        tilt_rate = abs(current_tilt - previous_tilt) / dt_hours
        
        if tilt_rate > self.rules.tilt_rate_critical:
            return 95.0, "CRITICAL"
        elif tilt_rate > self.rules.tilt_rate_high:
            return 75.0, "HIGH"
        elif tilt_rate > self.rules.tilt_rate_medium:
            return 45.0, "MEDIUM"
        else:
            return 15.0, "LOW"
    
    def evaluate_moisture(self, moisture: float) -> Tuple[float, str]:
        """Evaluate risk based on soil moisture"""
        if moisture > self.rules.moisture_critical:
            return 85.0, "CRITICAL"
        elif moisture > self.rules.moisture_high:
            return 60.0, "HIGH"
        else:
            return 20.0, "LOW"
    
    def evaluate_acoustic_emission(self, event_rate: float) -> Tuple[float, str]:
        """Evaluate risk based on acoustic emission rate"""
        if event_rate > self.rules.acoustic_critical:
            return 90.0, "CRITICAL"
        elif event_rate > self.rules.acoustic_high:
            return 65.0, "HIGH"
        else:
            return 25.0, "LOW"
    
    def predict(
        self,
        sensor_data: Dict[str, float],
        fos: float = None
    ) -> Dict:
        """
        Make prediction based on physics rules
        
        Args:
            sensor_data: Dict with keys: geophone, inclinometer, soil_moisture, acoustic
            fos: Optional Factor of Safety value
            
        Returns:
            Prediction dict with risk score, level, and explanations
        """
        risk_scores = []
        risk_levels = []
        explanations = []
        
        # Evaluate Factor of Safety if provided
        if fos is not None:
            score, level = self.evaluate_factor_of_safety(fos)
            risk_scores.append(score)
            risk_levels.append(level)
            explanations.append(f"Factor of Safety: {fos:.2f} → {level}")
        
        # Evaluate vibration
        if 'geophone' in sensor_data:
            score, level = self.evaluate_vibration(abs(sensor_data['geophone']))
            risk_scores.append(score)
            risk_levels.append(level)
            explanations.append(f"Vibration: {sensor_data['geophone']:.2f} mm/s → {level}")
        
        # Evaluate tilt rate (need current and previous)
        if 'inclinometer_current' in sensor_data and 'inclinometer_previous' in sensor_data:
            score, level = self.evaluate_tilt_rate(
                sensor_data['inclinometer_current'],
                sensor_data['inclinometer_previous']
            )
            risk_scores.append(score)
            risk_levels.append(level)
            rate = sensor_data['inclinometer_current'] - sensor_data['inclinometer_previous']
            explanations.append(f"Tilt rate: {rate:.4f}°/hr → {level}")
        
        # Evaluate soil moisture
        if 'soil_moisture' in sensor_data:
            score, level = self.evaluate_moisture(sensor_data['soil_moisture'])
            risk_scores.append(score)
            risk_levels.append(level)
            explanations.append(f"Soil moisture: {sensor_data['soil_moisture']:.1f}% → {level}")
        
        # Evaluate acoustic emission
        if 'acoustic' in sensor_data:
            score, level = self.evaluate_acoustic_emission(sensor_data['acoustic'])
            risk_scores.append(score)
            risk_levels.append(level)
            explanations.append(f"Acoustic events: {sensor_data['acoustic']:.1f}/hr → {level}")
        
        # Final risk: maximum of all scores (conservative approach)
        final_risk = max(risk_scores) if risk_scores else 0.0
        final_level = self._risk_to_level(final_risk)
        
        # Probability: convert risk score to probability
        probability = final_risk / 100.0
        
        return {
            "risk_score": final_risk,
            "risk_level": final_level,
            "probability": probability,
            "explanations": explanations,
            "component_scores": risk_scores,
            "component_levels": risk_levels,
            "confidence": 1.0  # Physics rules always have full confidence
        }
    
    def _risk_to_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 85:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 25:
            return "MEDIUM"
        else:
            return "LOW"
    
    def batch_predict(
        self,
        sensor_timeseries: Dict[str, np.ndarray],
        fos_timeseries: np.ndarray = None
    ) -> np.ndarray:
        """
        Predict on time series data
        
        Args:
            sensor_timeseries: Dict of sensor arrays (each array is time series)
            fos_timeseries: Optional FoS time series
            
        Returns:
            Array of risk probabilities
        """
        num_samples = len(next(iter(sensor_timeseries.values())))
        probabilities = np.zeros(num_samples)
        
        for i in range(num_samples):
            sensor_point = {k: v[i] for k, v in sensor_timeseries.items()}
            
            # Add previous tilt if available
            if 'inclinometer' in sensor_point and i > 0:
                sensor_point['inclinometer_current'] = sensor_point['inclinometer']
                sensor_point['inclinometer_previous'] = sensor_timeseries['inclinometer'][i-1]
            
            fos = fos_timeseries[i] if fos_timeseries is not None else None
            
            prediction = self.predict(sensor_point, fos=fos)
            probabilities[i] = prediction['probability']
        
        return probabilities


if __name__ == "__main__":
    # Example usage
    guardrail = PhysicsGuardrail()
    
    # Test case 1: Normal conditions
    sensor_data_normal = {
        "geophone": 0.5,
        "inclinometer_current": 0.02,
        "inclinometer_previous": 0.015,
        "soil_moisture": 25.0,
        "acoustic": 10.0
    }
    
    result = guardrail.predict(sensor_data_normal, fos=2.5)
    print("Normal conditions:")
    print(f"  Risk: {result['risk_score']:.1f}, Level: {result['risk_level']}, Prob: {result['probability']:.2%}")
    print(f"  Explanations: {result['explanations']}")
    
    # Test case 2: Critical conditions
    sensor_data_critical = {
        "geophone": 12.0,
        "inclinometer_current": 0.25,
        "inclinometer_previous": 0.15,
        "soil_moisture": 85.0,
        "acoustic": 120.0
    }
    
    result = guardrail.predict(sensor_data_critical, fos=1.1)
    print("\nCritical conditions:")
    print(f"  Risk: {result['risk_score']:.1f}, Level: {result['risk_level']}, Prob: {result['probability']:.2%}")
    print(f"  Explanations: {result['explanations']}")
