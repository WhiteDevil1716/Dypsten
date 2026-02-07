"""
Synthetic Sensor Data Generator
Generates realistic sensor data streams for testing and development
Physics-based simulation of geophone, inclinometer, soil moisture, and acoustic sensors
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Single sensor reading with timestamp"""
    timestamp: datetime
    value: float
    sensor_type: str
    sensor_id: str


class GeophoneSimulator:
    """Simulates ground vibration sensor (seismic activity)"""
    
    def __init__(self, baseline_noise: float = 0.5, seed: Optional[int] = None):
        """
        Args:
            baseline_noise: Background noise level (mm/s)
            seed: Random seed for reproducibility
        """
        self.baseline_noise = baseline_noise
        self.rng = np.random.default_rng(seed)
        
    def generate_signal(
        self,
        duration_hours: float = 24.0,
        sample_rate_hz: float = 1.0,
        events: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Generate geophone signal with optional seismic events
        
        Args:
            duration_hours: Duration of signal in hours
            sample_rate_hz: Sampling frequency (samples per second)
            events: List of (time_hours, magnitude) for seismic events
            
        Returns:
            Array of vibration amplitudes (mm/s)
        """
        num_samples = int(duration_hours * 3600 * sample_rate_hz)
        time_array = np.linspace(0, duration_hours, num_samples)
        
        # Baseline noise (Gaussian)
        signal = self.rng.normal(0, self.baseline_noise, num_samples)
        
        # Add low-frequency drift (environmental factors)
        drift_freq = 1 / (3600 * 6)  # 6-hour cycle
        drift_amplitude = self.baseline_noise * 0.3
        signal += drift_amplitude * np.sin(2 * np.pi * drift_freq * time_array * 3600)
        
        # Add seismic events if specified
        if events:
            for event_time, magnitude in events:
                event_signal = self._generate_seismic_event(
                    time_array, event_time, magnitude
                )
                signal += event_signal
        
        return signal
    
    def _generate_seismic_event(
        self, 
        time_array: np.ndarray, 
        event_time: float,
        magnitude: float
    ) -> np.ndarray:
        """Generate a seismic event pulse"""
        # Event parameters
        rise_time = 0.1  # hours
        decay_time = 0.5  # hours
        
        # Create exponentially decaying pulse
        time_from_event = (time_array - event_time) * 60  # Convert to minutes
        
        # Pre-event: zero
        # Post-event: exponential decay with oscillation
        event_signal = np.zeros_like(time_array)
        mask = time_from_event >= 0
        
        decay_factor = np.exp(-time_from_event[mask] / (decay_time * 60))
        oscillation = np.sin(2 * np.pi * 0.5 * time_from_event[mask])  # 0.5 Hz
        
        event_signal[mask] = magnitude * decay_factor * oscillation
        
        return event_signal


class InclinometerSimulator:
    """Simulates tilt sensor (micro-deformation)"""
    
    def __init__(self, baseline_tilt: float = 0.0, seed: Optional[int] = None):
        """
        Args:
            baseline_tilt: Initial tilt angle (degrees)
            seed: Random seed
        """
        self.baseline_tilt = baseline_tilt
        self.rng = np.random.default_rng(seed)
        
    def generate_signal(
        self,
        duration_hours: float = 24.0,
        sample_rate_hz: float = 1/60.0,  # 1 sample per minute
        instability_onset: Optional[float] = None,
        instability_rate: float = 0.05  # degrees per hour
    ) -> np.ndarray:
        """
        Generate inclinometer signal with optional slope instability
        
        Args:
            duration_hours: Duration in hours
            sample_rate_hz: Sampling frequency
            instability_onset: Time (hours) when instability begins
            instability_rate: Rate of tilt increase (degrees/hour) during instability
            
        Returns:
            Array of tilt angles (degrees)
        """
        num_samples = int(duration_hours * 3600 * sample_rate_hz)
        time_array = np.linspace(0, duration_hours, num_samples)
        
        # Baseline tilt with slow thermal drift
        thermal_noise = self.rng.normal(0, 0.001, num_samples)
        
        # Daily thermal cycle (expansion/contraction)
        thermal_cycle = 0.005 * np.sin(2 * np.pi * time_array / 24)
        
        signal = self.baseline_tilt + thermal_noise + thermal_cycle
        
        # Add instability if specified
        if instability_onset is not None:
            mask = time_array >= instability_onset
            time_since_onset = time_array[mask] - instability_onset
            
            # Accelerating tilt (quadratic growth - characteristic of slope failure)
            acceleration = 0.001  # degrees/hour²
            tilt_increase = (instability_rate * time_since_onset + 
                           0.5 * acceleration * time_since_onset**2)
            
            signal[mask] += tilt_increase
        
        return signal


class SoilMoistureSimulator:
    """Simulates soil moisture sensor (pore pressure proxy)"""
    
    def __init__(self, baseline_moisture: float = 20.0, seed: Optional[int] = None):
        """
        Args:
            baseline_moisture: Baseline volumetric water content (%)
            seed: Random seed
        """
        self.baseline_moisture = baseline_moisture
        self.rng = np.random.default_rng(seed)
        
    def generate_signal(
        self,
        duration_hours: float = 24.0,
        sample_rate_hz: float = 1/3600.0,  # 1 sample per hour
        rainfall_events: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Generate soil moisture signal with rainfall events
        
        Args:
            duration_hours: Duration in hours
            sample_rate_hz: Sampling frequency
            rainfall_events: List of (time_hours, rainfall_mm)
            
        Returns:
            Array of soil moisture (% volumetric water content)
        """
        num_samples = int(duration_hours * 3600 * sample_rate_hz)
        time_array = np.linspace(0, duration_hours, num_samples)
        
        # Start at baseline
        signal = np.ones(num_samples) * self.baseline_moisture
        
        # Natural evaporation (exponential decay)
        evaporation_rate = 0.05  # % per hour
        
        # Add rainfall events
        if rainfall_events:
            for event_time, rainfall_mm in rainfall_events:
                # Moisture increase (rough approximation: 1mm rain = 0.1% moisture increase)
                moisture_increase = rainfall_mm * 0.1
                
                # Find event index
                event_idx = np.searchsorted(time_array, event_time)
                
                # Apply moisture increase with infiltration lag
                for i in range(event_idx, num_samples):
                    time_since_rain = time_array[i] - event_time
                    
                    # Infiltration: gradual increase
                    infiltration = moisture_increase * (1 - np.exp(-time_since_rain / 2))
                    
                    # Drainage: exponential decay after peak
                    drainage = np.exp(-evaporation_rate * time_since_rain)
                    
                    signal[i] += infiltration * drainage
        
        # Natural daily variation
        daily_variation = 2.0 * np.sin(2 * np.pi * time_array / 24)
        signal += daily_variation
        
        # Add sensor noise
        signal += self.rng.normal(0, 0.5, num_samples)
        
        # Clamp to physical limits (0-100%)
        signal = np.clip(signal, 0, 100)
        
        return signal


class AcousticEmissionSimulator:
    """Simulates acoustic emission sensor (rock microfractures)"""
    
    def __init__(self, baseline_rate: float = 5.0, seed: Optional[int] = None):
        """
        Args:
            baseline_rate: Baseline crack rate (events per hour)
            seed: Random seed
        """
        self.baseline_rate = baseline_rate
        self.rng = np.random.default_rng(seed)
        
    def generate_signal(
        self,
        duration_hours: float = 24.0,
        sample_rate_hz: float = 1.0,
        stress_increase: Optional[float] = None,
        stress_onset: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate acoustic emission signal (event count)
        
        Args:
            duration_hours: Duration in hours
            sample_rate_hz: Sampling frequency
            stress_increase: Multiplier for crack rate during stress
            stress_onset: Time (hours) when stress begins
            
        Returns:
            Array of acoustic event counts per sample
        """
        num_samples = int(duration_hours * 3600 * sample_rate_hz)
        time_array = np.linspace(0, duration_hours, num_samples)
        
        # Base crack rate (Poisson process)
        base_rate = self.baseline_rate / 3600  # Convert to per-second
        signal = self.rng.poisson(base_rate / sample_rate_hz, num_samples)
        
        # Increase rate during stress
        if stress_increase is not None and stress_onset is not None:
            mask = time_array >= stress_onset
            time_since_stress = time_array[mask] - stress_onset
            
            # Exponentially increasing crack rate (precursor to failure)
            rate_multiplier = 1 + (stress_increase - 1) * (1 - np.exp(-time_since_stress / 5))
            
            increased_rate = base_rate * rate_multiplier / sample_rate_hz
            signal[mask] = self.rng.poisson(increased_rate)
        
        return signal.astype(float)


class SyntheticDataGenerator:
    """Orchestrates all sensor simulators"""
    
    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.geophone = GeophoneSimulator(seed=seed)
        self.inclinometer = InclinometerSimulator(seed=seed)
        self.soil_moisture = SoilMoistureSimulator(seed=seed)
        self.acoustic = AcousticEmissionSimulator(seed=seed)
        
    def generate_normal_scenario(self, duration_hours: float = 168.0) -> dict:
        """Generate 1 week of normal operating conditions"""
        logger.info(f"Generating normal scenario ({duration_hours}h)")
        
        return {
            "geophone": self.geophone.generate_signal(duration_hours),
            "inclinometer": self.inclinometer.generate_signal(duration_hours),
            "soil_moisture": self.soil_moisture.generate_signal(duration_hours),
            "acoustic": self.acoustic.generate_signal(duration_hours),
            "scenario": "normal",
            "duration_hours": duration_hours
        }
    
    def generate_rainfall_scenario(self, duration_hours: float = 72.0) -> dict:
        """Generate scenario with heavy rainfall"""
        logger.info(f"Generating rainfall scenario ({duration_hours}h)")
        
        # Simulate rainfall at hours 12, 18, 24 (cumulative 60mm)
        rainfall_events = [(12.0, 20.0), (18.0, 25.0), (24.0, 15.0)]
        
        return {
            "geophone": self.geophone.generate_signal(duration_hours),
            "inclinometer": self.inclinometer.generate_signal(
                duration_hours, 
                instability_onset=30.0,  # Instability 6h after last rain
                instability_rate=0.02
            ),
            "soil_moisture": self.soil_moisture.generate_signal(
                duration_hours,
                rainfall_events=rainfall_events
            ),
            "acoustic": self.acoustic.generate_signal(
                duration_hours,
                stress_increase=3.0,
                stress_onset=30.0
            ),
            "scenario": "rainfall",
            "duration_hours": duration_hours
        }
    
    def generate_seismic_scenario(self, duration_hours: float = 48.0) -> dict:
        """Generate scenario with seismic activity"""
        logger.info(f"Generating seismic scenario ({duration_hours}h)")
        
        # Seismic events at hours 6, 12 (magnitude 2.0, 3.5)
        seismic_events = [(6.0, 2.0), (12.0, 3.5)]
        
        return {
            "geophone": self.geophone.generate_signal(
                duration_hours,
                events=seismic_events
            ),
            "inclinometer": self.inclinometer.generate_signal(
                duration_hours,
                instability_onset=15.0,
                instability_rate=0.08
            ),
            "soil_moisture": self.soil_moisture.generate_signal(duration_hours),
            "acoustic": self.acoustic.generate_signal(
                duration_hours,
                stress_increase=5.0,
                stress_onset=13.0
            ),
            "scenario": "seismic",
            "duration_hours": duration_hours
        }


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate different scenarios
    normal_data = generator.generate_normal_scenario(168)  # 1 week
    rainfall_data = generator.generate_rainfall_scenario(72)  # 3 days
    seismic_data = generator.generate_seismic_scenario(48)  # 2 days
    
    print(f"Normal scenario: {len(normal_data['geophone'])} samples")
    print(f"Rainfall scenario: {len(rainfall_data['geophone'])} samples")
    print(f"Seismic scenario: {len(seismic_data['geophone'])} samples")
    
    # Save to numpy files
    import os
    os.makedirs("data/synthetic", exist_ok=True)
    
    for scenario_name, data in [("normal", normal_data), ("rainfall", rainfall_data), ("seismic", seismic_data)]:
        np.savez(
            f"data/synthetic/{scenario_name}_scenario.npz",
            **{k: v for k, v in data.items() if isinstance(v, np.ndarray)}
        )
        logger.info(f"Saved {scenario_name} scenario to data/synthetic/{scenario_name}_scenario.npz")
