"""
Stress Distribution Simulator
Calculates stress fields on slopes based on geotechnical parameters
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StressField:
    """Stress distribution data"""
    normal_stress: np.ndarray      # Normal stress (kPa)
    shear_stress: np.ndarray       # Shear stress (kPa)
    factor_of_safety: np.ndarray   # Factor of Safety (dimensionless)
    pore_pressure: np.ndarray      # Pore water pressure (kPa)
    

class StressSimulator:
    """Simulates stress distribution on slopes"""
    
    def __init__(
        self,
        rock_density: float = 2500.0,        # kg/m³
        cohesion: float = 100.0,              # kPa
        friction_angle: float = 35.0,         # degrees
        water_table_depth: float = 10.0      # meters below surface
    ):
        """
        Args:
            rock_density: Bulk density of rock mass
            cohesion: Soil/rock cohesion
            friction_angle: Internal friction angle (degrees)
            water_table_depth: Depth to water table
        """
        self.rock_density = rock_density
        self.cohesion = cohesion
        self.friction_angle = np.radians(friction_angle)
        self.water_table_depth = water_table_depth
        self.g = 9.81  # Gravitational acceleration (m/s²)
        
    def calculate_stress_field(
        self,
        elevation: np.ndarray,
        slope: np.ndarray,
        resolution: float = 1.0
    ) -> StressField:
        """
        Calculate stress distribution across terrain
        
        Args:
            elevation: DEM elevation grid (meters)
            slope: Slope angle grid (degrees)
            resolution: Grid resolution (meters)
            
        Returns:
            StressField object with stress components
        """
        logger.info(f"Calculating stress field for {elevation.shape} grid")
        
        # Convert slope to radians
        slope_rad = np.radians(slope)
        
        # Estimate depth below surface (simplified: use local elevation variation)
        # In reality, this would come from geological survey
        depth = self._estimate_depth_to_failure_plane(elevation, slope, resolution)
        
        # Calculate normal stress (perpendicular to slope)
        normal_stress = self._calculate_normal_stress(depth, slope_rad)
        
        # Calculate shear stress (parallel to slope)
        shear_stress = self._calculate_shear_stress(depth, slope_rad)
        
        # Calculate pore water pressure
        pore_pressure = self._calculate_pore_pressure(depth)
        
        # Calculate effective normal stress
        effective_normal_stress = normal_stress - pore_pressure
        
        # Calculate shear strength (Mohr-Coulomb criterion)
        shear_strength = self.cohesion + effective_normal_stress * np.tan(self.friction_angle)
        
        # Factor of Safety
        factor_of_safety = shear_strength / (shear_stress + 1e-6)  # Avoid division by zero
        factor_of_safety = np.clip(factor_of_safety, 0, 10)  # Reasonable bounds
        
        logger.info(f"FoS range: {factor_of_safety.min():.2f} - {factor_of_safety.max():.2f}")
        
        return StressField(
            normal_stress=normal_stress,
            shear_stress=shear_stress,
            factor_of_safety=factor_of_safety,
            pore_pressure=pore_pressure
        )
    
    def _estimate_depth_to_failure_plane(
        self,
        elevation: np.ndarray,
        slope: np.ndarray,
        resolution: float
    ) -> np.ndarray:
        """
        Estimate depth to potential failure plane
        Simplified model: depth proportional to slope height
        """
        # Use elevation gradient as proxy for potential failure depth
        # Steeper slopes → shallower failures
        # Higher slopes → deeper failures
        
        elev_range = elevation.max() - elevation.min()
        normalized_elev = (elevation - elevation.min()) / (elev_range + 1e-6)
        
        # Base depth: 5-20m depending on slope height
        base_depth = 5.0 + 15.0 * normalized_elev
        
        # Adjust for slope angle (steeper = shallower failure)
        slope_factor = np.clip(1.0 - slope / 60.0, 0.3, 1.0)
        
        depth = base_depth * slope_factor
        
        return depth
    
    def _calculate_normal_stress(self, depth: np.ndarray, slope_rad: np.ndarray) -> np.ndarray:
        """
        Calculate normal stress (perpendicular to slope surface)
        σₙ = γ * z * cos²(β)
        """
        unit_weight = self.rock_density * self.g / 1000  # Convert to kPa
        normal_stress = unit_weight * depth * np.cos(slope_rad) ** 2
        
        return normal_stress
    
    def _calculate_shear_stress(self, depth: np.ndarray, slope_rad: np.ndarray) -> np.ndarray:
        """
        Calculate shear stress (parallel to slope surface)
        τ = γ * z * sin(β) * cos(β)
        """
        unit_weight = self.rock_density * self.g / 1000
        shear_stress = unit_weight * depth * np.sin(slope_rad) * np.cos(slope_rad)
        
        return shear_stress
    
    def _calculate_pore_pressure(self, depth: np.ndarray) -> np.ndarray:
        """
        Calculate pore water pressure
        u = γ_water * (depth - water_table_depth)  if depth > water_table
        """
        water_unit_weight = 9.81  # kN/m³ = kPa/m
        
        # Pore pressure only below water table
        depth_below_wt = np.maximum(depth - self.water_table_depth, 0)
        pore_pressure = water_unit_weight * depth_below_wt
        
        return pore_pressure
    
    def apply_rainfall_effect(
        self,
        stress_field: StressField,
        rainfall_mm: float,
        duration_hours: float
    ) -> StressField:
        """
        Modify stress field based on rainfall
        
        Args:
            stress_field: Existing stress field
            rainfall_mm: Cumulative rainfall (mm)
            duration_hours: Rainfall duration (hours)
            
        Returns:
            Modified stress field
        """
        logger.info(f"Applying rainfall effect: {rainfall_mm}mm over {duration_hours}h")
        
        # Increase pore pressure (infiltration)
        # Simplified: 1mm rain ≈ 0.1 kPa pore pressure increase
        pore_pressure_increase = rainfall_mm * 0.1
        
        # Apply with exponential decay over duration
        time_factor = np.exp(-duration_hours / 24)  # Decay over 24 hours
        new_pore_pressure = stress_field.pore_pressure + pore_pressure_increase * time_factor
        
        # Recalculate FoS
        effective_normal = stress_field.normal_stress - new_pore_pressure
        shear_strength = self.cohesion + effective_normal * np.tan(self.friction_angle)
        new_fos = shear_strength / (stress_field.shear_stress + 1e-6)
        new_fos = np.clip(new_fos, 0, 10)
        
        logger.info(f"FoS after rainfall: {new_fos.min():.2f} - {new_fos.max():.2f}")
        
        return StressField(
            normal_stress=stress_field.normal_stress,
            shear_stress=stress_field.shear_stress,
            factor_of_safety=new_fos,
            pore_pressure=new_pore_pressure
        )
    
    def apply_excavation_effect(
        self,
        stress_field: StressField,
        excavation_mask: np.ndarray,
        stress_concentration_factor: float = 1.5
    ) -> StressField:
        """
        Modify stress field based on excavation
        
        Args:
            stress_field: Existing stress field
            excavation_mask: Boolean array marking excavated areas
            stress_concentration_factor: Stress increase at excavation edges
            
        Returns:
            Modified stress field
        """
        logger.info(f"Applying excavation effect")
        
        # Identify excavation boundaries (stress concentration zones)
        from scipy.ndimage import binary_dilation, binary_erosion
        
        excavation_edge = binary_dilation(excavation_mask) & ~excavation_mask
        
        # Increase shear stress at edges
        new_shear = stress_field.shear_stress.copy()
        new_shear[excavation_edge] *= stress_concentration_factor
        
        # Recalculate FoS
        effective_normal = stress_field.normal_stress - stress_field.pore_pressure
        shear_strength = self.cohesion + effective_normal * np.tan(self.friction_angle)
        new_fos = shear_strength / (new_shear + 1e-6)
        new_fos = np.clip(new_fos, 0, 10)
        
        logger.info(f"FoS after excavation: {new_fos.min():.2f} - {new_fos.max():.2f}")
        
        return StressField(
            normal_stress=stress_field.normal_stress,
            shear_stress=new_shear,
            factor_of_safety=new_fos,
            pore_pressure=stress_field.pore_pressure
        )
    
    def get_risk_map(self, stress_field: StressField) -> np.ndarray:
        """
        Convert Factor of Safety to risk level (0-100)
        
        Args:
            stress_field: Stress field with FoS
            
        Returns:
            Risk map (0=safe, 100=critical)
        """
        fos = stress_field.factor_of_safety
        
        # Risk mapping:
        # FoS > 2.0 → Low risk (0-25)
        # FoS 1.5-2.0 → Medium risk (25-60)
        # FoS 1.2-1.5 → High risk (60-85)
        # FoS < 1.2 → Critical risk (85-100)
        
        risk = np.zeros_like(fos)
        
        # Critical
        risk[fos < 1.2] = 85 + (1.2 - fos[fos < 1.2]) * 75  # Scale 85-100
        
        # High
        mask_high = (fos >= 1.2) & (fos < 1.5)
        risk[mask_high] = 60 + (1.5 - fos[mask_high]) / 0.3 * 25  # Scale 60-85
        
        # Medium
        mask_med = (fos >= 1.5) & (fos < 2.0)
        risk[mask_med] = 25 + (2.0 - fos[mask_med]) / 0.5 * 35  # Scale 25-60
        
        # Low
        mask_low = fos >= 2.0
        risk[mask_low] = np.maximum(0, 25 - (fos[mask_low] - 2.0) * 5)  # Scale 25-0
        
        risk = np.clip(risk, 0, 100)
        
        return risk


if __name__ == "__main__":
    # Example usage
    from data.dem_ingestion import DEMIngestion
    
    # Generate terrain
    dem_ingestor = DEMIngestion()
    dem = dem_ingestor.generate_synthetic_dem(width=128, height=128, slope_angle=35.0)
    slope = dem_ingestor.calculate_slope(dem)
    
    # Calculate stress field
    simulator = StressSimulator(
        rock_density=2500.0,
        cohesion=150.0,
        friction_angle=35.0,
        water_table_depth=10.0
    )
    
    stress_field = simulator.calculate_stress_field(dem.elevation, slope, dem.resolution)
    
    # Get risk map
    risk_map = simulator.get_risk_map(stress_field)
    
    print(f"Stress field calculated")
    print(f"Factor of Safety: {stress_field.factor_of_safety.min():.2f} - {stress_field.factor_of_safety.max():.2f}")
    print(f"Risk level: {risk_map.min():.1f} - {risk_map.max():.1f}")
    
    # Simulate rainfall
    stress_after_rain = simulator.apply_rainfall_effect(stress_field, rainfall_mm=50, duration_hours=6)
    risk_after_rain = simulator.get_risk_map(stress_after_rain)
    
    print(f"After 50mm rainfall:")
    print(f"Risk level: {risk_after_rain.min():.1f} - {risk_after_rain.max():.1f}")
