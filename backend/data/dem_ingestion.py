"""
NASA DEM Data Ingestion Module
Fetches and processes Digital Elevation Model data for terrain analysis
"""
import os
import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DEMData:
    """Container for DEM data and metadata"""
    elevation: np.ndarray
    transform: rasterio.Affine
    crs: str
    bounds: Tuple[float, float, float, float]
    resolution: float
    nodata_value: Optional[float]


class DEMIngestion:
    """Handles DEM data ingestion and processing"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dem_from_file(self, filepath: str) -> DEMData:
        """
        Load DEM data from a GeoTIFF file
        
        Args:
            filepath: Path to DEM GeoTIFF file
            
        Returns:
            DEMData object containing elevation and metadata
        """
        logger.info(f"Loading DEM from {filepath}")
        
        with rasterio.open(filepath) as src:
            elevation = src.read(1)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            bounds = src.bounds
            resolution = src.res[0]  # Assuming square pixels
            nodata_value = src.nodata
            
        logger.info(f"DEM loaded: {elevation.shape[0]}x{elevation.shape[1]} @ {resolution}m resolution")
        
        return DEMData(
            elevation=elevation,
            transform=transform,
            crs=crs,
            bounds=bounds,
            resolution=resolution,
            nodata_value=nodata_value
        )
    
    def generate_synthetic_dem(
        self, 
        width: int = 512, 
        height: int = 512,
        base_elevation: float = 1000.0,
        slope_angle: float = 35.0,
        noise_amplitude: float = 5.0,
        seed: int = 42
    ) -> DEMData:
        """
        Generate synthetic DEM for testing purposes
        Creates a sloped terrain with realistic noise
        
        Args:
            width: Width in pixels
            height: Height in pixels
            base_elevation: Base elevation in meters
            slope_angle: Average slope angle in degrees
            noise_amplitude: Amplitude of terrain noise
            seed: Random seed for reproducibility
            
        Returns:
            DEMData object with synthetic terrain
        """
        logger.info(f"Generating synthetic DEM: {width}x{height}")
        
        np.random.seed(seed)
        
        # Create base slope (linear gradient)
        x = np.linspace(0, 100, width)
        y = np.linspace(0, 100, height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate slope in meters per horizontal meter
        slope_factor = np.tan(np.radians(slope_angle))
        
        # Base elevation surface (sloping from top to bottom)
        elevation = base_elevation + (Y * slope_factor)
        
        # Add realistic terrain features using Perlin-like noise
        # Multiple octaves of noise for realistic terrain
        noise = np.zeros_like(elevation)
        for octave in range(1, 6):
            frequency = 2 ** octave
            amplitude = noise_amplitude / octave
            
            noise_layer = amplitude * self._generate_noise(height, width, frequency, seed + octave)
            noise += noise_layer
        
        elevation += noise
        
        # Add some localized features (potential instability zones)
        # Small depressions or bulges
        for _ in range(5):
            cx, cy = np.random.randint(width // 4, 3 * width // 4, 2)
            radius = np.random.randint(20, 50)
            feature_strength = np.random.uniform(-10, 10)
            
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = np.exp(-dist ** 2 / (2 * radius ** 2))
            elevation += feature_strength * mask
        
        # Create transform (assuming 1m resolution, centered at origin)
        transform = rasterio.Affine(
            1.0, 0.0, 0.0,
            0.0, -1.0, height
        )
        
        bounds = (0.0, 0.0, float(width), float(height))
        
        logger.info(f"Synthetic DEM generated: elevation range {elevation.min():.1f} - {elevation.max():.1f}m")
        
        return DEMData(
            elevation=elevation,
            transform=transform,
            crs="EPSG:32633",  # UTM Zone 33N (example)
            bounds=bounds,
            resolution=1.0,
            nodata_value=None
        )
    
    def _generate_noise(self, height: int, width: int, frequency: int, seed: int) -> np.ndarray:
        """Generate smooth Perlin-like noise"""
        np.random.seed(seed)
        
        # Generate noise at lower resolution
        low_res_h = max(height // frequency, 1)
        low_res_w = max(width // frequency, 1)
        
        low_res_noise = np.random.randn(low_res_h, low_res_w)
        
        # Upsample using bilinear interpolation
        from scipy.ndimage import zoom
        zoom_factor = (height / low_res_h, width / low_res_w)
        upsampled = zoom(low_res_noise, zoom_factor, order=1)
        
        # Ensure correct shape
        upsampled = upsampled[:height, :width]
        
        return upsampled
    
    def calculate_slope(self, dem: DEMData) -> np.ndarray:
        """
        Calculate slope in degrees from DEM
        
        Args:
            dem: DEMData object
            
        Returns:
            Slope array in degrees
        """
        elevation = dem.elevation
        resolution = dem.resolution
        
        # Calculate gradients
        dy, dx = np.gradient(elevation, resolution)
        
        # Calculate slope (rise over run)
        slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_degrees = np.degrees(slope_radians)
        
        logger.info(f"Slope calculated: mean={slope_degrees.mean():.1f}°, max={slope_degrees.max():.1f}°")
        
        return slope_degrees
    
    def calculate_aspect(self, dem: DEMData) -> np.ndarray:
        """
        Calculate aspect (direction of slope) in degrees
        
        Args:
            dem: DEMData object
            
        Returns:
            Aspect array in degrees (0-360, 0=North)
        """
        elevation = dem.elevation
        resolution = dem.resolution
        
        dy, dx = np.gradient(elevation, resolution)
        
        # Calculate aspect (azimuth of steepest descent)
        aspect_radians = np.arctan2(-dx, dy)  # Note: -dx for geographic convention
        aspect_degrees = np.degrees(aspect_radians)
        
        # Convert to 0-360 range
        aspect_degrees = (aspect_degrees + 360) % 360
        
        return aspect_degrees
    
    def save_dem(self, dem: DEMData, filepath: str):
        """
        Save DEM to GeoTIFF file
        
        Args:
            dem: DEMData object
            filepath: Output file path
        """
        logger.info(f"Saving DEM to {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=dem.elevation.shape[0],
            width=dem.elevation.shape[1],
            count=1,
            dtype=dem.elevation.dtype,
            crs=dem.crs,
            transform=dem.transform,
            nodata=dem.nodata_value
        ) as dst:
            dst.write(dem.elevation, 1)
        
        logger.info(f"DEM saved successfully")


if __name__ == "__main__":
    # Example usage
    ingestor = DEMIngestion()
    
    # Generate synthetic DEM for testing
    dem = ingestor.generate_synthetic_dem(
        width=512,
        height=512,
        slope_angle=35.0,
        noise_amplitude=5.0
    )
    
    # Calculate terrain properties
    slope = ingestor.calculate_slope(dem)
    aspect = ingestor.calculate_aspect(dem)
    
    # Save for later use
    ingestor.save_dem(dem, "data/raw/synthetic_dem.tif")
    
    print(f"DEM Shape: {dem.elevation.shape}")
    print(f"Elevation Range: {dem.elevation.min():.1f} - {dem.elevation.max():.1f}m")
    print(f"Mean Slope: {slope.mean():.1f}°")
    print(f"Resolution: {dem.resolution}m")
