"""
Terrain Builder for Digital Twin
Converts DEM data to 3D mesh format for visualization
"""
import numpy as np
from typing import Tuple, Dict, Optional
import json
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Mesh3D:
    """3D mesh data structure"""
    vertices: np.ndarray  # (N, 3) array of [x, y, z] coordinates
    faces: np.ndarray     # (M, 3) array of triangle vertex indices
    normals: np.ndarray   # (N, 3) array of vertex normals
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors per vertex
    uvs: Optional[np.ndarray] = None     # (N, 2) UV texture coordinates


class TerrainBuilder:
    """Builds 3D terrain meshes from DEM data"""
    
    def __init__(self, vertical_exaggeration: float = 1.0):
        """
        Args:
            vertical_exaggeration: Scale factor for elevation (for visualization)
        """
        self.vertical_exaggeration = vertical_exaggeration
        
    def dem_to_mesh(
        self,
        elevation: np.ndarray,
        resolution: float = 1.0,
        simplification_factor: int = 1
    ) -> Mesh3D:
        """
        Convert DEM elevation grid to 3D mesh
        
        Args:
            elevation: 2D array of elevation values
            resolution: Grid spacing in meters
            simplification_factor: Downsample factor (1=no simplification, 2=half resolution)
            
        Returns:
            Mesh3D object with vertices, faces, normals
        """
        logger.info(f"Building mesh from DEM: {elevation.shape}")
        
        # Simplify by downsampling if requested
        if simplification_factor > 1:
            elevation = elevation[::simplification_factor, ::simplification_factor]
            resolution *= simplification_factor
            logger.info(f"Simplified to {elevation.shape}")
        
        height, width = elevation.shape
        
        # Generate vertex grid
        vertices = self._create_vertices(elevation, resolution)
        
        # Generate triangle faces
        faces = self._create_faces(height, width)
        
        # Calculate normals
        normals = self._calculate_normals(vertices, faces)
        
        logger.info(f"Mesh created: {len(vertices)} vertices, {len(faces)} faces")
        
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            normals=normals
        )
    
    def _create_vertices(self, elevation: np.ndarray, resolution: float) -> np.ndarray:
        """Create vertex array from elevation grid"""
        height, width = elevation.shape
        
        # Create grid coordinates
        x = np.arange(width) * resolution
        y = np.arange(height) * resolution
        X, Y = np.meshgrid(x, y)
        
        # Apply vertical exaggeration to elevations
        Z = elevation * self.vertical_exaggeration
        
        # Flatten and stack into (N, 3) array
        vertices = np.column_stack([
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        ])
        
        return vertices
    
    def _create_faces(self, height: int, width: int) -> np.ndarray:
        """
        Create triangle faces for grid mesh
        Each grid cell becomes 2 triangles
        """
        faces = []
        
        for row in range(height - 1):
            for col in range(width - 1):
                # Vertex indices for current cell
                top_left = row * width + col
                top_right = top_left + 1
                bottom_left = (row + 1) * width + col
                bottom_right = bottom_left + 1
                
                # Two triangles per cell
                # Triangle 1: top-left, bottom-left, top-right
                faces.append([top_left, bottom_left, top_right])
                
                # Triangle 2: top-right, bottom-left, bottom-right
                faces.append([top_right, bottom_left, bottom_right])
        
        return np.array(faces, dtype=np.int32)
    
    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate per-vertex normals using face normals"""
        num_vertices = len(vertices)
        normals = np.zeros((num_vertices, 3), dtype=np.float32)
        
        # Calculate face normals
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # Cross product of two edges
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Accumulate to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms
        
        return normals
    
    def add_color_by_elevation(self, mesh: Mesh3D) -> Mesh3D:
        """
        Color vertices by elevation (terrain colormap)
        Low = green, Mid = yellow/brown, High = gray/white
        """
        z_values = mesh.vertices[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        
        # Normalize elevation to 0-1
        z_norm = (z_values - z_min) / (z_max - z_min + 1e-6)
        
        # Color mapping (RGB)
        colors = np.zeros((len(z_values), 3), dtype=np.float32)
        
        for i, z in enumerate(z_norm):
            if z < 0.3:
                # Low elevation: green
                colors[i] = [0.2 + z, 0.6, 0.2]
            elif z < 0.6:
                # Mid elevation: yellow/brown
                t = (z - 0.3) / 0.3
                colors[i] = [0.6 + 0.2*t, 0.5 - 0.2*t, 0.2]
            else:
                # High elevation: gray/white
                t = (z - 0.6) / 0.4
                colors[i] = [0.5 + 0.5*t, 0.5 + 0.5*t, 0.5 + 0.5*t]
        
        mesh.colors = colors
        logger.info("Applied elevation colormap")
        
        return mesh
    
    def add_color_by_slope(self, mesh: Mesh3D, slope: np.ndarray) -> Mesh3D:
        """
        Color vertices by slope angle
        Low slope = green (safe), High slope = red (dangerous)
        """
        slope_flat = slope.ravel()
        
        # Normalize slope (assume max dangerous slope is 45°)
        slope_norm = np.clip(slope_flat / 45.0, 0, 1)
        
        # Color gradient: green -> yellow -> red
        colors = np.zeros((len(slope_norm), 3), dtype=np.float32)
        
        for i, s in enumerate(slope_norm):
            if s < 0.5:
                # Green to yellow
                t = s * 2
                colors[i] = [t, 1.0, 0.0]
            else:
                # Yellow to red
                t = (s - 0.5) * 2
                colors[i] = [1.0, 1.0 - t, 0.0]
        
        mesh.colors = colors
        logger.info("Applied slope colormap")
        
        return mesh
    
    def export_to_gltf(self, mesh: Mesh3D, filepath: str):
        """
        Export mesh to glTF format (for Three.js)
        Uses simple JSON structure
        """
        logger.info(f"Exporting mesh to {filepath}")
        
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1
                    },
                    "indices": 2,
                    "mode": 4  # TRIANGLES
                }]
            }],
            "accessors": [
                # POSITION
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(mesh.vertices),
                    "type": "VEC3",
                    "max": mesh.vertices.max(axis=0).tolist(),
                    "min": mesh.vertices.min(axis=0).tolist()
                },
                # NORMAL
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": len(mesh.normals),
                    "type": "VEC3"
                },
                # INDICES
                {
                    "bufferView": 2,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": len(mesh.faces) * 3,
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": mesh.vertices.nbytes
                },
                {
                    "buffer": 0,
                    "byteOffset": mesh.vertices.nbytes,
                    "byteLength": mesh.normals.nbytes
                },
                {
                    "buffer": 0,
                    "byteOffset": mesh.vertices.nbytes + mesh.normals.nbytes,
                    "byteLength": mesh.faces.nbytes
                }
            ],
            "buffers": [{
                "byteLength": mesh.vertices.nbytes + mesh.normals.nbytes + mesh.faces.nbytes,
                "uri": Path(filepath).stem + ".bin"
            }]
        }
        
        # Save JSON
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(gltf, f, indent=2)
        
        # Save binary data
        bin_path = Path(filepath).with_suffix('.bin')
        with open(bin_path, 'wb') as f:
            f.write(mesh.vertices.astype(np.float32).tobytes())
            f.write(mesh.normals.astype(np.float32).tobytes())
            f.write(mesh.faces.astype(np.uint32).tobytes())
        
        logger.info(f"Exported to {filepath} and {bin_path}")
    
    def export_to_json(self, mesh: Mesh3D, filepath: str):
        """
        Export mesh to simple JSON format (alternative to glTF)
        Easier to parse in JavaScript
        """
        logger.info(f"Exporting mesh to {filepath}")
        
        data = {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "normals": mesh.normals.tolist()
        }
        
        if mesh.colors is not None:
            data["colors"] = mesh.colors.tolist()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Exported {len(mesh.vertices)} vertices to {filepath}")


if __name__ == "__main__":
    # Example usage
    from data.dem_ingestion import DEMIngestion
    
    # Load or generate DEM
    dem_ingestor = DEMIngestion()
    dem = dem_ingestor.generate_synthetic_dem(
        width=256,
        height=256,
        slope_angle=35.0
    )
    
    # Build mesh
    builder = TerrainBuilder(vertical_exaggeration=1.5)
    mesh = builder.dem_to_mesh(
        dem.elevation,
        resolution=dem.resolution,
        simplification_factor=2  # Reduce to 128x128 for performance
    )
    
    # Calculate slope for coloring
    slope = dem_ingestor.calculate_slope(dem)
    mesh = builder.add_color_by_slope(mesh, slope)
    
    # Export
    builder.export_to_json(mesh, "data/meshes/terrain.json")
    builder.export_to_gltf(mesh, "data/meshes/terrain.gltf")
    
    print(f"Mesh exported: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
