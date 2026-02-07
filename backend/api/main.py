"""
FastAPI Application - Main API Server
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging
from datetime import datetime
import numpy as np

from models.ensemble_engine import EnsembleEngine, EnsemblePrediction
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.physics_guardrail import PhysicsGuardrail
from alerts.alert_manager import AlertManager, Alert
from digital_twin.terrain_builder import TerrainBuilder
from digital_twin.stress_simulator import StressSimulator
from data.dem_ingestion import DEMIngestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Dypsten API",
    description="AI-Driven Rockfall Early Warning System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
alert_manager = AlertManager()
physics_guardrail = PhysicsGuardrail()
ensemble_engine = EnsembleEngine(physics_model=physics_guardrail)
terrain_builder = TerrainBuilder()
stress_simulator = StressSimulator()
dem_ingestion = DEMIngestion()

# WebSocket clients
websocket_clients: List[WebSocket] = []


# Pydantic models
class PredictionRequest(BaseModel):
    sensor_data: dict
    fos: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    risk_level: str
    risk_score: float
    probability: float
    confidence: float
    recommendation: str
    explanations: List[str]
    drift_detected: bool


class AlertResponse(BaseModel):
    id: str
    timestamp: str
    priority: str
    risk_level: str
    risk_score: float
    message: str
    recommendation: str
    acknowledged: bool


# API Routes

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "operational",
        "system": "Dypsten Early Warning System",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    active_alerts = alert_manager.get_active_alerts()
    
    return {
        "status": "active",
        "models": {
            "ensemble": "operational",
            "physics_guardrail": "operational",
            "lstm": "ready",
            "cnn": "ready"
        },
        "active_alerts": len(active_alerts),
        "websocket_clients": len(websocket_clients),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """
    Make a prediction based on sensor data
    
    Args:
        sensor_data: Dict of sensor readings (geophone, inclinometer, etc.)
        fos: Optional Factor of Safety
        
    Returns:
        Prediction with risk level, score, and recommendations
    """
    try:
        # Make prediction using ensemble
        prediction = ensemble_engine.predict(
            sensor_data=request.sensor_data,
            fos=request.fos
        )
        
        # Create alert if risk is elevated
        if prediction.risk_score >= 60:
            alert = alert_manager.create_alert(
                risk_level=prediction.risk_level,
                risk_score=prediction.risk_score,
                probability=prediction.probability,
                message=f"Risk detected: {prediction.risk_level}",
                recommendation=prediction.recommendation,
                explanations=prediction.explanations,
                confidence=prediction.confidence
            )
            
            # Deliver alert
            await alert_manager.deliver_alert(alert)
            
            # Broadcast to WebSocket clients
            await broadcast_alert(alert)
        
        # Return prediction
        return PredictionResponse(
            prediction_id=f"PRED-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            risk_level=prediction.risk_level,
            risk_score=prediction.risk_score,
            probability=prediction.probability,
            confidence=prediction.confidence,
            recommendation=prediction.recommendation,
            explanations=prediction.explanations,
            drift_detected=prediction.drift_detected
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts", response_model=List[AlertResponse])
async def get_alerts(active_only: bool = False, limit: int = 50):
    """
    Get alerts
    
    Args:
        active_only: Only return unacknowledged alerts
        limit: Maximum number of alerts to return
        
    Returns:
        List of alerts
    """
    if active_only:
        alerts = alert_manager.get_active_alerts()
    else:
        alerts = alert_manager.get_alert_history(limit=limit)
    
    return [
        AlertResponse(
            id=a.id,
            timestamp=a.timestamp.isoformat(),
            priority=a.priority,
            risk_level=a.risk_level,
            risk_score=a.risk_score,
            message=a.message,
            recommendation=a.recommendation,
            acknowledged=a.acknowledged
        )
        for a in alerts
    ]


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = "operator"):
    """Acknowledge an alert"""
    success = alert_manager.acknowledge_alert(alert_id, acknowledged_by)
    
    if success:
        # Broadcast acknowledgment to WebSocket clients
        await broadcast_message({
            "type": "alert_acknowledged",
            "alert_id": alert_id,
            "acknowledged_by": acknowledged_by,
            "timestamp": datetime.now().isoformat()
        })
        return {"status": "acknowledged", "alert_id": alert_id}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")


@app.get("/api/terrain/mesh")
async def get_terrain_mesh(size: int = 128):
    """
    Generate terrain mesh for digital twin
    
    Args:
        size: Grid size (default 128x128)
        
    Returns:
        3D mesh data in JSON format
    """
    try:
        # Generate DEM
        dem = dem_ingestion.generate_synthetic_dem(size, size, slope_angle=35.0)
        
        # Calculate slope
        slope = dem_ingestion.calculate_slope(dem)
        
        # Build mesh
        mesh = terrain_builder.dem_to_mesh(
            dem.elevation,
            resolution=dem.resolution,
            simplification_factor=2
        )
        
        # Add slope coloring
        mesh = terrain_builder.add_color_by_slope(mesh, slope)
        
        # Calculate stress field
        stress_field = stress_simulator.calculate_stress_field(
            dem.elevation,
            slope,
            dem.resolution
        )
        
        risk_map = stress_simulator.get_risk_map(stress_field)
        
        return {
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "normals": mesh.normals.tolist(),
            "colors": mesh.colors.tolist(),
            "risk_map": risk_map.tolist(),
            "metadata": {
                "size": size,
                "resolution": dem.resolution,
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces)
            }
        }
        
    except Exception as e:
        logger.error(f"Mesh generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Dypsten real-time updates",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Receive messages from client (e.g., acknowledgments)
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")
            
            # Echo back
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(websocket_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)


async def broadcast_alert(alert: Alert):
    """Broadcast alert to all WebSocket clients"""
    message = {
        "type": "alert",
        "id": alert.id,
        "timestamp": alert.timestamp.isoformat(),
        "priority": alert.priority,
        "risk_level": alert.risk_level,
        "risk_score": alert.risk_score,
        "message": alert.message,
        "recommendation": alert.recommendation,
        "confidence": alert.confidence
    }
    
    await broadcast_message(message)


async def broadcast_message(message: dict):
    """Broadcast message to all WebSocket clients"""
    disconnected = []
    
    for client in websocket_clients:
        try:
            await client.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to client: {e}")
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        if client in websocket_clients:
            websocket_clients.remove(client)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Dypsten API starting up...")
    logger.info("Alert manager initialized")
    logger.info("Ensemble engine ready")
    logger.info("API server operational")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Dypsten API shutting down...")
    
    # Close all WebSocket connections
    for client in websocket_clients:
        await client.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
