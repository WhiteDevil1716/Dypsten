# Dypsten 🏔️
## AI-Driven Rockfall Early Warning System

Dypsten is an advanced, AI-powered early warning system designed to predict slope instability in mining operations **1 hour in advance**. It combines machine learning, digital twin visualization, and physics-informed models to keep miners safe.

---

## Features

### 🤖 AI-Powered Prediction
- **LSTM Temporal Model**: Analyzes time-series sensor data for trend prediction
- **CNN Spatial Model**: Processes terrain stress maps for spatial risk analysis
- **Physics-Informed Guardrail**: Rule-based safety model that never fails
- **Ensemble Engine**: Combines multiple models with drift detection

### 🌍 Digital Twin Visualization
- **3D Terrain Mesh**: Interactive visualization built from NASA DEM data
- **Real-Time Stress Overlay**: Color-coded risk zones (green → yellow → red)
- **Deformation Animation**: Time-series visualization of slope movement
- **"What-If" Scenarios**: Simulate rainfall, excavation, freeze-thaw impacts

### 🚨 Multi-Channel Alerting
- **Risk Classification**: Low / Medium / High / Critical
- **Explainable Alerts**: Natural language descriptions with contributing factors
- **Delivery Channels**: Dashboard, Email, SMS, Mobile Push
- **Acknowledgment Tracking**: Ensures alerts are never missed

### 📊 Real-Time Dashboard
- **Live Risk Gauge**: Current instability probability
- **Prediction Charts**: Historical trends and future forecasts
- **Alert History**: Searchable, filterable event log
- **Scenario Simulator**: Interactive "what-if" controls

---

## Technology Stack

### Backend
- Python 3.10+
- PyTorch / TensorFlow (ML models)
- FastAPI (API server)
- PostgreSQL + PostGIS (database)
- Redis (caching)

### Frontend
- React 18 + TypeScript
- Three.js / React Three Fiber (3D visualization)
- Recharts + D3.js (charts)
- Vite (build tool)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment
```bash
docker-compose up -d
```

---

## Project Structure

```
Dypsten/
├── backend/
│   ├── data/              # Data ingestion & synthetic generation
│   ├── models/            # ML models (LSTM, CNN, ensemble)
│   ├── digital_twin/      # 3D terrain & stress simulation
│   ├── alerts/            # Alert engine & delivery system
│   ├── api/               # FastAPI server
│   ├── config/            # Configuration files
│   └── tests/             # Unit & integration tests
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── utils/         # Utilities
│   └── public/            # Static assets
├── docs/                  # Documentation
└── scripts/               # Deployment & utility scripts
```

---

## Development Roadmap

- [x] Phase 1: Project Setup ✅
- [ ] Phase 2: Data Pipeline & Synthetic Generator
- [ ] Phase 3: Digital Twin Construction
- [ ] Phase 4: ML Model Development
- [ ] Phase 5: Alert & Decision Engine
- [ ] Phase 6: Frontend Dashboard
- [ ] Phase 7: Integration & Testing
- [ ] Phase 8: Safety Validation
- [ ] Phase 9: Documentation & Deployment
- [ ] Phase 10: Hardware Integration Prep

---

## Safety & Redundancy

Dypsten is designed with multiple layers of safety:
- **Model Redundancy**: LSTM + CNN + Physics-based rules
- **Data Redundancy**: Synthetic + historical datasets
- **Alert Redundancy**: Multi-channel delivery with retry logic
- **Fail-Safe Defaults**: Conservative alerts if confidence drops

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions, issues, or contributions, please open an issue on GitHub.
