import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';
import DigitalTwin from './DigitalTwin';
import RiskGauge from './RiskGauge';
import PredictionChart from './PredictionChart';
import AlertFeed from './AlertFeed';
import SensorGrid from './SensorGrid';
import AlertControl from './AlertControl';

interface SystemStatus {
    status: string;
    models: {
        ensemble: string;
        physics_guardrail: string;
    };
    active_alerts: number;
    websocket_clients: number;
}

interface Alert {
    id: string;
    timestamp: string;
    priority: string;
    risk_level: string;
    risk_score: number;
    message: string;
    recommendation: string;
    acknowledged: boolean;
}

const Dashboard: React.FC = () => {
    const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
    const [currentRisk, setCurrentRisk] = useState(25);
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [ws, setWs] = useState<WebSocket | null>(null);
    const [connected, setConnected] = useState(false);

    // Connect to WebSocket
    useEffect(() => {
        const websocket = new WebSocket('ws://localhost:8000/ws');

        websocket.onopen = () => {
            console.log('WebSocket connected');
            setConnected(true);
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('WebSocket message:', data);

            if (data.type === 'alert') {
                setAlerts(prev => [data, ...prev]);
                setCurrentRisk(data.risk_score);
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            setConnected(false);
        };

        websocket.onclose = () => {
            console.log('WebSocket disconnected');
            setConnected(false);
        };

        setWs(websocket);

        return () => {
            websocket.close();
        };
    }, []);

    // Fetch system status
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/status');
                const data = await response.json();
                setSystemStatus(data);
            } catch (error) {
                console.error('Failed to fetch status:', error);
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 5000);
        return () => clearInterval(interval);
    }, []);

    // Fetch alerts
    useEffect(() => {
        const fetchAlerts = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/alerts?limit=10');
                const data = await response.json();
                setAlerts(data);
            } catch (error) {
                console.error('Failed to fetch alerts:', error);
            }
        };

        fetchAlerts();
    }, []);

    const handleAlertAcknowledge = async (alertId: string) => {
        try {
            await fetch(`http://localhost:8000/api/alerts/${alertId}/acknowledge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ acknowledged_by: 'operator' })
            });

            setAlerts(prev => prev.map(alert =>
                alert.id === alertId ? { ...alert, acknowledged: true } : alert
            ));
        } catch (error) {
            console.error('Failed to acknowledge alert:', error);
        }
    };

    return (
        <div className="dashboard">
            {/* Clean Header */}
            <header className="dashboard-header">
                <div className="header-content">
                    <div className="header-left">
                        <Link to="/" className="logo-link">
                            <span className="logo-icon">🏔️</span>
                            <h1>DYPSTEN</h1>
                        </Link>
                        <span className="header-divider">|</span>
                        <span className="header-subtitle">LIVE DASHBOARD</span>
                    </div>

                    <div className="header-right">
                        <div className="status-indicator">
                            <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
                            <span>{connected ? 'LIVE' : 'OFFLINE'}</span>
                        </div>

                        <div className="user-profile">
                            <div className="user-avatar">OP</div>
                            <div className="user-info">
                                <div className="user-name">Operator</div>
                                <div className="user-role">Admin</div>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Grid */}
            <main className="dashboard-main">
                <div className="dashboard-grid">
                    {/* Digital Twin - Large */}
                    <div className="panel digital-twin-panel">
                        <div className="panel-header">
                            <h2 className="panel-title">DIGITAL TWIN</h2>
                        </div>
                        <div className="panel-body">
                            <DigitalTwin />
                        </div>
                    </div>

                    {/* Risk Panel */}
                    <div className="panel risk-panel">
                        <div className="panel-header">
                            <h2 className="panel-title">RISK STATUS</h2>
                        </div>
                        <div className="panel-body">
                            <RiskGauge value={currentRisk} />
                        </div>
                    </div>

                    {/* Prediction Chart */}
                    <div className="panel chart-panel">
                        <div className="panel-header">
                            <h2 className="panel-title">PREDICTION TIMELINE</h2>
                        </div>
                        <div className="panel-body">
                            <PredictionChart />
                        </div>
                    </div>

                    {/* Sensors */}
                    <div className="panel sensors-panel">
                        <div className="panel-header">
                            <h2 className="panel-title">LIVE SENSORS</h2>
                        </div>
                        <div className="panel-body">
                            <SensorGrid />
                        </div>
                    </div>

                    {/* Alert Control */}
                    <div className="panel alert-control-panel">
                        <div className="panel-header">
                            <h2 className="panel-title">ALERT CONTROL</h2>
                        </div>
                        <div className="panel-body">
                            <AlertControl />
                        </div>
                    </div>

                    {/* Alerts */}
                    <div className="panel alerts-panel">
                        <div className="panel-header">
                            <h2 className="panel-title">ALERT FEED</h2>
                            <div className="header-actions">
                                <span className="alert-count">{alerts.filter(a => !a.acknowledged).length} UNACK</span>
                            </div>
                        </div>
                        <div className="panel-body">
                            <AlertFeed alerts={alerts} onAcknowledge={handleAlertAcknowledge} />
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Dashboard;
