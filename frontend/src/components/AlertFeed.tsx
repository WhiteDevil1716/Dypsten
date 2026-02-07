import React from 'react';
import './AlertFeed.css';

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

interface AlertFeedProps {
    alerts: Alert[];
    onAcknowledge: (alertId: string) => void;
}

const AlertFeed: React.FC<AlertFeedProps> = ({ alerts, onAcknowledge }) => {
    const getPriorityIcon = (priority: string): string => {
        switch (priority.toLowerCase()) {
            case 'critical': return '🚨';
            case 'high': return '⚠️';
            case 'medium': return '⚡';
            default: return '📍';
        }
    };

    const getPriorityClass = (priority: string): string => {
        return `priority-${priority.toLowerCase()}`;
    };

    const formatTime = (timestamp: string): string => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    if (alerts.length === 0) {
        return (
            <div className="alert-feed">
                <div className="empty-  state">
                    <div className="empty-icon">✅</div>
                    <p>No active alerts</p>
                    <span>System operating normally</span>
                </div>
            </div>
        );
    }

    return (
        <div className="alert-feed">
            {alerts.map((alert) => (
                <div
                    key={alert.id}
                    className={`alert-card ${getPriorityClass(alert.priority)} ${alert.acknowledged ? 'acknowledged' : ''}`}
                >
                    <div className="alert-header">
                        <div className="alert-icon">
                            {getPriorityIcon(alert.priority)}
                        </div>

                        <div className="alert-info">
                            <div className="alert-title">
                                <span className="alert-level">{alert.risk_level}</span>
                                <span className="alert-score">{alert.risk_score.toFixed(0)}%</span>
                            </div>
                            <div className="alert-time">{formatTime(alert.timestamp)}</div>
                        </div>

                        {!alert.acknowledged && (
                            <button
                                className="ack-button"
                                onClick={() => onAcknowledge(alert.id)}
                            >
                                Acknowledge
                            </button>
                        )}

                        {alert.acknowledged && (
                            <div className="ack-badge">
                                ✓ Acknowledged
                            </div>
                        )}
                    </div>

                    <div className="alert-body">
                        <p className="alert-message">{alert.message}</p>
                        <div className="alert-recommendation">
                            <strong>Recommendation:</strong> {alert.recommendation}
                        </div>
                    </div>

                    <div className="alert-footer">
                        <span className="alert-id">ID: {alert.id}</span>
                    </div>
                </div>
            ))}
        </div>
    );
};

export default AlertFeed;
