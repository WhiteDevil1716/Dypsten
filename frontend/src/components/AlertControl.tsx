import React, { useState, useEffect } from 'react';
import './AlertControl.css';

interface PendingAlert {
    id: string;
    type: 'AI' | 'MANUAL';
    priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    message: string;
    confidence?: number;
    timestamp: number;
    autoApproveAt: number;
}

const AlertControl: React.FC = () => {
    const [pendingAlerts, setPendingAlerts] = useState<PendingAlert[]>([]);
    const [manualMessage, setManualMessage] = useState('');
    const [manualPriority, setManualPriority] = useState<'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'>('MEDIUM');
    const [currentTime, setCurrentTime] = useState(Date.now());

    const AUTO_APPROVE_MINUTES = 5;

    // Update current time every second for countdown
    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentTime(Date.now());
        }, 1000);

        return () => clearInterval(interval);
    }, []);

    // Check for auto-approvals
    useEffect(() => {
        const checkAutoApprove = setInterval(() => {
            const now = Date.now();
            setPendingAlerts(prev => {
                const toApprove = prev.filter(alert => alert.autoApproveAt <= now);
                const remaining = prev.filter(alert => alert.autoApproveAt > now);

                // Auto-approve alerts
                toApprove.forEach(alert => {
                    sendAlert(alert);
                });

                return remaining;
            });
        }, 1000);

        return () => clearInterval(checkAutoApprove);
    }, []);

    // Simulate AI alert suggestions
    useEffect(() => {
        const simulateAIAlert = setInterval(() => {
            // Random chance for AI alert
            if (Math.random() > 0.7) {
                const priorities: ('LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL')[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
                const messages = [
                    'Unusual vibration pattern detected in sector A',
                    'Soil moisture increasing rapidly',
                    'Acoustic anomaly detected',
                    'Tilt angle deviation from baseline'
                ];

                const newAlert: PendingAlert = {
                    id: `ai-${Date.now()}`,
                    type: 'AI',
                    priority: priorities[Math.floor(Math.random() * priorities.length)],
                    message: messages[Math.floor(Math.random() * messages.length)],
                    confidence: Math.floor(Math.random() * 30 + 70), // 70-100%
                    timestamp: Date.now(),
                    autoApproveAt: Date.now() + (AUTO_APPROVE_MINUTES * 60 * 1000)
                };

                setPendingAlerts(prev => [...prev, newAlert]);
            }
        }, 20000); // Every 20 seconds (for demo)

        return () => clearInterval(simulateAIAlert);
    }, []);

    const sendAlert = async (alert: PendingAlert) => {
        try {
            await fetch('http://localhost:8000/api/alerts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    priority: alert.priority,
                    message: alert.message,
                    source: alert.type,
                    confidence: alert.confidence
                })
            });
        } catch (error) {
            console.error('Failed to send alert:', error);
        }
    };

    const handleManualAlert = () => {
        if (!manualMessage.trim()) return;

        const alert: PendingAlert = {
            id: `manual-${Date.now()}`,
            type: 'MANUAL',
            priority: manualPriority,
            message: manualMessage,
            timestamp: Date.now(),
            autoApproveAt: Date.now() // Immediate
        };

        sendAlert(alert);
        setManualMessage('');
    };

    const approveAlert = (id: string) => {
        const alert = pendingAlerts.find(a => a.id === id);
        if (alert) {
            sendAlert(alert);
            setPendingAlerts(prev => prev.filter(a => a.id !== id));
        }
    };

    const rejectAlert = (id: string) => {
        setPendingAlerts(prev => prev.filter(a => a.id !== id));
    };

    const getTimeRemaining = (autoApproveAt: number): string => {
        const remaining = Math.max(0, autoApproveAt - currentTime);
        const minutes = Math.floor(remaining / 60000);
        const seconds = Math.floor((remaining % 60000) / 1000);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };

    return (
        <div className="alert-control">
            {/* Manual Alert Section */}
            <div className="control-section">
                <h3 className="section-title">MANUAL ALERT</h3>

                <div className="form-group">
                    <label>PRIORITY</label>
                    <div className="priority-buttons">
                        {(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] as const).map(p => (
                            <button
                                key={p}
                                className={`priority-btn ${manualPriority === p ? 'active' : ''} priority-${p.toLowerCase()}`}
                                onClick={() => setManualPriority(p)}
                            >
                                {p}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="form-group">
                    <label>MESSAGE</label>
                    <textarea
                        value={manualMessage}
                        onChange={(e) => setManualMessage(e.target.value)}
                        placeholder="Enter alert message..."
                        rows={3}
                    />
                </div>

                <button
                    className="submit-btn"
                    onClick={handleManualAlert}
                    disabled={!manualMessage.trim()}
                >
                    SEND ALERT
                </button>
            </div>

            {/* AI Pending Alerts */}
            <div className="control-section">
                <h3 className="section-title">
                    AI ALERTS
                    {pendingAlerts.length > 0 && (
                        <span className="badge">{pendingAlerts.length}</span>
                    )}
                </h3>

                {pendingAlerts.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-icon">AI</div>
                        <p>No pending AI alerts</p>
                    </div>
                ) : (
                    <div className="pending-alerts">
                        {pendingAlerts.map(alert => (
                            <div key={alert.id} className={`pending-alert priority-${alert.priority.toLowerCase()}`}>
                                <div className="alert-header-row">
                                    <div className="alert-priority">{alert.priority}</div>
                                    {alert.confidence && (
                                        <div className="alert-confidence">
                                            {alert.confidence}% CONF
                                        </div>
                                    )}
                                </div>

                                <div className="alert-message">
                                    {alert.message}
                                </div>

                                <div className="alert-timer">
                                    <span className="timer-label">AUTO-APPROVE IN</span>
                                    <span className="timer-value">{getTimeRemaining(alert.autoApproveAt)}</span>
                                </div>

                                <div className="alert-actions">
                                    <button
                                        className="approve-btn"
                                        onClick={() => approveAlert(alert.id)}
                                    >
                                        APPROVE
                                    </button>
                                    <button
                                        className="reject-btn"
                                        onClick={() => rejectAlert(alert.id)}
                                    >
                                        REJECT
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default AlertControl;
