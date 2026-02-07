import React from 'react';
import './RiskGauge.css';

interface RiskGaugeProps {
    value: number; // 0-100
}

const RiskGauge: React.FC<RiskGaugeProps> = ({ value }) => {
    const getRiskLevel = (val: number): string => {
        if (val < 25) return 'LOW';
        if (val < 60) return 'MEDIUM';
        if (val < 85) return 'HIGH';
        return 'CRITICAL';
    };

    const getRiskColor = (val: number): string => {
        if (val < 25) return '#22c55e';
        if (val < 60) return '#f59e0b';
        if (val < 85) return '#ef4444';
        return '#dc2626';
    };

    const level = getRiskLevel(value);
    const color = getRiskColor(value);

    return (
        <div className="risk-gauge">
            {/* Large Value Display */}
            <div className="gauge-main">
                <div className="gauge-value" style={{ color }}>
                    {value}
                </div>
                <div className="gauge-unit">%</div>
            </div>

            {/* Status Label */}
            <div className="gauge-label">
                RISK LEVEL
            </div>

            <div className="gauge-status" style={{
                color,
                borderColor: color
            }}>
                {level}
            </div>

            {/* Simple Progress Bar */}
            <div className="gauge-bar">
                <div
                    className="gauge-bar-fill"
                    style={{
                        width: `${value}%`,
                        backgroundColor: color
                    }}
                ></div>
            </div>

            {/* Stats Grid */}
            <div className="gauge-stats">
                <div className="stat-item">
                    <div className="stat-label">CONFIDENCE</div>
                    <div className="stat-value">87%</div>
                </div>
                <div className="stat-item">
                    <div className="stat-label">HORIZON</div>
                    <div className="stat-value">60min</div>
                </div>
                <div className="stat-item">
                    <div className="stat-label">STATUS</div>
                    <div className="stat-value">STABLE</div>
                </div>
            </div>
        </div>
    );
};

export default RiskGauge;
