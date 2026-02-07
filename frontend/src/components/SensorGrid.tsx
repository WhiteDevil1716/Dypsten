import React, { useState, useEffect } from 'react';
import './SensorGrid.css';

interface SensorReading {
    value: number;
    status: 'normal' | 'warning' | 'critical';
    trend: 'up' | 'down' | 'stable';
}

interface Sensors {
    geophone: SensorReading;
    inclinometer: SensorReading;
    moisture: SensorReading;
    acoustic: SensorReading;
}

const SensorGrid: React.FC = () => {
    const [sensors, setSensors] = useState<Sensors>({
        geophone: { value: 2.3, status: 'normal', trend: 'stable' },
        inclinometer: { value: 0.045, status: 'normal', trend: 'up' },
        moisture: { value: 42, status: 'normal', trend: 'down' },
        acoustic: { value: 28, status: 'normal', trend: 'stable' },
    });

    useEffect(() => {
        // Simulate live sensor updates
        const interval = setInterval(() => {
            setSensors(prev => ({
                geophone: updateSensor(prev.geophone, 0, 15, 10),
                inclinometer: updateSensor(prev.inclinometer, 0, 0.2, 0.1),
                moisture: updateSensor(prev.moisture, 0, 100, 80),
                acoustic: updateSensor(prev.acoustic, 0, 150, 100),
            }));
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    const updateSensor = (
        current: SensorReading,
        min: number,
        max: number,
        warningThreshold: number
    ): SensorReading => {
        const change = (Math.random() - 0.5) * (max - min) * 0.05;
        const newValue = Math.max(min, Math.min(max, current.value + change));

        let status: 'normal' | 'warning' | 'critical' = 'normal';
        if (newValue > warningThreshold * 1.2) status = 'critical';
        else if (newValue > warningThreshold) status = 'warning';

        let trend: 'up' | 'down' | 'stable' = 'stable';
        if (Math.abs(change) > (max - min) * 0.01) {
            trend = change > 0 ? 'up' : 'down';
        }

        return { value: newValue, status, trend };
    };

    const getTrendSymbol = (trend: string): string => {
        switch (trend) {
            case 'up': return '▲';
            case 'down': return '▼';
            default: return '━';
        }
    };

    const sensorConfig = [
        {
            key: 'geophone' as keyof Sensors,
            icon: 'VIB',
            label: 'Ground Vibration',
            unit: 'mm/s',
            decimals: 1,
        },
        {
            key: 'inclinometer' as keyof Sensors,
            icon: 'TILT',
            label: 'Tilt Angle',
            unit: '°',
            decimals: 3,
        },
        {
            key: 'moisture' as keyof Sensors,
            icon: 'MSTR',
            label: 'Soil Moisture',
            unit: '%',
            decimals: 0,
        },
        {
            key: 'acoustic' as keyof Sensors,
            icon: 'ACST',
            label: 'Acoustic Events',
            unit: '/hr',
            decimals: 0,
        },
    ];

    return (
        <div className="sensor-grid">
            {sensorConfig.map((config) => {
                const sensor = sensors[config.key];

                return (
                    <div key={config.key} className={`sensor-card ${sensor.status}`}>
                        <div className="sensor-header">
                            <div className="sensor-info">
                                <div className="sensor-icon">{config.icon}</div>
                                <h4>{config.label}</h4>
                            </div>
                            <div className={`sensor-trend ${sensor.trend}`}>
                                {getTrendSymbol(sensor.trend)}
                            </div>
                        </div>

                        <div className="sensor-value">
                            {sensor.value.toFixed(config.decimals)}
                            <span className="sensor-unit">{config.unit}</span>
                        </div>

                        <div className="sensor-status">
                            {sensor.status.toUpperCase()}
                        </div>

                        <div className="sensor-bar">
                            <div
                                className="sensor-bar-fill"
                                style={{ width: `${Math.min(100, (sensor.value / 15) * 100)}%` }}
                            ></div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default SensorGrid;
