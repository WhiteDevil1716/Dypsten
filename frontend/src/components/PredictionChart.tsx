import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import './PredictionChart.css';

const PredictionChart: React.FC = () => {
    const [data, setData] = useState<any[]>([]);

    useEffect(() => {
        // Generate mock time series data
        const generateData = () => {
            const now = Date.now();
            const points = [];

            for (let i = 60; i >= 0; i--) {
                const time = new Date(now - i * 60000); // 1-hour history
                const baseRisk = 20 + Math.sin(i / 10) * 15;
                const noise = Math.random() * 5;

                points.push({
                    time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    risk: Math.max(0, Math.min(100, baseRisk + noise)),
                    confidence: 85 + Math.random() * 10,
                    threshold: 60, // High risk threshold
                });
            }

            return points;
        };

        setData(generateData());

        // Update every minute
        const interval = setInterval(() => {
            setData(generateData());
        }, 60000);

        return () => clearInterval(interval);
    }, []);

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            return (
                <div className="custom-tooltip">
                    <p className="tooltip-time">{payload[0].payload.time}</p>
                    <p className="tooltip-risk">
                        Risk: <span style={{ color: getRiskColor(payload[0].value) }}>
                            {payload[0].value.toFixed(1)}%
                        </span>
                    </p>
                    <p className="tooltip-confidence">
                        Confidence: {payload[1]?.value?.toFixed(1)}%
                    </p>
                </div>
            );
        }
        return null;
    };

    const getRiskColor = (value: number): string => {
        if (value < 25) return '#22c55e';
        if (value < 60) return '#f59e0b';
        if (value < 85) return '#ef4444';
        return '#dc2626';
    };

    return (
        <div className="prediction-chart">
            <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                    </defs>

                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />

                    <XAxis
                        dataKey="time"
                        stroke="rgba(255, 255, 255, 0.3)"
                        tick={{ fill: 'rgba(255, 255, 255, 0.5)', fontSize: 12 }}
                        interval="preserveStartEnd"
                    />

                    <YAxis
                        stroke="rgba(255, 255, 255, 0.3)"
                        tick={{ fill: 'rgba(255, 255, 255, 0.5)', fontSize: 12 }}
                        domain={[0, 100]}
                    />

                    <Tooltip content={<CustomTooltip />} />

                    {/* Threshold line */}
                    <Line
                        type="monotone"
                        dataKey="threshold"
                        stroke="#ef4444"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                    />

                    {/* Confidence area */}
                    <Area
                        type="monotone"
                        dataKey="confidence"
                        stroke="#8b5cf6"
                        strokeWidth={1}
                        fillOpacity={1}
                        fill="url(#confidenceGradient)"
                    />

                    {/* Risk line */}
                    <Area
                        type="monotone"
                        dataKey="risk"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        fillOpacity={1}
                        fill="url(#riskGradient)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

export default PredictionChart;
