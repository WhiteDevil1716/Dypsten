import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import './LandingPage.css';

const LandingPage: React.FC = () => {
    return (
        <div className="landing-page">
            {/* Navigation */}
            <nav className="nav-bar">
                <div className="nav-content">
                    <div className="nav-logo">
                        <span className="logo-icon">🏔️</span>
                        <span className="logo-text">DYPSTEN</span>
                    </div>

                    <div className="nav-links">
                        <a href="#features">Technology</a>
                        <a href="#capabilities">Capabilities</a>
                        <a href="#about">About</a>
                        <Link to="/dashboard" className="nav-cta">Launch Dashboard</Link>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-bg">
                    <img
                        src="/assets/hero-slope.png"
                        alt="Slope Monitoring"
                        className="hero-bg-image"
                    />
                    <div className="hero-overlay"></div>
                </div>

                <div className="hero-content">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <h1 className="hero-title">
                            Are you ready for<br />
                            <span className="highlight-orange">perfectly safe</span><br />
                            mining operations?
                        </h1>

                        <p className="hero-subtitle">
                            AI-powered rockfall prediction with digital twin visualization.<br />
                            Predict slope instability 60 minutes in advance.
                        </p>

                        <div className="hero-actions">
                            <Link to="/dashboard" className="btn-primary">
                                View Live Dashboard
                            </Link>
                            <a href="#features" className="btn-secondary">
                                Learn More
                            </a>
                        </div>
                    </motion.div>

                    <motion.div
                        className="hero-stats"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, duration: 0.8 }}
                    >
                        <div className="stat-item">
                            <div className="stat-value">60min</div>
                            <div className="stat-label">Prediction Horizon</div>
                        </div>
                        <div className="stat-divider"></div>
                        <div className="stat-item">
                            <div className="stat-value">95%</div>
                            <div className="stat-label">Accuracy Rate</div>
                        </div>
                        <div className="stat-divider"></div>
                        <div className="stat-item">
                            <div className="stat-value">24/7</div>
                            <div className="stat-label">Real-time Monitoring</div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="features-section">
                <div className="section-header">
                    <span className="section-tag">AI-POWERED EARLY WARNING</span>
                    <h2 className="section-title">
                        Risk Prediction<br />
                        <span className="highlight-orange">Redefined</span>
                    </h2>
                </div>

                <div className="features-grid">
                    <motion.div
                        className="feature-card large"
                        initial={{ opacity: 0, x: -30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6 }}
                    >
                        <div className="feature-image">
                            <img
                                src="/assets/ai-prediction.jpg"
                                alt="AI Prediction"
                            />
                        </div>
                        <div className="feature-content">
                            <h3>Hybrid AI Engine</h3>
                            <p>
                                LSTM + CNN + Physics-informed models working in ensemble.<br />
                                Drift detection and confidence scoring ensure reliable predictions.
                            </p>
                            <ul className="feature-list">
                                <li>✓ 10X Faster than traditional methods</li>
                                <li>✓ AI-driven anomaly detection</li>
                                <li>✓ Continuous  learning from sensor data</li>
                            </ul>
                        </div>
                    </motion.div>

                    <motion.div
                        className="feature-card"
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                    >
                        <img src="/assets/icon-digital-twin.png" alt="Digital Twin" className="feature-icon-img" />
                        <h3>Digital Twin</h3>
                        <p>Real-time 3D terrain visualization with stress analysis and risk heat mapping</p>
                    </motion.div>

                    <motion.div
                        className="feature-card"
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.3 }}
                    >
                        <img src="/assets/icon-iot-sensors.png" alt="IoT Sensors" className="feature-icon-img" />
                        <h3>IoT Sensors</h3>
                        <p>Geophone, inclinometer, moisture, and acoustic emission sensors integrated</p>
                    </motion.div>

                    <motion.div
                        className="feature-card"
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.4 }}
                    >
                        <img src="/assets/icon-alerts.png" alt="Alerts" className="feature-icon-img" />
                        <h3>Multi-Channel Alerts</h3>
                        <p>SMS, Email, Push notifications with escalation protocols and acknowledgment</p>
                    </motion.div>

                    <motion.div
                        className="feature-card"
                        initial={{ opacity: 0, x: 30 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.5 }}
                    >
                        <img src="/assets/icon-ai-engine.png" alt="AI Engine" className="feature-icon-img" />
                        <h3>Predictive AI Engine</h3>
                        <p>Advanced machine learning models with ensemble techniques for accurate predictions</p>
                    </motion.div>
                </div>
            </section>

            {/* Capabilities Section */}
            <section id="capabilities" className="capabilities-section">
                <div className="section-header">
                    <span className="section-tag orange">COMPREHENSIVE MONITORING</span>
                    <h2 className="section-title">
                        Built for<br />
                        <span className="highlight-orange">Mission-Critical Reliability</span>
                    </h2>
                </div>

                <div className="capabilities-grid">
                    {[
                        { icon: '⚡', title: 'Up to 10X Faster', desc: 'Real-time processing with edge computing capabilities' },
                        { icon: '🔄', title: 'Automated', desc: 'Continuous monitoring with minimal human intervention' },
                        { icon: '📈', title: 'Scalable', desc: 'From single slopes to entire mine operations' },
                        { icon: '🎯', title: 'High Precision', desc: 'Validated accuracy for critical applications' },
                        { icon: '🔐', title: 'Secure', desc: 'Enterprise-grade security and data encryption' },
                        { icon: '🌐', title: 'Cloud Ready', desc: 'Deploy on-premise or in the cloud' },
                    ].map((cap, idx) => (
                        <motion.div
                            key={idx}
                            className="capability-card"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: idx * 0.1, duration: 0.5 }}
                        >
                            <div className="capability-icon">{cap.icon}</div>
                            <h4>{cap.title}</h4>
                            <p>{cap.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* Technology Showcase */}
            <section className="tech-section">
                <div className="tech-content">
                    <div className="tech-text">
                        <span className="section-tag orange">CUTTING-EDGE TECHNOLOGY</span>
                        <h2>
                            Sensor Network<br />
                            <span className="highlight-orange">Intelligence</span>
                        </h2>
                        <p>
                            Our advanced IoT sensor array continuously monitors ground vibration, tilt,
                            soil moisture, and acoustic emissions. All data streams into our AI engine
                            for comprehensive risk analysis.
                        </p>
                        <ul className="tech-features">
                            <li>
                                <span className="check-icon">✓</span>
                                <div>
                                    <strong>Real-time Processing</strong>
                                    <p>Sub-second latency from sensor to prediction</p>
                                </div>
                            </li>
                            <li>
                                <span className="check-icon">✓</span>
                                <div>
                                    <strong>Mesh Network</strong>
                                    <p>Self-healing sensor communication</p>
                                </div>
                            </li>
                            <li>
                                <span className="check-icon">✓</span>
                                <div>
                                    <strong>Edge Computing</strong>
                                    <p>Local processing for critical alerts</p>
                                </div>
                            </li>
                        </ul>
                    </div>
                    <div className="tech-visual">
                        <img
                            src="/assets/sensor-network.png"
                            alt="Sensor Network"
                        />
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="cta-section">
                <div className="cta-content">
                    <h2>Ready to revolutionize<br />slope safety?</h2>
                    <p>Experience the future of rockfall prediction</p>
                    <div className="cta-actions">
                        <Link to="/dashboard" className="btn-primary large">
                            Launch Dashboard
                        </Link>
                        <a href="mailto:contact@dypsten.com" className="btn-secondary large">
                            Request Demo
                        </a>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="footer">
                <div className="footer-content">
                    <div className="footer-brand">
                        <div className="footer-logo">
                            <span className="logo-icon">🏔️</span>
                            <span className="logo-text">DYPSTEN</span>
                        </div>
                        <p>AI-Driven Rockfall Early Warning System</p>
                    </div>

                    <div className="footer-links">
                        <div className="footer-column">
                            <h4>Product</h4>
                            <Link to="/dashboard">Dashboard</Link>
                            <a href="#features">Technology</a>
                            <a href="#capabilities">Capabilities</a>
                        </div>
                        <div className="footer-column">
                            <h4>Company</h4>
                            <a href="#about">About</a>
                            <a href="#contact">Contact</a>
                            <a href="#careers">Careers</a>
                        </div>
                    </div>
                </div>

                <div className="footer-bottom">
                    <p>© 2026 Dypsten. All rights reserved.</p>
                    <div className="footer-legal">
                        <a href="#privacy">Privacy Policy</a>
                        <a href="#terms">Terms of Service</a>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
