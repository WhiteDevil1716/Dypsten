"""
Alert Delivery System
Multi-channel alert delivery with SMS, Email, and WebSocket
"""
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import json
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    """Alert delivery channels"""
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"


class AlertPriority(str, Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    priority: AlertPriority
    risk_level: str
    risk_score: float
    probability: float
    message: str
    recommendation: str
    explanations: List[str]
    confidence: float
    channels: List[AlertChannel]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class AlertManager:
    """Central alert management system"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.subscribers: Dict[AlertChannel, List] = {
            AlertChannel.WEBSOCKET: [],
            AlertChannel.SMS: [],
            AlertChannel.EMAIL: [],
            AlertChannel.PUSH: [],
        }
        logger.info("Alert Manager initialized")
    
    def create_alert(
        self,
        risk_level: str,
        risk_score: float,
        probability: float,
        message: str,
        recommendation: str,
        explanations: List[str],
        confidence: float
    ) -> Alert:
        """
        Create new alert from prediction
        
        Args:
            risk_level: LOW/MEDIUM/HIGH/CRITICAL
            risk_score: 0-100
            probability: 0-1
            message: Alert message
            recommendation: Actionable recommendation
            explanations: List of explanation strings
            confidence: Model confidence 0-1
            
        Returns:
            Alert object
        """
        # Determine priority and channels based on risk level
        priority, channels = self._determine_priority_and_channels(risk_level, risk_score)
        
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            priority=priority,
            risk_level=risk_level,
            risk_score=risk_score,
            probability=probability,
            message=message,
            recommendation=recommendation,
            explanations=explanations,
            confidence=confidence,
            channels=channels
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.info(f"Created alert {alert_id}: {risk_level} (score: {risk_score:.1f})")
        
        return alert
    
    def _determine_priority_and_channels(
        self,
        risk_level: str,
        risk_score: float
    ) -> tuple:
        """Determine alert priority and delivery channels"""
        if risk_level == "CRITICAL" or risk_score >= 85:
            return (
                AlertPriority.CRITICAL,
                [AlertChannel.SMS, AlertChannel.EMAIL, AlertChannel.PUSH, AlertChannel.WEBSOCKET]
            )
        elif risk_level == "HIGH" or risk_score >= 60:
            return (
                AlertPriority.HIGH,
                [AlertChannel.EMAIL, AlertChannel.PUSH, AlertChannel.WEBSOCKET]
            )
        elif risk_level == "MEDIUM" or risk_score >= 25:
            return (
                AlertPriority.MEDIUM,
                [AlertChannel.WEBSOCKET, AlertChannel.EMAIL]
            )
        else:
            return (
                AlertPriority.LOW,
                [AlertChannel.WEBSOCKET]
            )
    
    async def deliver_alert(self, alert: Alert):
        """
        Deliver alert through all specified channels
        
        Args:
            alert: Alert to deliver
        """
        logger.info(f"Delivering alert {alert.id} via {len(alert.channels)} channels")
        
        tasks = []
        for channel in alert.channels:
            if channel == AlertChannel.SMS:
                tasks.append(self._deliver_sms(alert))
            elif channel == AlertChannel.EMAIL:
                tasks.append(self._deliver_email(alert))
            elif channel == AlertChannel.PUSH:
                tasks.append(self._deliver_push(alert))
            elif channel == AlertChannel.WEBSOCKET:
                tasks.append(self._deliver_websocket(alert))
        
        # Deliver in parallel
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Alert {alert.id} delivered successfully")
    
    async def _deliver_sms(self, alert: Alert):
        """Deliver SMS alert via Twilio"""
        try:
            # In production, use Twilio client
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=self._format_sms(alert),
            #     from_='+1234567890',
            #     to='+1234567890'
            # )
            
            logger.info(f"SMS sent for alert {alert.id}: {self._format_sms(alert)[:50]}...")
        except Exception as e:
            logger.error(f"SMS delivery failed for {alert.id}: {e}")
    
    async def _deliver_email(self, alert: Alert):
        """Deliver email alert via SMTP"""
        try:
            # In production, use smtplib or sendgrid
            # import smtplib
            # from email.message import EmailMessage
            # msg = EmailMessage()
            # msg['Subject'] = f"[{alert.priority.upper()}] Dypsten Alert"
            # msg['From'] = 'alerts@dypsten.com'
            # msg['To'] = 'operator@mine.com'
            # msg.set_content(self._format_email(alert))
            # smtp.send_message(msg)
            
            logger.info(f"Email sent for alert {alert.id}")
        except Exception as e:
            logger.error(f"Email delivery failed for {alert.id}: {e}")
    
    async def _deliver_push(self, alert: Alert):
        """Deliver push notification via Firebase"""
        try:
            # In production, use Firebase Cloud Messaging
            # from firebase_admin import messaging
            # message = messaging.Message(
            #     notification=messaging.Notification(
            #         title=f"Dypsten {alert.risk_level}",
            #         body=alert.message
            #     ),
            #     data={'alert_id': alert.id}
            # )
            # messaging.send(message)
            
            logger.info(f"Push notification sent for alert {alert.id}")
        except Exception as e:
            logger.error(f"Push delivery failed for {alert.id}: {e}")
    
    async def _deliver_websocket(self, alert: Alert):
        """Broadcast alert to WebSocket subscribers"""
        try:
            alert_data = {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "priority": alert.priority,
                "risk_level": alert.risk_level,
                "risk_score": alert.risk_score,
                "message": alert.message,
                "recommendation": alert.recommendation,
                "confidence": alert.confidence
            }
            
            # In production, broadcast via WebSocket
            # for websocket in self.subscribers[AlertChannel.WEBSOCKET]:
            #     await websocket.send_json(alert_data)
            
            logger.info(f"WebSocket broadcast for alert {alert.id}")
        except Exception as e:
            logger.error(f"WebSocket delivery failed for {alert.id}: {e}")
    
    def _format_sms(self, alert: Alert) -> str:
        """Format alert for SMS (160 chars)"""
        return f"🚨 DYPSTEN {alert.risk_level}: {alert.message[:100]}. {alert.recommendation[:50]}"
    
    def _format_email(self, alert: Alert) -> str:
        """Format alert for email"""
        return f"""
Dypsten Rockfall Early Warning System
{'='*50}

ALERT: {alert.risk_level}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Risk Score: {alert.risk_score:.1f}/100
Probability: {alert.probability:.1%}
Confidence: {alert.confidence:.1%}

MESSAGE:
{alert.message}

RECOMMENDATION:
{alert.recommendation}

ANALYSIS:
{chr(10).join(f"• {exp}" for exp in alert.explanations)}

---
Alert ID: {alert.id}
"""
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts"""
        return [a for a in self.alerts.values() if not a.acknowledged]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history"""
        return self.alert_history[-limit:]


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    manager = AlertManager()
    
    # Create mock alert
    alert = manager.create_alert(
        risk_level="HIGH",
        risk_score=75.0,
        probability=0.75,
        message="Elevated ground vibration and soil moisture detected in Sector B",
        recommendation="⚠️ EVACUATE NON-ESSENTIAL PERSONNEL - High instability detected",
        explanations=[
            "Vibration: 6.5 mm/s → HIGH",
            "Soil moisture: 72% → HIGH",
            "Tilt rate: 0.06°/hr → HIGH",
            "Factor of Safety: 1.4 → HIGH"
        ],
        confidence=0.87
    )
    
    # Deliver alert
    asyncio.run(manager.deliver_alert(alert))
    
    # Show active alerts
    active = manager.get_active_alerts()
    print(f"\nActive alerts: {len(active)}")
    for a in active:
        print(f"  {a.id}: {a.risk_level} - {a.message[:60]}...")
    
    # Acknowledge
    manager.acknowledge_alert(alert.id, "operator@mine.com")
    print(f"\nAlert {alert.id} acknowledged")
