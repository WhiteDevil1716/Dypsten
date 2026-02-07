"""
Dypsten Configuration Management
Loads settings from environment variables with sensible defaults
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application Settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application
    app_name: str = "Dypsten"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dypsten"
    postgres_user: str = "dypsten_user"
    postgres_password: str = "changeme"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ML Models
    model_checkpoint_dir: str = "models/checkpoints"
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    cnn_num_filters: int = 64
    ensemble_weights: List[float] = [0.4, 0.4, 0.2]  # LSTM, CNN, Physics
    
    # Risk Thresholds
    risk_threshold_low: int = 25
    risk_threshold_medium: int = 60
    risk_threshold_high: int = 85
    risk_threshold_critical: int = 100
    
    # Alert Settings
    alert_sms_enabled: bool = False
    alert_email_enabled: bool = True
    alert_push_enabled: bool = False
    
    # Twilio (SMS)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    
    # SendGrid (Email)
    sendgrid_api_key: str = ""
    sendgrid_from_email: str = "alerts@dypsten.com"
    
    # Firebase (Push Notifications)
    firebase_credentials_path: str = ""
    
    # Physics Parameters (Geotechnical)
    rock_cohesion_min: float = 100.0  # kPa
    rock_cohesion_max: float = 500.0  # kPa
    friction_angle_min: float = 30.0  # degrees
    friction_angle_max: float = 45.0  # degrees
    factor_of_safety_critical: float = 1.2
    
    # Data Generation
    synthetic_data_enabled: bool = True
    synthetic_noise_level: float = 0.1
    
    # Monitoring
    prometheus_enabled: bool = True
    prometheus_port: int = 9090


# Global settings instance
settings = Settings()
