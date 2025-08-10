"""
Configuration management for the Pump.fun token ranking system.
"""

from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path
import os


class DataConfig(BaseSettings):
    """Data ingestion and storage configuration."""
    
    # Data paths
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data storage")
    processed_dir: Path = Field(default=Path("data/processed"), description="Processed data storage")
    features_dir: Path = Field(default=Path("data/features"), description="Feature storage")
    snapshots_dir: Path = Field(default=Path("data/snapshots"), description="Historical snapshots")
    
    # Data retention
    max_snapshot_age_hours: int = Field(default=168, description="Maximum snapshot age in hours")
    snapshot_interval_minutes: int = Field(default=15, description="Snapshot interval in minutes")
    
    # Feature computation windows
    feature_windows: List[int] = Field(default=[1, 5, 15, 60], description="Feature windows in minutes")
    
    class Config:
        env_prefix = "DATA_"


class ModelConfig(BaseSettings):
    """Model training and serving configuration."""
    
    # Model types to train
    model_families: List[str] = Field(
        default=["glm", "xgboost", "lightgbm", "catboost"],
        description="Model families to train"
    )
    
    # Training parameters
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    test_size: float = Field(default=0.2, description="Test set size")
    validation_size: float = Field(default=0.2, description="Validation set size")
    
    # Cross-validation
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    temporal_cv: bool = Field(default=True, description="Use temporal cross-validation")
    
    # Calibration
    calibration_method: str = Field(default="isotonic", description="Calibration method")
    
    # Model registry
    model_registry_dir: Path = Field(default=Path("models"), description="Model registry directory")
    model_version_format: str = Field(default="v{date}_{version}", description="Model version format")
    
    class Config:
        env_prefix = "MODEL_"


class FeatureConfig(BaseSettings):
    """Feature engineering configuration."""
    
    # Feature families
    feature_families: List[str] = Field(
        default=["chain_flow", "wallet_quality", "social", "content", "image", "regime"],
        description="Feature families to compute"
    )
    
    # Chain flow features
    chain_flow_windows: List[int] = Field(default=[1, 5, 15, 60], description="Chain flow windows in minutes")
    holder_concentration_buckets: List[float] = Field(default=[0.1, 0.25, 0.5, 0.75, 0.9], description="Holder concentration buckets")
    
    # Wallet quality features
    wallet_quality_lookback_days: int = Field(default=30, description="Wallet quality lookback period")
    bot_detection_threshold: float = Field(default=0.7, description="Bot detection threshold")
    
    # Social features
    social_platforms: List[str] = Field(default=["twitter", "telegram", "discord"], description="Social platforms to monitor")
    sentiment_lookback_hours: int = Field(default=24, description="Sentiment lookback period")
    
    # Content features
    nlp_embedding_model: str = Field(default="all-MiniLM-L6-v2", description="NLP embedding model")
    content_max_length: int = Field(default=1000, description="Maximum content length for processing")
    
    # Image features
    image_embedding_model: str = Field(default="clip-vit-base-patch32", description="Image embedding model")
    image_max_size: int = Field(default=512, description="Maximum image size for processing")
    
    # Feature preprocessing
    outlier_winsorize_quantile: float = Field(default=0.99, description="Winsorization quantile for outliers")
    imputation_method: str = Field(default="learned", description="Missing value imputation method")
    
    # Regime features
    regime_volatility_buckets: List[float] = Field(default=[0.1, 0.25, 0.5, 1.0], description="Volatility regime buckets")
    
    class Config:
        env_prefix = "FEATURE_"


class LabelConfig(BaseSettings):
    """Label generation configuration."""
    
    # Primary targets
    primary_targets: List[str] = Field(
        default=["return_1h", "return_6h", "return_24h", "liquidity_survival", "sustained_activity", "downside_control"],
        description="Primary prediction targets"
    )
    
    # Return targets
    return_thresholds: List[float] = Field(default=[0.1, 0.25, 0.5, 1.0], description="Return thresholds for binary targets")
    return_horizons: List[int] = Field(default=[1, 6, 24], description="Return horizons in hours")
    
    # Liquidity survival
    liquidity_survival_threshold: float = Field(default=1000, description="Liquidity survival threshold in USD")
    liquidity_survival_horizon: int = Field(default=6, description="Liquidity survival horizon in hours")
    
    # Sustained activity
    activity_percentile: float = Field(default=0.75, description="Activity percentile threshold")
    activity_duration_hours: int = Field(default=6, description="Activity duration requirement")
    
    # Downside control
    max_drawdown_threshold: float = Field(default=0.3, description="Maximum drawdown threshold")
    drawdown_horizon: int = Field(default=6, description="Drawdown measurement horizon")
    
    # Market regime conditioning
    regime_volatility_buckets: List[float] = Field(default=[0.1, 0.25, 0.5, 1.0], description="Volatility regime buckets")
    
    class Config:
        env_prefix = "LABEL_"


class ServingConfig(BaseSettings):
    """Model serving configuration."""
    
    # API settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of worker processes")
    
    # Scoring settings
    batch_size: int = Field(default=100, description="Batch size for scoring")
    max_concurrent_requests: int = Field(default=50, description="Maximum concurrent requests")
    
    # Data freshness
    max_data_age_minutes: int = Field(default=5, description="Maximum data age for scoring")
    data_freshness_check: bool = Field(default=True, description="Enable data freshness checks")
    
    # Output format
    include_confidence_intervals: bool = Field(default=True, description="Include confidence intervals")
    include_shap_values: bool = Field(default=True, description="Include SHAP values")
    include_feature_importance: bool = Field(default=True, description="Include feature importance")
    
    class Config:
        env_prefix = "SERVING_"


class EvaluationConfig(BaseSettings):
    """Model evaluation configuration."""
    
    # Results storage
    results_dir: Path = Field(default=Path("evaluation_results"), description="Evaluation results directory")
    results_retention_days: int = Field(default=90, description="Results retention period in days")
    
    # Evaluation metrics
    primary_metrics: List[str] = Field(
        default=["rmse", "mae", "r2", "precision", "recall", "f1"],
        description="Primary metrics to compute"
    )
    
    # Cross-validation
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    temporal_cv: bool = Field(default=True, description="Use temporal cross-validation")
    
    # Performance thresholds
    min_performance_threshold: float = Field(default=0.6, description="Minimum performance threshold")
    degradation_threshold: float = Field(default=0.1, description="Performance degradation threshold")
    
    class Config:
        env_prefix = "EVALUATION_"


class MonitoringConfig(BaseSettings):
    """Monitoring and drift detection configuration."""
    
    # Drift detection
    drift_detection_enabled: bool = Field(default=True, description="Enable drift detection")
    psi_threshold: float = Field(default=0.1, description="PSI threshold for drift detection")
    ks_threshold: float = Field(default=0.05, description="KS test threshold for drift detection")
    
    # Calibration monitoring
    calibration_monitoring_enabled: bool = Field(default=True, description="Enable calibration monitoring")
    calibration_bins: int = Field(default=10, description="Number of calibration bins")
    max_calibration_error: float = Field(default=0.05, description="Maximum calibration error")
    
    # Auto-retraining
    auto_retrain_enabled: bool = Field(default=True, description="Enable automatic retraining")
    retrain_threshold: float = Field(default=0.15, description="Drift threshold for auto-retraining")
    min_retrain_interval_hours: int = Field(default=24, description="Minimum retraining interval")
    
    # Metrics collection
    metrics_interval_minutes: int = Field(default=15, description="Metrics collection interval")
    metrics_retention_days: int = Field(default=30, description="Metrics retention period")
    
    class Config:
        env_prefix = "MONITORING_"


class Config(BaseSettings):
    """Main configuration class combining all sub-configs."""
    
    # Environment
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Sub-configs
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Paths
    base_dir: Path = Field(default=Path("."), description="Base project directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")
    
    @validator("base_dir", "logs_dir", "cache_dir", pre=True)
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
config = Config()

# Ensure all required directories exist
for path in [config.data.data_dir, config.logs_dir, config.cache_dir, config.model.model_registry_dir, config.evaluation.results_dir]:
    path.mkdir(parents=True, exist_ok=True)
