"""
Ranking service API for the Pump.fun token ranking system.

This module implements a FastAPI-based service for serving token rankings
with real-time scoring, model management, and monitoring.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from .models import TokenRankingModel
from .features import FeatureStore
from .data import DataConnector, DataStorage
from .config import config


def to_native(x):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    return x


class TokenRequest(BaseModel):
    """Request model for token ranking."""
    token_addresses: List[str] = Field(..., description="List of token addresses to rank")
    include_features: bool = Field(default=False, description="Include computed features in response")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")
    include_shap: bool = Field(default=False, description="Include SHAP values")
    model_name: Optional[str] = Field(default=None, description="Specific model to use for prediction")


class TokenResponse(BaseModel):
    """Response model for token ranking."""
    token_address: str
    ranking_score: float
    confidence_interval: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    shap_values: Optional[Dict[str, float]] = None
    features: Optional[Dict[str, Any]] = None
    prediction_timestamp: str
    model_used: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_status: Dict[str, Any]
    data_freshness: Dict[str, Any]
    system_metrics: Dict[str, Any]


class RankingService:
    """Main ranking service for token predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(
            title="Pump.fun Token Ranking Service",
            description="AI-powered token ranking and prediction service",
            version="0.1.0"
        )
        
        # Initialize components
        # FeatureStore doesn't need config parameter
        self.feature_store = FeatureStore()
        # DataConnector and DataStorage expect config_dict
        self.data_connector = DataConnector(config)
        self.data_storage = DataStorage(config)
        # TokenRankingModel expects config
        self.ranking_model = TokenRankingModel(config)
        
        # Service state
        self.is_healthy = True
        self.last_data_update = None
        self.model_performance = {}
        self.request_count = 0
        self.error_count = 0
        
        # Setup API routes
        self._setup_routes()
        self._setup_middleware()
        
    def _setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "Pump.fun Token Ranking Service",
                "version": "0.1.0",
                "status": "running"
            }
            
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return await self._get_health_status()
            
        @self.app.post("/rank")
        async def rank_tokens(request: TokenRequest):
            """Rank tokens and return predictions."""
            result = await self._rank_tokens(request)
            return JSONResponse(content=to_native(result))
            
        @self.app.get("/models", response_model=Dict[str, Any])
        async def list_models():
            """List available models and their performance."""
            return self._get_models_info()
            
        @self.app.post("/models/reload")
        async def reload_models():
            """Reload models from disk."""
            return await self._reload_models()
            
        @self.app.get("/features/{token_address}")
        async def get_token_features(token_address: str):
            """Get computed features for a specific token."""
            return await self._get_token_features(token_address)
            
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self._get_service_metrics()
            
        @self.app.post("/refresh")
        async def refresh_data(background_tasks: BackgroundTasks):
            """Refresh data and features."""
            background_tasks.add_task(self._refresh_data_background)
            return {"message": "Data refresh started in background"}
            
    async def _rank_tokens(self, request: TokenRequest) -> List[TokenResponse]:
        """Rank tokens based on their features and model predictions."""
        try:
            logger.info(f"Processing ranking request for {len(request.token_addresses)} tokens")
            
            # Add timeout to prevent hanging on external API calls
            import asyncio
            try:
                # Set a 10-second timeout for the entire ranking operation
                result = await asyncio.wait_for(
                    self._process_ranking_request(request), 
                    timeout=10.0
                )
                return result
            except asyncio.TimeoutError:
                logger.error("Ranking request timed out after 10 seconds")
                raise HTTPException(
                    status_code=408, 
                    detail="Ranking request timed out. This may be due to slow external API responses or VPN routing issues."
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in token ranking: {e}")
            raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")
    
    async def _process_ranking_request(self, request: TokenRequest) -> List[TokenResponse]:
        """Process the actual ranking request."""
        # Get features for tokens
        features = await self._get_features_for_tokens(request.token_addresses)
        
        if features.empty:
            raise HTTPException(status_code=404, detail="No features found for provided tokens")
        
        # Make predictions using the ranking model
        predictions = {}
        for model_name in self.ranking_model.models.keys():
            try:
                # Prepare features for prediction - only include columns the model was trained on
                # Get the feature names from the model metadata
                model_feature_names = self.ranking_model.feature_names
                if model_feature_names:
                    # Filter features to only include columns the model knows about
                    available_features = [col for col in model_feature_names if col in features.columns]
                    if available_features:
                        X = features[available_features].fillna(0)
                        logger.info(f"Using {len(available_features)} features for prediction (filtered from {len(features.columns)} total)")
                    else:
                        logger.warning(f"No matching features found for model {model_name}")
                        continue
                else:
                    # Fallback: exclude obvious non-feature columns
                    feature_cols = [col for col in features.columns if col not in ['token_address', 'token_id', 'timestamp']]
                    X = features[feature_cols].fillna(0)
                    logger.info(f"Using fallback feature selection: {len(feature_cols)} features")
                
                # Ensure we have data to predict on - fix DataFrame boolean logic
                if X.empty:
                    logger.warning(f"No valid features for prediction with model {model_name}")
                    continue
                
                if len(X) == 0:
                    logger.warning(f"Empty feature matrix for model {model_name}")
                    continue
                
                # Make prediction
                pred_dict = self.ranking_model.predict(X, model_name)
                
                # Extract the prediction array from the dictionary
                if pred_dict and model_name in pred_dict:
                    pred_array = pred_dict[model_name]
                    if pred_array is not None and len(pred_array) > 0:
                        predictions[model_name] = pred_array
                        logger.info(f"Successfully generated predictions with model {model_name}")
                    else:
                        logger.warning(f"Model {model_name} returned empty predictions")
                else:
                    logger.warning(f"Model {model_name} not found in prediction results")
                
            except Exception as e:
                logger.error(f"Failed to make prediction with model {model_name}: {e}")
                continue
        
        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to generate predictions with any model")
        
        # Build responses
        responses = []
        for token_address in request.token_addresses:
            # Filter features for this specific token
            token_features = features[features['token_address'] == token_address]
            if not token_features.empty:
                token_features_series = token_features.iloc[0]
                response = await self._build_token_response(
                    token_address, 
                    token_features_series, 
                    predictions, 
                    request.model_name,
                    request.include_confidence, 
                    request.include_shap
                )
                responses.append(response)
        
        return responses
            
    async def _get_features_for_tokens(self, token_addresses: List[str]) -> pd.DataFrame:
        """Get features for the specified tokens."""
        # Try to get from feature store first
        features = await self.feature_store.get_features_batch(token_addresses)
        
        if features.empty:
            # Fallback to computing features on-demand
            logger.info("Features not found in store, computing on-demand")
            features = await self._compute_features_on_demand(token_addresses)
            
        return features
        
    async def _compute_features_on_demand(self, token_addresses: List[str]) -> pd.DataFrame:
        """Compute features for tokens on-demand."""
        # Get raw data for tokens
        raw_data = await self.data_connector.get_token_data_batch(token_addresses)
        
        if raw_data.empty:
            return pd.DataFrame()
            
        # Compute features
        features = await self.feature_store.compute_features_batch(raw_data)
        
        return features
        
    async def _build_token_response(self, token_address: str, features: Optional[pd.Series],
                                  predictions: Dict[str, np.ndarray], model_name: Optional[str],
                                  include_confidence: bool, include_shap: bool) -> TokenResponse:
        """Build response for a single token."""
        # Determine which model to use
        if model_name and model_name in predictions:
            model_key = model_name
        else:
            # Use first available model
            model_key = list(predictions.keys())[0]
            
        # Handle different prediction result types
        pred_result = predictions[model_key]
        
        # Extract the ranking score based on the type of pred_result
        if isinstance(pred_result, np.ndarray):
            if pred_result.size == 1:
                # Single value array
                ranking_score = float(pred_result.item())
            elif len(pred_result) > 0:
                # Multiple values, take the first one
                ranking_score = float(pred_result[0])
            else:
                raise ValueError(f"Empty prediction result from model {model_key}")
        elif isinstance(pred_result, (list, tuple)):
            if len(pred_result) > 0:
                ranking_score = float(pred_result[0])
            else:
                raise ValueError(f"Empty prediction result from model {model_key}")
        elif isinstance(pred_result, (int, float)):
            ranking_score = float(pred_result)
        else:
            raise ValueError(f"Unexpected prediction result type: {type(pred_result)} from model {model_key}")
        
        # Build response
        response = TokenResponse(
            token_address=token_address,
            ranking_score=ranking_score,
            prediction_timestamp=datetime.now().isoformat(),
            model_used=model_key
        )
        
        # Add optional fields
        if include_confidence:
            response.confidence_interval = self._calculate_confidence_interval(
                predictions, ranking_score
            )
            
        if include_shap:
            response.shap_values = self._get_shap_values(token_address, features, model_key)
            
        if features is not None:
            response.features = features.to_dict()
            
        # Get feature importance
        feature_importance = self.ranking_model.get_feature_importance(model_key)
        if feature_importance:
            response.feature_importance = feature_importance
            
        return response
        
    def _calculate_confidence_interval(self, predictions: Dict[str, np.ndarray], 
                                     score: float) -> Dict[str, float]:
        """Calculate confidence interval from multiple model predictions."""
        all_predictions = np.array(list(predictions.values())).flatten()
        
        # Simple confidence interval based on prediction variance
        mean_pred = np.mean(all_predictions)
        std_pred = np.std(all_predictions)
        
        return {
            "lower": float(mean_pred - 1.96 * std_pred),
            "upper": float(mean_pred + 1.96 * std_pred),
            "confidence_level": 0.95
        }
        
    def _get_shap_values(self, token_address: str, features: Optional[pd.Series], 
                         model_name: str) -> Dict[str, float]:
        """Get SHAP values for feature explanation."""
        # This would integrate with SHAP library for actual values
        # For now, return feature importance as proxy
        if features is None:
            return {}
            
        feature_importance = self.ranking_model.get_feature_importance(model_name)
        if not feature_importance:
            return {}
            
        # Return top features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:10])  # Top 10 features
        
    async def _get_health_status(self) -> HealthResponse:
        """Get comprehensive health status."""
        # Check model status
        model_status = {
            "models_loaded": len(self.ranking_model.models),
            "model_names": list(self.ranking_model.models.keys()),
            "last_training": self._get_last_training_time()
        }
        
        # Check data freshness
        data_freshness = {
            "last_update": self.last_data_update.isoformat() if self.last_data_update else None,
            "data_age_minutes": self._get_data_age_minutes(),
            "feature_store_status": await self._check_feature_store_status()
        }
        
        # System metrics
        system_metrics = {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "uptime_seconds": time.time() - self._get_start_time()
        }
        
        return HealthResponse(
            status="healthy" if self.is_healthy else "unhealthy",
            timestamp=datetime.now().isoformat(),
            model_status=model_status,
            data_freshness=data_freshness,
            system_metrics=system_metrics
        )
        
    def _get_last_training_time(self) -> Optional[str]:
        """Get last model training time."""
        if not self.ranking_model.models:
            return None
            
        # Get training time from first model
        first_model = list(self.ranking_model.models.values())[0]
        if hasattr(first_model, 'training_metadata'):
            return first_model.training_metadata.get('training_date')
        return None
        
    def _get_data_age_minutes(self) -> Optional[float]:
        """Get age of most recent data in minutes."""
        if not self.last_data_update:
            return None
        return (datetime.now() - self.last_data_update).total_seconds() / 60
        
    async def _check_feature_store_status(self) -> str:
        """Check feature store status."""
        try:
            # Simple check - try to get a small sample
            sample_features = await self.feature_store.get_features_batch(["sample_token"])
            return "operational"
        except Exception:
            return "error"
            
    def _get_start_time(self) -> float:
        """Get service start time."""
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        return self._start_time
        
    def _get_models_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        models_info = {}
        
        for name, model in self.ranking_model.models.items():
            models_info[name] = {
                "type": model.model_type,
                "is_trained": model.is_trained,
                "n_features": len(model.feature_names) if model.feature_names else 0,
                "training_date": model.training_metadata.get('training_date', 'unknown')
            }
            
        # Add performance metrics
        models_info["performance"] = self.ranking_model.model_performance
        
        return models_info
        
    async def _reload_models(self) -> Dict[str, str]:
        """Reload models from disk."""
        try:
            model_dir = Path(self.config.get('model_registry_dir', 'models'))
            self.ranking_model.load_models(model_dir)
            
            logger.info("Models reloaded successfully")
            return {"status": "success", "message": "Models reloaded"}
            
        except Exception as e:
            logger.error(f"Failed to reload models: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    async def _get_token_features(self, token_address: str) -> Dict[str, Any]:
        """Get features for a specific token."""
        try:
            features = await self.feature_store.get_features_batch([token_address])
            
            if features.empty:
                raise HTTPException(status_code=404, detail="Token not found")
                
            return {
                "token_address": token_address,
                "features": features.iloc[0].to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting features for {token_address}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    def _get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        return {
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "success_rate": (self.request_count - self.error_count) / max(self.request_count, 1)
            },
            "performance": {
                "uptime_seconds": time.time() - self._get_start_time(),
                "requests_per_minute": self._calculate_requests_per_minute()
            },
            "models": {
                "loaded": len(self.ranking_model.models),
                "performance": self.ranking_model.model_performance
            }
        }
        
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute rate."""
        uptime_minutes = (time.time() - self._get_start_time()) / 60
        return self.request_count / max(uptime_minutes, 1)
        
    async def _refresh_data_background(self):
        """Background task to refresh data and features."""
        try:
            logger.info("Starting background data refresh")
            
            # Refresh raw data
            await self.data_connector.refresh_data()
            
            # Update features
            await self.feature_store.refresh_features()
            
            # Update timestamp
            self.last_data_update = datetime.now()
            
            logger.info("Background data refresh completed")
            
        except Exception as e:
            logger.error(f"Background data refresh failed: {str(e)}")
            self.is_healthy = False
            
    def start(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Start the ranking service."""
        logger.info(f"Starting Pump.fun Token Ranking Service on {host}:{port}")
        
        # Load models if available
        try:
            model_dir = Path(self.config.get('model_registry_dir', 'models'))
            if model_dir.exists():
                self.ranking_model.load_models(model_dir)
                logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}")
            
        # Start the service
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )
        
    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance."""
        return self.app


class ModelServer:
    """Model serving server for offline predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ranking_model = TokenRankingModel(config)
        
    def load_models(self, model_dir: Path):
        """Load models from directory."""
        self.ranking_model.load_models(model_dir)
        
    def predict(self, model_name: str, input_data: pd.DataFrame) -> np.ndarray:
        """Make predictions using a specific model."""
        return self.ranking_model.predict(input_data, model_name=model_name)
        
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.ranking_model.models.keys())
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if model_name not in self.ranking_model.models:
            return {}
            
        model = self.ranking_model.models[model_name]
        return {
            'name': model.name,
            'type': model.model_type,
            'is_trained': model.is_trained,
            'feature_names': model.feature_names,
            'training_metadata': model.training_metadata
        }


# Factory function for creating the service
def create_ranking_service(config: Dict[str, Any] = None) -> RankingService:
    """Create a new ranking service instance."""
    if config is None:
        try:
            # Try to get config from the imported config module
            if hasattr(config, 'dict'):
                config_dict = config.dict()
            elif hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            else:
                config_dict = {}
        except Exception:
            config_dict = {}
    else:
        config_dict = config
        
    return RankingService(config_dict)


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pump.fun Token Ranking Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        if args.config:
            # Load from file - implement file loading logic here
            service_config = {}
        else:
            # Use default config
            try:
                if hasattr(config, 'dict'):
                    service_config = config.dict()
                elif hasattr(config, 'model_dump'):
                    service_config = config.model_dump()
                else:
                    service_config = {}
            except Exception:
                service_config = {}
    except Exception as e:
        print(f"Error loading configuration: {e}")
        service_config = {}
        
    # Create and start service
    service = create_ranking_service(service_config)
    service.start(host=args.host, port=args.port)
