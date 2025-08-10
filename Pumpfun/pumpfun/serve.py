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
        self.feature_store = FeatureStore(config)
        self.data_connector = DataConnector(config)
        self.data_storage = DataStorage(config)
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
            
        @self.app.post("/rank", response_model=List[TokenResponse])
        async def rank_tokens(request: TokenRequest):
            """Rank tokens and return predictions."""
            return await self._rank_tokens(request)
            
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
        """Rank tokens and return predictions."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.info(f"Processing ranking request for {len(request.token_addresses)} tokens")
            
            # Validate token addresses
            if not request.token_addresses:
                raise HTTPException(status_code=400, detail="No token addresses provided")
                
            # Get features for tokens
            features_df = await self._get_features_for_tokens(request.token_addresses)
            
            if features_df.empty:
                raise HTTPException(status_code=404, detail="No features found for provided tokens")
                
            # Make predictions
            predictions = self.ranking_model.predict(
                features_df, 
                model_name=request.model_name
            )
            
            # Build responses
            responses = []
            for i, token_address in enumerate(request.token_addresses):
                if token_address in features_df.index:
                    response = await self._build_token_response(
                        token_address=token_address,
                        features=features_df.loc[token_address] if request.include_features else None,
                        predictions=predictions,
                        model_name=request.model_name,
                        include_confidence=request.include_confidence,
                        include_shap=request.include_shap
                    )
                    responses.append(response)
                    
            # Log performance
            processing_time = time.time() - start_time
            logger.info(f"Ranking completed in {processing_time:.3f}s for {len(responses)} tokens")
            
            return responses
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in token ranking: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")
            
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
            
        ranking_score = float(predictions[model_key][0])
        
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
        config = config.dict() if hasattr(config, 'dict') else {}
        
    return RankingService(config)


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pump.fun Token Ranking Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load from file
        pass
    else:
        # Use default config
        service_config = config.dict()
        
    # Create and start service
    service = create_ranking_service(service_config)
    service.start(host=args.host, port=args.port)
