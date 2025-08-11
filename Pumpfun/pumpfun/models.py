"""
Token ranking models for the Pump.fun system.

This module implements various ML models for token ranking including:
- GLMs (Generalized Linear Models)
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Ensemble methods
- Model calibration and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pickle
import json
from datetime import datetime, timedelta
import logging

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# import xgboost as xgb  # Temporarily commented out due to OpenMP dependency
import lightgbm as lgb
import catboost as cb

from loguru import logger


class BaseModel:
    """Base class for all token ranking models."""
    
    def __init__(self, name: str, model_type: str, **kwargs):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.is_trained = False
        self.training_metadata = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """Fit the model to training data."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (for classification models)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        raise NotImplementedError
        
    def save(self, path: Path) -> None:
        """Save the model to disk."""
        model_data = {
            'name': self.name,
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load(self, path: Path) -> 'BaseModel':
        """Load the model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        for key, value in model_data.items():
            setattr(self, key, value)
            
        return self


class GLMModel(BaseModel):
    """Generalized Linear Model for token ranking."""
    
    def __init__(self, name: str, task: str = "regression", **kwargs):
        super().__init__(name, "glm", **kwargs)
        self.task = task
        self.task_type = kwargs.get('task_type', 'regression')
        
        if self.task_type == 'regression':
            self.model = Ridge(alpha=kwargs.get('alpha', 1.0))
        else:
            self.model = LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
            
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'GLMModel':
        """Fit the GLM model."""
        logger.info(f"Training GLM model: {self.name}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        # Store training metadata
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'task_type': self.task_type
        }
        
        self.is_trained = True
        logger.info(f"GLM model {self.name} trained successfully")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.task_type == 'regression':
            raise ValueError("Probability prediction not available for regression models")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict_proba(X_scaled)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from coefficients."""
        if not self.is_trained:
            return {}
            
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_.flatten())
        else:
            importance = np.zeros(len(self.feature_names))
            
        return dict(zip(self.feature_names, importance))


class XGBoostModel(BaseModel):
    """XGBoost model for token ranking."""
    
    def __init__(self, name: str, task: str = "regression", **kwargs):
        super().__init__(name, "xgboost", **kwargs)
        self.task = task
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with provided parameters
        default_params.update(kwargs)
        
        if task == "regression":
            self.model = xgb.XGBRegressor(**default_params)
        else:
            self.model = xgb.XGBClassifier(**default_params)
            
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostModel':
        """Fit the XGBoost model."""
        logger.info(f"Training XGBoost model: {self.name}")
        
        self.feature_names = list(X.columns)
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'task_type': self.task
        }
        
        self.is_trained = True
        logger.info(f"XGBoost model {self.name} trained successfully")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.task == "regression":
            raise ValueError("Probability prediction not available for regression models")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict_proba(X_scaled)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost."""
        if not self.is_trained:
            return {}
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LightGBMModel(BaseModel):
    """LightGBM model for token ranking."""
    
    def __init__(self, name: str, task: str = "regression", **kwargs):
        super().__init__(name, "lightgbm", **kwargs)
        self.task = task
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        default_params.update(kwargs)
        
        if task == "regression":
            self.model = lgb.LGBMRegressor(**default_params)
        else:
            self.model = lgb.LGBMClassifier(**default_params)
            
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'LightGBMModel':
        """Fit the LightGBM model."""
        logger.info(f"Training LightGBM model: {self.name}")
        
        self.feature_names = list(X.columns)
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.model.fit(X_scaled, y)
        
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'task_type': self.task
        }
        
        self.is_trained = True
        logger.info(f"LightGBM model {self.name} trained successfully")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.task == "regression":
            raise ValueError("Probability prediction not available for regression models")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict_proba(X_scaled)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from LightGBM."""
        if not self.is_trained:
            return {}
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class CatBoostModel(BaseModel):
    """CatBoost model for token ranking."""
    
    def __init__(self, name: str, task: str = "regression", **kwargs):
        super().__init__(name, "catboost", **kwargs)
        self.task = task
        
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': False
        }
        
        default_params.update(kwargs)
        
        if task == "regression":
            self.model = cb.CatBoostRegressor(**default_params)
        else:
            self.model = cb.CatBoostClassifier(**default_params)
            
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'CatBoostModel':
        """Fit the CatBoost model."""
        logger.info(f"Training CatBoost model: {self.name}")
        
        self.feature_names = list(X.columns)
        
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.model.fit(X_scaled, y)
        
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'task_type': self.task
        }
        
        self.is_trained = True
        logger.info(f"CatBoost model {self.name} trained successfully")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.task == "regression":
            raise ValueError("Probability prediction not available for regression models")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict_proba(X_scaled)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from CatBoost."""
        if not self.is_trained:
            return {}
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, name: str, base_models: List[BaseModel], 
                 ensemble_method: str = "weighted_average", **kwargs):
        super().__init__(name, "ensemble", **kwargs)
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleModel':
        """Fit the ensemble model."""
        logger.info(f"Training ensemble model: {self.name}")
        
        # Train all base models
        for model in self.base_models:
            model.fit(X, y, **kwargs)
            
        # Compute ensemble weights based on validation performance
        if self.ensemble_method == "weighted_average":
            self._compute_weights(X, y)
            
        self.feature_names = list(X.columns)
        self.is_trained = True
        
        logger.info(f"Ensemble model {self.name} trained successfully")
        return self
        
    def _compute_weights(self, X: pd.DataFrame, y: pd.Series):
        """Compute optimal weights for base models."""
        # Use cross-validation to estimate model performance
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for model in self.base_models:
            model_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = r2_score(y_val, pred) if hasattr(y_val, 'dtype') and np.issubdtype(y_val.dtype, np.number) else roc_auc_score(y_val, pred)
                model_scores.append(score)
                
            scores.append(np.mean(model_scores))
            
        # Convert scores to weights (softmax)
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))
        self.weights = exp_scores / exp_scores.sum()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
            
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        if self.ensemble_method == "weighted_average":
            return np.average(predictions, axis=0, weights=self.weights)
        else:
            return np.mean(predictions, axis=0)
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
            
        probabilities = []
        for model in self.base_models:
            try:
                prob = model.predict_proba(X)
                probabilities.append(prob)
            except:
                # Convert regression predictions to probabilities if needed
                pred = model.predict(X)
                prob = self._regression_to_probability(pred)
                probabilities.append(prob)
                
        probabilities = np.array(probabilities)
        
        if self.ensemble_method == "weighted_average":
            return np.average(probabilities, axis=0, weights=self.weights)
        else:
            return np.mean(probabilities, axis=0)
            
    def _regression_to_probability(self, predictions: np.ndarray) -> np.ndarray:
        """Convert regression predictions to probabilities."""
        # Simple sigmoid transformation
        return 1 / (1 + np.exp(-predictions))
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance."""
        if not self.is_trained:
            return {}
            
        # Aggregate feature importance from base models
        importance_dict = {}
        
        for i, model in enumerate(self.base_models):
            model_importance = model.get_feature_importance()
            weight = self.weights[i] if self.weights is not None else 1.0
            
            for feature, importance in model_importance.items():
                if feature not in importance_dict:
                    importance_dict[feature] = 0.0
                importance_dict[feature] += importance * weight
                
        return importance_dict


class TokenRankingModel:
    """Main token ranking model orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.model_performance = {}
        self.feature_names = None
        
        # Auto-load models if models directory exists
        models_dir = Path(config.get('model_registry_dir', 'models'))
        if models_dir.exists():
            self.load_models(models_dir)
            
    def create_model(self, model_name: str, model_type: str, **kwargs) -> BaseModel:
        """Create a new model instance."""
        if model_type == "glm":
            return GLMModel(model_name, **kwargs)
        elif model_type == "xgboost":
            return XGBoostModel(model_name, **kwargs)
        elif model_type == "lightgbm":
            return LightGBMModel(model_name, **kwargs)
        elif model_type == "catboost":
            return CatBoostModel(model_name, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    model_configs: List[Dict[str, Any]]) -> Dict[str, BaseModel]:
        """Train multiple models with different configurations."""
        logger.info("Starting model training pipeline")
        
        self.feature_names = list(X.columns)
        
        for config in model_configs:
            model_name = config['name']
            model_type = config['type']
            
            logger.info(f"Training {model_type} model: {model_name}")
            
            # Create and train model
            model = self.create_model(model_name, model_type, **config.get('params', {}))
            model.fit(X, y)
            
            # Store model
            self.models[model_name] = model
            
            # Evaluate model
            self._evaluate_model(model_name, X, y)
            
        logger.info(f"Training completed. {len(self.models)} models trained.")
        return self.models
        
    def _evaluate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series):
        """Evaluate a trained model."""
        model = self.models[model_name]
        
        # Adaptive cross-validation based on sample size
        n_samples = len(X)
        if n_samples < 3:
            # Too few samples for cross-validation
            logger.warning(f"Too few samples ({n_samples}) for cross-validation. Skipping evaluation.")
            self.model_performance[model_name] = {
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'cv_scores': [],
                'note': 'Insufficient samples for cross-validation'
            }
            return
        
        # Use fewer splits for small datasets
        n_splits = min(3, n_samples - 1)  # At most 3 splits, and ensure we have enough samples
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        try:
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Retrain on this fold
                temp_model = self.create_model(model_name, model.model_type)
                temp_model.fit(X_train, y_train)
                
                # Predict and score
                pred = temp_model.predict(X_val)
                
                if hasattr(y_val, 'dtype') and np.issubdtype(y_val.dtype, np.number):
                    score = r2_score(y_val, pred)
                else:
                    score = roc_auc_score(y_val, pred)
                    
                cv_scores.append(score)
                
            # Store performance metrics
            self.model_performance[model_name] = {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'cv_scores': cv_scores
            }
            
            logger.info(f"Model {model_name} - CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Skipping evaluation.")
            self.model_performance[model_name] = {
                'cv_mean': np.nan,
                'cv_std': np.nan,
                'cv_scores': [],
                'note': f'Cross-validation failed: {str(e)}'
            }
        
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Make predictions using trained models."""
        if not self.models:
            raise ValueError("No models available for prediction")
            
        predictions = {}
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            predictions[model_name] = self.models[model_name].predict(X)
        else:
            # Use all models
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
                
        return predictions
        
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get feature importance from models."""
        if not self.models:
            return {}
            
        importance = {}
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            importance[model_name] = self.models[model_name].get_feature_importance()
        else:
            for name, model in self.models.items():
                importance[name] = model.get_feature_importance()
                
        return importance
        
    def save_models(self, save_dir: Path):
        """Save all trained models to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = save_dir / f"{name}.pkl"
            model.save(model_path)
            
        # Save metadata
        metadata = {
            'model_performance': self.model_performance,
            'feature_names': self.feature_names,
            'save_date': datetime.now().isoformat()
        }
        
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Models saved to {save_dir}")
        
    def load_models(self, load_dir: Path):
        """Load trained models from disk."""
        # Load metadata
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            self.model_performance = metadata['model_performance']
            self.feature_names = metadata['feature_names']
        
        # Load models
        for model_file in load_dir.glob("*.pkl"):
            if model_file.name == "metadata.json":
                continue
                
            try:
                model_name = model_file.stem
                
                # Load the model data to determine its type
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create appropriate model instance based on the loaded data
                model_type = model_data.get('model_type', 'unknown')
                if model_type == 'glm':
                    model = GLMModel(model_name)
                elif model_type == 'xgboost':
                    model = XGBoostModel(model_name)
                elif model_type == 'lightgbm':
                    model = LightGBMModel(model_name)
                elif model_type == 'catboost':
                    model = CatBoostModel(model_name)
                elif model_type == 'ensemble':
                    model = EnsembleModel(model_name, [])
                else:
                    # Fallback: try to determine type from filename or create generic
                    logger.warning(f"Unknown model type for {model_name}, creating generic model")
                    model = BaseModel(model_name, "generic")
                
                # Load the actual model data
                model.load(model_file)
                self.models[model_name] = model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
                continue
            
        logger.info(f"Loaded {len(self.models)} models from {load_dir}")


class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_dir = Path(config.get('model_registry_dir', 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry state
        self.registered_models = {}
        self.model_metadata = {}
        
        # Load existing models
        self._load_registered_models()
        
    def _load_registered_models(self):
        """Load all registered models from disk."""
        if not self.models_dir.exists():
            return
            
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create appropriate model instance
                model_type = model_data.get('model_type', 'unknown')
                if model_type == 'glm':
                    model = GLMModel(model_name)
                elif model_type == 'xgboost':
                    model = XGBoostModel(model_name)
                elif model_type == 'lightgbm':
                    model = LightGBMModel(model_name)
                elif model_type == 'catboost':
                    model = CatBoostModel(model_name)
                elif model_type == 'ensemble':
                    model = EnsembleModel(model_name, [])
                else:
                    continue
                
                # Load model data
                model.load(model_file)
                self.registered_models[model_name] = model
                
                # Load metadata
                metadata_file = model_file.parent / f"{model_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.model_metadata[model_name] = json.load(f)
                        
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {e}")
                
    def register_model(self, model: BaseModel, metadata: Optional[Dict[str, Any]] = None):
        """Register a trained model."""
        self.registered_models[model.name] = model
        
        if metadata:
            self.model_metadata[model.name] = metadata
            
        # Save model to disk
        model_path = self.models_dir / f"{model.name}.pkl"
        model.save(model_path)
        
        # Save metadata
        if metadata:
            metadata_path = self.models_dir / f"{model.name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        logger.info(f"Model {model.name} registered successfully")
        
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get a registered model by name."""
        return self.registered_models.get(model_name)
        
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.registered_models.keys())
        
    def remove_model(self, model_name: str):
        """Remove a model from the registry."""
        if model_name in self.registered_models:
            del self.registered_models[model_name]
            
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]
            
        # Remove files
        model_file = self.models_dir / f"{model_name}.pkl"
        metadata_file = self.models_dir / f"{model_name}_metadata.json"
        
        if model_file.exists():
            model_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
            
        logger.info(f"Model {model_name} removed from registry")
        
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        return self.model_metadata.get(model_name)
        
    def update_model_metadata(self, model_name: str, metadata: Dict[str, Any]):
        """Update metadata for a specific model."""
        if model_name in self.registered_models:
            self.model_metadata[model_name] = metadata
            
            # Save updated metadata
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get a summary of the model registry."""
        return {
            'total_models': len(self.registered_models),
            'model_names': list(self.registered_models.keys()),
            'model_types': {
                name: model.model_type for name, model in self.registered_models.items()
            },
            'trained_models': {
                name: model.is_trained for name, model in self.registered_models.items()
            }
        }
