"""
Model evaluation and performance assessment for the Pump.fun token ranking system.

This module implements comprehensive evaluation metrics, cross-validation,
and performance analysis for the ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, GridSearchCV,
    RandomizedSearchCV, validation_curve
)
import warnings
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict

from .config import config
from .models import ModelRegistry
from .features import FeatureEngineer


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Regression metrics
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    
    # Classification metrics (if applicable)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None
    
    # Additional metrics
    feature_importance: Optional[Dict[str, float]] = None
    prediction_bias: Optional[float] = None
    residual_std: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelPerformance:
    """Container for model performance results."""
    
    model_name: str
    model_version: str
    evaluation_timestamp: datetime
    metrics: EvaluationMetrics
    training_time: float
    prediction_time: float
    model_size_mb: float
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['evaluation_timestamp'] = self.evaluation_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformance':
        """Create from dictionary."""
        data['evaluation_timestamp'] = datetime.fromisoformat(data['evaluation_timestamp'])
        return cls(**data)


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config.evaluation.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model_registry = ModelRegistry(config.model_dump())
        self.feature_engineer = FeatureEngineer(config)
        
        # Evaluation state
        self.evaluation_history = self._load_evaluation_history()
        self.current_evaluation = None
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _load_evaluation_history(self) -> Dict[str, List[ModelPerformance]]:
        """Load evaluation history from disk."""
        history_file = self.results_dir / "evaluation_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                history = {}
                for model_name, performance_list in history_data.items():
                    history[model_name] = [ModelPerformance.from_dict(p) for p in performance_list]
                return history
            except Exception as e:
                self.logger.error(f"Failed to load evaluation history: {e}")
                return {}
        return {}
    
    def _save_evaluation_history(self):
        """Save evaluation history to disk."""
        try:
            history_data = {}
            for model_name, performance_list in self.evaluation_history.items():
                history_data[model_name] = [p.to_dict() for p in performance_list]
            
            history_file = self.results_dir / "evaluation_history.json"
            with open(history_file, 'w') as f:
                json.dump(history_data, f, default=str, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save evaluation history: {e}")
    
    def evaluate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                      test_size: float = 0.2, random_state: int = 42,
                      cv_folds: int = 5) -> ModelPerformance:
        """Evaluate a single model comprehensively."""
        self.logger.info(f"Starting evaluation of {model_name}")
        
        # Get model from registry
        model = self.model_registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._split_data(X, y, test_size, random_state)
        
        # Train model and measure time
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions and measure time
        start_time = datetime.now()
        y_pred = model.predict(X_test)
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, model, X_test)
        
        # Get model size
        model_size_mb = self._get_model_size(model)
        
        # Create performance record
        performance = ModelPerformance(
            model_name=model_name,
            model_version=config.model.model_version_format.format(
                date=datetime.now().strftime("%Y.%m.%d"),
                version="01"
            ),
            evaluation_timestamp=datetime.now(),
            metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            model_size_mb=model_size_mb,
            hyperparameters=model.get_params() if hasattr(model, 'get_params') else {},
            feature_names=list(X.columns)
        )
        
        # Store in history
        if model_name not in self.evaluation_history:
            self.evaluation_history[model_name] = []
        self.evaluation_history[model_name].append(performance)
        self._save_evaluation_history()
        
        self.current_evaluation = performance
        self.logger.info(f"Evaluation completed for {model_name}")
        
        return performance
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
        """Split data into train and test sets with time series consideration."""
        # For time series data, use TimeSeriesSplit
        if 'timestamp' in X.columns:
            # Sort by timestamp
            sorted_indices = X['timestamp'].sort_values().index
            X_sorted = X.loc[sorted_indices]
            y_sorted = y.loc[sorted_indices]
            
            # Use time series split
            split_point = int(len(X_sorted) * (1 - test_size))
            X_train = X_sorted.iloc[:split_point]
            X_test = X_sorted.iloc[split_point:]
            y_train = y_sorted.iloc[:split_point]
            y_test = y_sorted.iloc[split_point:]
        else:
            # Use random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        return X_train, X_test, y_train, y_test
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                          model: Any, X_test: pd.DataFrame) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Initialize metrics
        metrics = EvaluationMetrics(
            mse=mse, rmse=rmse, mae=mae, r2=r2, mape=mape
        )
        
        # Check if this is a classification problem
        if len(np.unique(y_true)) <= 10:  # Assume classification if few unique values
            try:
                # Convert to classification metrics
                y_pred_class = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
                
                metrics.precision = precision_score(y_true, y_pred_class, average='weighted', zero_division=0)
                metrics.recall = recall_score(y_true, y_pred_class, average='weighted', zero_division=0)
                metrics.f1 = f1_score(y_true, y_pred_class, average='weighted', zero_division=0)
                
                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        metrics.auc = roc_auc_score(y_true, y_pred)
                    except:
                        pass
            except:
                pass
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
            metrics.feature_importance = feature_importance
        
        # Prediction bias and residual analysis
        residuals = y_true - y_pred
        metrics.prediction_bias = np.mean(residuals)
        metrics.residual_std = np.std(residuals)
        
        return metrics
    
    def _get_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            # Save model to temporary file to get size
            temp_file = self.results_dir / "temp_model.pkl"
            with open(temp_file, 'wb') as f:
                pickle.dump(model, f)
            
            size_bytes = temp_file.stat().st_size
            temp_file.unlink()
            
            return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                           cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """Perform cross-validation for a model."""
        self.logger.info(f"Starting cross-validation for {model_name}")
        
        model = self.model_registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Use TimeSeriesSplit for time series data
        if 'timestamp' in X.columns:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Convert negative MSE to positive
        if scoring == 'neg_mean_squared_error':
            cv_scores = -cv_scores
        
        results = {
            'model_name': model_name,
            'cv_folds': cv_folds,
            'scoring': scoring,
            'cv_scores': cv_scores.tolist(),
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores)
        }
        
        self.logger.info(f"Cross-validation completed for {model_name}")
        return results
    
    def hyperparameter_tuning(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                             param_grid: Dict[str, List], cv_folds: int = 3,
                             n_iter: int = 20, random_state: int = 42) -> Dict[str, Any]:
        """Perform hyperparameter tuning using randomized search."""
        self.logger.info(f"Starting hyperparameter tuning for {model_name}")
        
        model = self.model_registry.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Use TimeSeriesSplit for time series data
        if 'timestamp' in X.columns:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring='neg_mean_squared_error',
            random_state=random_state, n_jobs=-1, verbose=1
        )
        
        random_search.fit(X, y)
        
        results = {
            'model_name': model_name,
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,  # Convert back to positive
            'cv_results': random_search.cv_results_,
            'best_estimator': random_search.best_estimator_
        }
        
        self.logger.info(f"Hyperparameter tuning completed for {model_name}")
        return results
    
    def compare_models(self, model_names: List[str], X: pd.DataFrame, y: pd.Series,
                      test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """Compare multiple models side by side."""
        self.logger.info(f"Starting model comparison for {len(model_names)} models")
        
        comparison_results = []
        
        for model_name in model_names:
            try:
                performance = self.evaluate_model(
                    model_name, X, y, test_size, random_state
                )
                
                comparison_results.append({
                    'model_name': model_name,
                    'rmse': performance.metrics.rmse,
                    'mae': performance.metrics.mae,
                    'r2': performance.metrics.r2,
                    'mape': performance.metrics.mape,
                    'training_time': performance.training_time,
                    'prediction_time': performance.prediction_time,
                    'model_size_mb': performance.model_size_mb
                })
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                comparison_results.append({
                    'model_name': model_name,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'mape': np.nan,
                    'training_time': np.nan,
                    'prediction_time': np.nan,
                    'model_size_mb': np.nan
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('rmse')
        
        self.logger.info("Model comparison completed")
        return comparison_df
    
    def generate_evaluation_report(self, model_name: str, 
                                 save_path: Optional[Path] = None) -> str:
        """Generate a comprehensive evaluation report."""
        if model_name not in self.evaluation_history:
            raise ValueError(f"No evaluation history found for {model_name}")
        
        performance_list = self.evaluation_history[model_name]
        if not performance_list:
            raise ValueError(f"No performance data found for {model_name}")
        
        latest_performance = max(performance_list, key=lambda p: p.evaluation_timestamp)
        
        # Generate report
        report = f"""
# Model Evaluation Report: {model_name}

## Model Information
- **Model Version**: {latest_performance.model_version}
- **Evaluation Timestamp**: {latest_performance.evaluation_timestamp}
- **Training Time**: {latest_performance.training_time:.2f} seconds
- **Prediction Time**: {latest_performance.prediction_time:.4f} seconds
- **Model Size**: {latest_performance.model_size_mb:.2f} MB

## Performance Metrics
- **RMSE**: {latest_performance.metrics.rmse:.4f}
- **MAE**: {latest_performance.metrics.mae:.4f}
- **R² Score**: {latest_performance.metrics.r2:.4f}
- **MAPE**: {latest_performance.metrics.mape:.2f}%

"""
        
        # Add classification metrics if available
        if latest_performance.metrics.precision is not None:
            report += f"""
## Classification Metrics
- **Precision**: {latest_performance.metrics.precision:.4f}
- **Recall**: {latest_performance.metrics.recall:.4f}
- **F1 Score**: {latest_performance.metrics.f1:.4f}
"""
            if latest_performance.metrics.auc is not None:
                report += f"- **AUC**: {latest_performance.metrics.auc:.4f}\n"
        
        # Add additional metrics
        if latest_performance.metrics.prediction_bias is not None:
            report += f"""
## Additional Metrics
- **Prediction Bias**: {latest_performance.metrics.prediction_bias:.4f}
- **Residual Std**: {latest_performance.metrics.residual_std:.4f}
"""
        
        # Add feature importance if available
        if latest_performance.metrics.feature_importance:
            report += "\n## Top Feature Importances\n"
            sorted_features = sorted(
                latest_performance.metrics.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:10]
            
            for feature, importance in sorted_features:
                report += f"- **{feature}**: {importance:.4f}\n"
        
        # Add hyperparameters
        if latest_performance.hyperparameters:
            report += "\n## Hyperparameters\n"
            for param, value in latest_performance.hyperparameters.items():
                report += f"- **{param}**: {value}\n"
        
        # Add historical performance
        if len(performance_list) > 1:
            report += "\n## Historical Performance\n"
            for i, performance in enumerate(performance_list[-5:]):  # Last 5 evaluations
                report += f"- **{performance.evaluation_timestamp.strftime('%Y-%m-%d %H:%M')}**: RMSE={performance.metrics.rmse:.4f}, R²={performance.metrics.r2:.4f}\n"
        
        # Save report if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_performance_trends(self, model_name: str, save_path: Optional[Path] = None):
        """Plot performance trends over time."""
        if model_name not in self.evaluation_history:
            raise ValueError(f"No evaluation history found for {model_name}")
        
        performance_list = self.evaluation_history[model_name]
        if len(performance_list) < 2:
            raise ValueError(f"Insufficient data for trend analysis")
        
        # Sort by timestamp
        performance_list.sort(key=lambda p: p.evaluation_timestamp)
        
        timestamps = [p.evaluation_timestamp for p in performance_list]
        rmse_scores = [p.metrics.rmse for p in performance_list]
        r2_scores = [p.metrics.r2 for p in performance_list]
        training_times = [p.training_time for p in performance_list]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Trends: {model_name}', fontsize=16)
        
        # RMSE trend
        axes[0, 0].plot(timestamps, rmse_scores, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('RMSE Trend')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² trend
        axes[0, 1].plot(timestamps, r2_scores, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('R² Score Trend')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training time trend
        axes[1, 0].plot(timestamps, training_times, 'r-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Training Time Trend')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model size trend
        model_sizes = [p.model_size_mb for p in performance_list]
        axes[1, 1].plot(timestamps, model_sizes, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Model Size Trend')
        axes[1, 1].set_ylabel('Model Size (MB)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Evaluation Timestamp')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20,
                               save_path: Optional[Path] = None):
        """Plot feature importance for a model."""
        if model_name not in self.evaluation_history:
            raise ValueError(f"No evaluation history found for {model_name}")
        
        performance_list = self.evaluation_history[model_name]
        if not performance_list:
            raise ValueError(f"No performance data found for {model_name}")
        
        latest_performance = max(performance_list, key=lambda p: p.evaluation_timestamp)
        
        if not latest_performance.metrics.feature_importance:
            raise ValueError(f"No feature importance data available for {model_name}")
        
        # Sort features by importance
        sorted_features = sorted(
            latest_performance.metrics.feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            plt.text(importance + 0.001, i, f'{importance:.4f}', 
                    va='center', fontsize=10)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances: {model_name}')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis for better readability
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_best_model(self, metric: str = 'rmse') -> Optional[ModelPerformance]:
        """Get the best performing model based on a metric."""
        best_performance = None
        best_score = float('inf') if metric in ['rmse', 'mae', 'mape'] else float('-inf')
        
        for model_name, performance_list in self.evaluation_history.items():
            if not performance_list:
                continue
            
            latest_performance = max(performance_list, key=lambda p: p.evaluation_timestamp)
            score = getattr(latest_performance.metrics, metric, None)
            
            if score is None:
                continue
            
            if metric in ['rmse', 'mae', 'mape']:
                if score < best_score:
                    best_score = score
                    best_performance = latest_performance
            else:  # r2, precision, recall, f1, auc
                if score > best_score:
                    best_score = score
                    best_performance = latest_performance
        
        return best_performance
    
    def export_evaluation_summary(self, save_path: Path):
        """Export a summary of all evaluations."""
        summary = {
            'evaluation_summary': {
                'total_models': len(self.evaluation_history),
                'total_evaluations': sum(len(perfs) for perfs in self.evaluation_history.values()),
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'model_performances': {}
        }
        
        for model_name, performance_list in self.evaluation_history.items():
            if not performance_list:
                continue
            
            latest_performance = max(performance_list, key=lambda p: p.evaluation_timestamp)
            
            summary['model_performances'][model_name] = {
                'latest_evaluation': latest_performance.to_dict(),
                'total_evaluations': len(performance_list),
                'performance_trend': {
                    'rmse_trend': [p.metrics.rmse for p in performance_list],
                    'r2_trend': [p.metrics.r2 for p in performance_list],
                    'timestamps': [p.evaluation_timestamp.isoformat() for p in performance_list]
                }
            }
        
        # Save summary
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, default=str, indent=2)
        
        self.logger.info(f"Evaluation summary exported to {save_path}")


# Convenience functions
def quick_evaluate(model_name: str, X: pd.DataFrame, y: pd.Series) -> ModelPerformance:
    """Quick evaluation of a model."""
    evaluator = ModelEvaluator(config.dict())
    return evaluator.evaluate_model(model_name, X, y)


def compare_models_quick(model_names: List[str], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Quick comparison of multiple models."""
    evaluator = ModelEvaluator(config.dict())
    return evaluator.compare_models(model_names, X, y)
