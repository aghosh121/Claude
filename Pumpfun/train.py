#!/usr/bin/env python3
"""
Training script for the Pump.fun token ranking system.

This script demonstrates how to:
1. Load and prepare historical token data
2. Build features using the feature engineering pipeline
3. Generate training labels
4. Train multiple ML models
5. Evaluate model performance
6. Save trained models
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

import numpy as np
import pandas as pd
from loguru import logger

from pumpfun.config import config
from pumpfun.data import DataConnector, DataStorage
from pumpfun.features import FeatureStore
from pumpfun.labels import LabelBuilder
from pumpfun.models import TokenRankingModel


class PumpfunTrainer:
    """Main training orchestrator for the Pump.fun system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_connector = DataConnector(config)
        self.data_storage = DataStorage(config)
        self.feature_store = FeatureStore(config)
        self.label_builder = LabelBuilder(config)
        self.ranking_model = TokenRankingModel(config)
        
        # Training state
        self.training_data = None
        self.features = None
        self.labels = None
        self.training_results = {}
        
    async def prepare_training_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Prepare training data for the specified date range."""
        logger.info(f"Preparing training data from {start_date} to {end_date}")
        
        # Get historical data
        raw_data = await self.data_connector.get_historical_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if raw_data.empty:
            raise ValueError("No historical data found for the specified date range")
            
        logger.info(f"Loaded {len(raw_data)} data points")
        self.training_data = raw_data
        
        return raw_data
        
    async def build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build features from raw training data."""
        logger.info("Building features from training data")
        
        # Compute features using the feature store
        features = await self.feature_store.compute_features_batch(data)
        
        if features.empty:
            raise ValueError("Failed to compute features from training data")
            
        logger.info(f"Built {len(features.columns)} features for {len(features)} samples")
        self.features = features
        
        return features
        
    def build_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build training labels from historical data."""
        logger.info("Building training labels")
        
        # Build all label types
        labels = self.label_builder.build_all_labels(data)
        
        if labels.empty:
            raise ValueError("Failed to build training labels")
            
        logger.info(f"Built {len(labels.columns)} labels for {len(labels)} samples")
        self.labels = labels
        
        return labels
        
    def prepare_training_dataset(self, features: pd.DataFrame, labels: pd.DataFrame) -> tuple:
        """Prepare the final training dataset by aligning features and labels."""
        logger.info("Preparing training dataset")
        
        # Ensure features and labels have the same index
        common_index = features.index.intersection(labels.index)
        
        if len(common_index) == 0:
            raise ValueError("No common samples between features and labels")
            
        # Align features and labels
        aligned_features = features.loc[common_index]
        aligned_labels = labels.loc[common_index]
        
        # Remove any remaining missing values
        valid_mask = ~(aligned_features.isnull().any(axis=1) | aligned_labels.isnull().any(axis=1))
        
        final_features = aligned_features[valid_mask]
        final_labels = aligned_labels[valid_mask]
        
        logger.info(f"Final training dataset: {len(final_features)} samples, {len(final_features.columns)} features")
        
        return final_features, final_labels
        
    def train_models(self, features: pd.DataFrame, labels: pd.DataFrame, 
                    target_column: str = "return_1h_continuous") -> Dict[str, Any]:
        """Train multiple models on the prepared dataset."""
        logger.info(f"Training models for target: {target_column}")
        
        # Prepare target variable
        if target_column not in labels.columns:
            raise ValueError(f"Target column {target_column} not found in labels")
            
        y = labels[target_column]
        X = features
        
        # Remove samples with missing target values
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training on {len(X)} samples with {len(X.columns)} features")
        
        # Define model configurations
        model_configs = [
            {
                'name': 'glm_ridge',
                'type': 'glm',
                'params': {
                    'task_type': 'regression',
                    'alpha': 1.0
                }
            },
            {
                'name': 'xgboost_reg',
                'type': 'xgboost',
                'params': {
                    'task': 'regression',
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            {
                'name': 'lightgbm_reg',
                'type': 'lightgbm',
                'params': {
                    'task': 'regression',
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            },
            {
                'name': 'catboost_reg',
                'type': 'catboost',
                'params': {
                    'task': 'regression',
                    'iterations': 200,
                    'depth': 8,
                    'learning_rate': 0.1,
                    'random_seed': 42
                }
            }
        ]
        
        # Train models
        trained_models = self.ranking_model.train_models(X, y, model_configs)
        
        # Store training results
        self.training_results = {
            'target_column': target_column,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'models_trained': list(trained_models.keys()),
            'model_performance': self.ranking_model.model_performance,
            'feature_names': list(X.columns)
        }
        
        logger.info(f"Training completed. {len(trained_models)} models trained successfully")
        return self.training_results
        
    def evaluate_models(self, features: pd.DataFrame, labels: pd.DataFrame,
                       target_column: str = "return_1h_continuous") -> Dict[str, Any]:
        """Evaluate trained models on validation data."""
        logger.info("Evaluating trained models")
        
        # Prepare validation data
        y_val = labels[target_column]
        X_val = features
        
        # Remove samples with missing target values
        valid_mask = ~y_val.isnull()
        X_val = X_val[valid_mask]
        y_val = y_val[valid_mask]
        
        # Make predictions with all models
        predictions = self.ranking_model.predict(X_val)
        
        # Calculate evaluation metrics
        evaluation_results = {}
        
        for model_name, pred in predictions.items():
            # Basic regression metrics
            mse = np.mean((y_val - pred) ** 2)
            mae = np.mean(np.abs(y_val - pred))
            r2 = 1 - (np.sum((y_val - pred) ** 2) / np.sum((y_val - y_val.mean()) ** 2))
            
            evaluation_results[model_name] = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'rmse': float(np.sqrt(mse))
            }
            
        # Store evaluation results
        self.training_results['evaluation'] = evaluation_results
        
        logger.info("Model evaluation completed")
        return evaluation_results
        
    def save_training_results(self, save_dir: Path) -> None:
        """Save all training results and models."""
        logger.info(f"Saving training results to {save_dir}")
        
        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.ranking_model.save_models(save_dir)
        
        # Save training results
        results_path = save_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
            
        # Save feature names
        if self.features is not None:
            feature_names_path = save_dir / "feature_names.txt"
            with open(feature_names_path, 'w') as f:
                for feature in self.features.columns:
                    f.write(f"{feature}\n")
                    
        # Save sample features and labels for reference
        if self.features is not None and self.labels is not None:
            sample_data = pd.concat([
                self.features.head(100),
                self.labels.head(100)
            ], axis=1)
            sample_data.to_parquet(save_dir / "sample_training_data.parquet")
            
        logger.info("Training results saved successfully")
        
    def print_training_summary(self) -> None:
        """Print a summary of the training results."""
        if not self.training_results:
            logger.warning("No training results available")
            return
            
        print("\n" + "="*60)
        print("PUMP.FUN TRAINING SUMMARY")
        print("="*60)
        
        print(f"Target Column: {self.training_results.get('target_column', 'N/A')}")
        print(f"Training Samples: {self.training_results.get('n_samples', 'N/A'):,}")
        print(f"Features: {self.training_results.get('n_features', 'N/A'):,}")
        print(f"Models Trained: {len(self.training_results.get('models_trained', []))}")
        
        print("\nModel Performance:")
        performance = self.training_results.get('model_performance', {})
        for model_name, metrics in performance.items():
            cv_score = metrics.get('cv_mean', 0)
            cv_std = metrics.get('cv_std', 0)
            print(f"  {model_name}: {cv_score:.4f} ± {cv_std:.4f}")
            
        if 'evaluation' in self.training_results:
            print("\nValidation Results:")
            evaluation = self.training_results['evaluation']
            for model_name, metrics in evaluation.items():
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                print(f"  {model_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
                
        print("="*60)


async def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pump.fun token ranking models")
    parser.add_argument("--start-date", required=True, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--target", default="return_1h_continuous", help="Target column for training")
    parser.add_argument("--save-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/training.log", rotation="1 day", retention="7 days")
    
    try:
        # Initialize trainer
        trainer = PumpfunTrainer(config.dict())
        
        # Prepare training data
        raw_data = await trainer.prepare_training_data(args.start_date, args.end_date)
        
        # Build features
        features = await trainer.build_features(raw_data)
        
        # Build labels
        labels = trainer.build_labels(raw_data)
        
        # Prepare final dataset
        X, y = trainer.prepare_training_dataset(features, labels)
        
        # Train models
        training_results = trainer.train_models(X, y, args.target)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X, y, args.target)
        
        # Save results
        save_dir = Path(args.save_dir)
        trainer.save_training_results(save_dir)
        
        # Print summary
        trainer.print_training_summary()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
