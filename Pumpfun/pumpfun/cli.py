"""
Command-line interface for the Pump.fun token ranking system.

This module provides a comprehensive CLI for interacting with all system
components including data management, model training, evaluation, and serving.
"""

import click
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import asyncio
import sys

from .config import config
from .data.connector import DataConnector
from .data.storage import DataStorage
from .data.snapshotter import DataSnapshotter
from .features.engineers import FeatureEngineer
from .features.store import FeatureStore
from .models import ModelRegistry, TokenRankingModel
from .evaluation import ModelEvaluator
from .labels import LabelBuilder
from .serve import ModelServer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Path to config file')
def cli(verbose, config_file):
    """Pump.fun Token Ranking System CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config_file:
        # Load custom config
        click.echo(f"Loading config from {config_file}")
    
    click.echo("🚀 Pump.fun Token Ranking System")


@cli.group()
def data():
    """Data management commands"""
    pass


@data.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--limit', '-l', type=int, default=1000, help='Number of records to fetch')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'json', 'parquet']), default='csv')
def fetch(source, limit, output, output_format):
    """Fetch data from a source"""
    try:
        import asyncio
        
        async def _fetch_data():
            connector = DataConnector(config.model_dump())
            async with connector as conn:
                if source == 'pumpfun':
                    # Get historical data for testing
                    data = await conn.get_historical_data(
                        start_date="2024-01-01",
                        end_date="2024-01-02",
                        token_limit=min(limit // 25, 10)  # Estimate tokens needed
                    )
                else:
                    click.echo(f"Unknown source: {source}")
                    return None
                return data
        
        # Run async function
        data = asyncio.run(_fetch_data())
        
        if data is None or data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        click.echo(f"Fetched {len(data)} records from {source}")
        
        if output:
            if output_format == 'csv':
                data.to_csv(output, index=False)
            elif output_format == 'json':
                data.to_json(output, orient='records', indent=2)
            elif output_format == 'parquet':
                data.to_parquet(output, index=False)
            click.echo(f"Data saved to {output}")
        else:
            # Display sample
            click.echo("\nSample data:")
            click.echo(data.head().to_string())
            
    except Exception as e:
        click.echo(f"Error fetching data: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
@click.option('--compression', type=click.Choice(['snappy', 'gzip', 'brotli']), default='snappy')
def store(source, storage_type, compression):
    """Store data from a source"""
    try:
        # Fetch and store data
        import asyncio
        
        async def _fetch_and_store():
            connector = DataConnector(config.model_dump())
            storage = DataStorage(config.model_dump())
            
            async with connector as conn:
                if source == 'pumpfun':
                    # Get historical data for testing
                    data = await conn.get_historical_data(
                        start_date="2024-01-01",
                        end_date="2024-01-02",
                        token_limit=5
                    )
                else:
                    click.echo(f"Unknown source: {source}")
                    return None, None
                return data, storage
        
        # Run async function
        data, storage = asyncio.run(_fetch_and_store())
        
        if data is None or data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        metadata = storage.store_token_data(source, data, data_type=storage_type)
        
        click.echo(f"Stored {len(data)} records from {source}")
        if metadata:
            click.echo(f"Data stored successfully")
        else:
            click.echo(f"Data stored but no metadata returned")
        
    except Exception as e:
        click.echo(f"Error storing data: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--source', '-s', help='Data source name (optional)')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
def list_sources(source, storage_type):
    """List available data sources"""
    try:
        storage = DataStorage(config.model_dump())
        
        if source:
            info = storage.get_data_info(source, storage_type)
            if info:
                click.echo(f"\nData source: {source}")
                click.echo(f"Total files: {info['total_files']}")
                click.echo(f"Total size: {info['total_size_gb']:.2f} GB")
                click.echo(f"Total records: {info['total_records']}")
                click.echo(f"Latest: {info['latest_timestamp']}")
                click.echo(f"Oldest: {info['oldest_timestamp']}")
                
                click.echo("\nFile details:")
                for file_info in info['file_info'][:5]:  # Show first 5 files
                    click.echo(f"  {file_info['filename']} - {file_info['size_bytes']} bytes - {file_info['age_hours']:.1f} hours old")
            else:
                click.echo(f"No data found for source: {source}")
        else:
            # Use the available methods from DataStorage
            try:
                # Get storage stats to show available data
                stats = storage.get_storage_stats()
                if stats:
                    click.echo(f"Storage statistics for {storage_type} data:")
                    click.echo(f"  Total size: {stats.get('total_size_gb', 0):.2f} GB")
                    click.echo(f"  Total files: {stats.get('total_files', 0)}")
                    click.echo(f"  Data types: {', '.join(stats.get('data_types', []))}")
                else:
                    click.echo(f"No {storage_type} data found")
            except Exception as e:
                click.echo(f"Could not retrieve storage stats: {e}")
                click.echo("Try fetching some data first with: pumpfun data fetch <source>")
                
    except Exception as e:
        click.echo(f"Error listing sources: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--max-age-hours', type=int, default=168, help='Maximum age in hours to keep')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without actually deleting')
def cleanup(max_age_hours, storage_type, dry_run):
    """Clean up old data files"""
    try:
        storage = DataStorage(config.model_dump())
        
        if dry_run:
            click.echo(f"DRY RUN: Would clean up data older than {max_age_hours} hours")
            # This would need to be implemented in the storage class
            click.echo("Dry run cleanup not yet implemented")
        else:
            result = storage.cleanup_old_data(max_age_hours, storage_type)
            click.echo(f"Cleanup completed: {result['files_removed']} files removed")
            click.echo(f"Freed space: {result['bytes_freed'] / (1024**3):.2f} GB")
            
    except Exception as e:
        click.echo(f"Error during cleanup: {e}", err=True)
        sys.exit(1)


@cli.group()
def features():
    """Feature engineering commands"""
    pass


@features.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--storage-type', type=click.Choice(['raw', 'processed']), default='raw')
def engineer(source, output, storage_type):
    """Generate features from data"""
    try:
        # Load data
        storage = DataStorage(config.model_dump())
        
        # Use the correct method to retrieve data
        try:
            # Try to get data for the source
            data = storage.retrieve_token_data(source, data_type=storage_type)
            if data.empty:
                click.echo(f"No data found for source: {source}")
                return
        except Exception as e:
            click.echo(f"Error loading data: {e}")
            return
        
        # Engineer features
        feature_engineer = FeatureEngineer(config)
        
        # Process each token's data to generate features
        all_features = []
        unique_tokens = data['token_id'].unique()  # Use token_id instead of token_address
        
        for token_id in unique_tokens[:5]:  # Limit to first 5 tokens for testing
            token_data = data[data['token_id'] == token_id]
            if not token_data.empty:
                # Use the first timestamp for this token
                timestamp = token_data['timestamp'].iloc[0]
                try:
                    features = feature_engineer.compute_features(token_data, token_id, timestamp)
                    if not features.empty:
                        all_features.append(features)
                except Exception as e:
                    click.echo(f"Warning: Failed to compute features for token {token_id}: {e}")
                    continue
        
        if all_features:
            features = pd.concat(all_features, ignore_index=True)
            click.echo(f"Generated {len(features)} feature records from {len(data)} data records")
            click.echo(f"Feature columns: {', '.join(features.columns)}")
            
            if output:
                features.to_parquet(output, index=False)
                click.echo(f"Features saved to {output}")
            else:
                # Store features using FeatureStore instead of DataStorage
                feature_store = FeatureStore()
                
                # Create metadata for the features
                from pumpfun.features.store import FeatureMetadata
                from datetime import datetime
                
                metadata = FeatureMetadata(
                    token_id=source,
                    timestamp=datetime.now(),
                    feature_count=len(features.columns),
                    feature_names=list(features.columns),
                    feature_families=list(feature_engineer.engineers.keys()),
                    computation_time=0.0,  # We don't track computation time in CLI
                    data_sources=[source],
                    version="1.0"
                )
                
                # Store features using FeatureStore
                feature_file = feature_store.store_features(source, features, metadata)
                click.echo(f"Features stored in feature store: {feature_file}")
        else:
            click.echo("No features could be generated from the data")
            
    except Exception as e:
        click.echo(f"Error engineering features: {e}", err=True)
        sys.exit(1)


@features.command()
@click.option('--source', '-s', help='Data source name (optional)')
def list_features(source):
    """List available features"""
    try:
        # FeatureStore doesn't need config parameter - it uses global config
        feature_store = FeatureStore()
        
        if source:
            features = feature_store.load_features(source)
            if not features.empty:
                click.echo(f"Features for {source}:")
                click.echo(features.head().to_string())
            else:
                click.echo(f"No features found for {source}")
        else:
            # Get all available tokens with features
            tokens = feature_store.get_all_tokens()
            if tokens:
                click.echo("Available feature sources:")
                for token_id in tokens:
                    summary = feature_store.get_feature_summary(token_id)
                    if summary:
                        latest = summary['latest_features']
                        click.echo(f"  {token_id}: {latest['feature_count']} features, {len(latest['feature_families'])} families")
            else:
                click.echo("No feature sources found")
                
    except Exception as e:
        click.echo(f"Error listing features: {e}", err=True)
        sys.exit(1)


@cli.group()
def models():
    """Model management commands"""
    pass


@models.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility')
def train(model, source, target, test_size, random_state):
    """Train a model"""
    try:
        # Load features and labels
        storage = DataStorage(config.model_dump())
        feature_store = FeatureStore()
        
        # Get features for the source
        features = feature_store.load_features(source)
        if features is None or features.empty:
            click.echo(f"No features found for source: {source}")
            return
            
        # Get labels from processed storage
        labels = storage.retrieve_token_data(f"{source}_labels", "processed")
        if labels.empty:
            click.echo(f"No labels found for source: {source}")
            return
            
        # Combine features and labels
        data = features.merge(labels, on=['token_id', 'timestamp'], how='inner')
        
        if target not in data.columns:
            click.echo(f"Target column '{target}' not found in data. Available columns: {list(data.columns)}")
            return
        
        # Prepare features and target
        # Drop non-feature columns (keep only feature columns and target)
        feature_columns = [col for col in features.columns if col not in ['token_id', 'timestamp']]
        X = data[feature_columns]
        y = data[target]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Debug: Check what we have
        click.echo(f"Features shape: {features.shape}")
        click.echo(f"Labels shape: {labels.shape}")
        click.echo(f"Features columns: {list(features.columns)}")
        click.echo(f"Labels columns: {list(labels.columns)}")
        click.echo(f"Features token_id sample: {features['token_id'].iloc[0] if not features.empty else 'None'}")
        click.echo(f"Labels token_id sample: {labels['token_id'].iloc[0] if not labels.empty else 'None'}")
        click.echo(f"Features timestamp sample: {features['timestamp'].iloc[0] if not features.empty else 'None'}")
        click.echo(f"Labels timestamp sample: {labels['timestamp'].iloc[0] if not labels.empty else 'None'}")
        click.echo(f"Merged data shape: {data.shape}")
        
        if data.empty:
            click.echo("Warning: No data after merge. Trying to debug...")
            # Check if token_ids match
            feature_tokens = set(features['token_id'].unique())
            label_tokens = set(labels['token_id'].unique())
            click.echo(f"Feature tokens: {feature_tokens}")
            click.echo(f"Label tokens: {label_tokens}")
            
            # Check if timestamps match
            feature_times = set(features['timestamp'].unique())
            label_times = set(labels['timestamp'].unique())
            click.echo(f"Feature timestamps: {len(feature_times)} unique times")
            click.echo(f"Label timestamps: {len(label_times)} unique times")
            
            # Try merge on just token_id
            data = features.merge(labels, on=['token_id'], how='inner')
            click.echo(f"Merge on token_id only: {data.shape}")
            
            if data.empty:
                click.echo("Still no data. Exiting.")
                return
        
        click.echo(f"Training {model} with {len(X)} samples and {len(X.columns)} features")
        
        # Train model using TokenRankingModel
        token_ranking = TokenRankingModel(config.model_dump())
        
        # Create model configuration
        model_config = {
            'name': model,
            'type': 'lightgbm' if 'lightgbm' in model else 'glm',
            'params': {'task': 'regression'}
        }
        
        # Train the model
        trained_models = token_ranking.train_models(X, y, [model_config])
        trained_model = trained_models[model]
        
        click.echo(f"Model {model} trained successfully")
        
        # Save model
        save_dir = Path(config.model.model_registry_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        token_ranking.save_models(save_dir)
        click.echo(f"Model saved to {save_dir}")
        
    except Exception as e:
        click.echo(f"Error training model: {e}", err=True)
        sys.exit(1)


@models.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility')
def evaluate(model, source, target, test_size, random_state):
    """Evaluate a model"""
    try:
        # Load features and labels
        storage = DataStorage(config.model_dump())
        feature_store = FeatureStore()
        
        # Get features for the source
        features = feature_store.load_features(source)
        if features is None or features.empty:
            click.echo(f"No features found for source: {source}")
            return
            
        # Get labels from processed storage
        labels = storage.retrieve_token_data(f"{source}_labels", "processed")
        if labels.empty:
            click.echo(f"No labels found for source: {source}")
            return
            
        # Combine features and labels
        data = features.merge(labels, on=['token_id', 'timestamp'], how='inner')
        
        if target not in data.columns:
            click.echo(f"Target column '{target}' not found in data. Available columns: {list(data.columns)}")
            return
        
        # Prepare features and target
        # Drop non-feature columns (keep only feature columns and target)
        feature_columns = [col for col in features.columns if col not in ['token_id', 'timestamp']]
        X = data[feature_columns]
        y = data[target]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        click.echo(f"Evaluating {model} with {len(X)} samples and {len(X.columns)} features")
        
        # Evaluate model
        evaluator = ModelEvaluator(config)  # Pass the actual config object, not the dictionary
        performance = evaluator.evaluate_model(model, X, y, test_size, random_state)
        
        click.echo(f"\nEvaluation Results for {model}:")
        click.echo(f"RMSE: {performance.metrics.rmse:.4f}")
        click.echo(f"MAE: {performance.metrics.mae:.4f}")
        click.echo(f"R² Score: {performance.metrics.r2:.4f}")
        click.echo(f"MAPE: {performance.metrics.mape:.2f}%")
        click.echo(f"Training Time: {performance.training_time:.2f} seconds")
        click.echo(f"Prediction Time: {performance.prediction_time:.4f} seconds")
        click.echo(f"Model Size: {performance.model_size_mb:.2f} MB")
        
    except Exception as e:
        click.echo(f"Error evaluating model: {e}", err=True)
        sys.exit(1)


@models.command()
@click.option('--models', '-m', required=True, help='Comma-separated list of model names')
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', type=float, default=0.2, help='Test set size')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility')
def compare(models, source, target, test_size, random_state):
    """Compare multiple models"""
    try:
        model_names = [m.strip() for m in models.split(',')]
        
        # Load features and labels
        storage = DataStorage(config.model_dump())
        feature_store = FeatureStore()
        
        # Get features for the source
        features = feature_store.load_features(source)
        if features is None or features.empty:
            click.echo(f"No features found for source: {source}")
            return
            
        # Get labels from processed storage
        labels = storage.retrieve_token_data(f"{source}_labels", "processed")
        if labels.empty:
            click.echo(f"No labels found for source: {source}")
            return
            
        # Combine features and labels
        data = features.merge(labels, on=['token_id', 'timestamp'], how='inner')
        
        if target not in data.columns:
            click.echo(f"Target column '{target}' not found in data. Available columns: {list(data.columns)}")
            return
        
        # Prepare features and target
        # Drop non-feature columns (keep only feature columns and target)
        feature_columns = [col for col in features.columns if col not in ['token_id', 'timestamp']]
        X = data[feature_columns]
        y = data[target]
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        click.echo(f"Comparing {len(model_names)} models with {len(X)} samples and {len(X.columns)} features")
        
        # Compare models
        evaluator = ModelEvaluator(config.model_dump())
        comparison = evaluator.compare_models(model_names, X, y, test_size, random_state)
        
        click.echo("\nModel Comparison Results:")
        click.echo(comparison.to_string(index=False))
        
        # Find best model
        best_model = comparison.iloc[0]
        click.echo(f"\nBest performing model: {best_model['model_name']} (RMSE: {best_model['rmse']:.4f})")
        
    except Exception as e:
        click.echo(f"Error comparing models: {e}", err=True)
        sys.exit(1)


@models.command()
def list_models():
    """List available models"""
    try:
        model_registry = ModelRegistry(config.model_dump())
        available_models = model_registry.list_models()
        
        if available_models:
            click.echo("Available models:")
            for model_name in available_models:
                click.echo(f"  {model_name}")
        else:
            click.echo("No models available")
            
    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)
        sys.exit(1)


@cli.group()
def labels():
    """Label building commands"""
    pass


@labels.command()
@click.option('--source', '-s', required=True, help='Data source name')
@click.option('--method', '-m', type=click.Choice(['future_return', 'ranking', 'classification']), default='future_return')
@click.option('--horizon', type=int, default=24, help='Prediction horizon in hours')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def build_labels(source, method, horizon, output):
    """Build labels for training"""
    try:
        # Load data - use retrieve_token_data for the specific source
        storage = DataStorage(config.model_dump())
        
        # Get all available tokens for this source
        available_tokens = storage.list_available_tokens("raw")
        if not available_tokens:
            click.echo(f"No data found for source: {source}")
            return
            
        # Load data for the first available token (or we could iterate through all)
        token_id = available_tokens[0]
        data = storage.retrieve_token_data(token_id, "raw")
        
        if data.empty:
            click.echo(f"No data found for source: {source}")
            return
        
        click.echo(f"Building {method} labels for {len(data)} records with {horizon}h horizon")
        
        # Build labels
        label_builder = LabelBuilder(config.model_dump())
        labels = label_builder.build_all_labels(data)
        
        click.echo(f"Generated {len(labels)} label records")
        
        if output:
            labels.to_parquet(output, index=False)
            click.echo(f"Labels saved to {output}")
        else:
            # Store in processed storage using store_token_data
            storage.store_token_data(f"{source}_labels", labels, "processed")
            click.echo(f"Labels stored in processed storage")
            
    except Exception as e:
        click.echo(f"Error building labels: {e}", err=True)
        sys.exit(1)


@cli.group()
def serve():
    """Model serving commands"""
    pass


@serve.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def start(host, port, workers, reload):
    """Start the model serving API"""
    try:
        click.echo(f"Starting Pump.fun API server on {host}:{port}")
        
        # Use RankingService instead of ModelServer for the API server
        from .serve import create_ranking_service
        server = create_ranking_service(config.model_dump())
        
        if reload:
            click.echo("Auto-reload enabled")
            # This would need to be implemented in the server class
            click.echo("Auto-reload not yet implemented")
        
        # Start server
        server.start(host=host, port=port, workers=workers)
        
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        sys.exit(1)


@serve.command()
@click.option('--model', '-m', required=True, help='Model name')
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input data file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def predict(model, input, output):
    """Make predictions using a trained model"""
    try:
        # Load input data
        if input.endswith('.csv'):
            data = pd.read_csv(input)
        elif input.endswith('.parquet'):
            data = pd.read_parquet(input)
        else:
            click.echo("Input file must be CSV or Parquet")
            return
        
        click.echo(f"Making predictions with {model} on {len(data)} records")
        
        # Load model first to get the expected feature names
        model_registry = ModelRegistry(config.model_dump())
        trained_model = model_registry.get_model(model)
        
        if trained_model is None:
            click.echo(f"Model {model} not found")
            return
        
        # Get the feature names the model was trained with
        expected_features = trained_model.feature_names
        click.echo(f"Model expects {len(expected_features)} features")
        
        # Prepare features - use only the features the model was trained with
        available_features = [col for col in data.columns if col in expected_features]
        missing_features = [col for col in expected_features if col not in data.columns]
        
        if missing_features:
            click.echo(f"Warning: Missing features: {missing_features[:5]}...")
        
        X = data[available_features]
        click.echo(f"Using {len(X.columns)} available features for prediction")
        
        # Make predictions
        predictions = trained_model.predict(X)
        
        # Add predictions to data
        result_data = data.copy()
        result_data['prediction'] = predictions
        
        click.echo(f"Predictions completed")
        
        if output:
            if output.endswith('.csv'):
                result_data.to_csv(output, index=False)
            elif output.endswith('.parquet'):
                result_data.to_parquet(output, index=False)
            else:
                result_data.to_csv(output, index=False)
            click.echo(f"Results saved to {output}")
        else:
            # Display sample
            click.echo("\nSample predictions:")
            click.echo(result_data[['prediction']].head().to_string())
            
    except Exception as e:
        click.echo(f"Error making predictions: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status"""
    try:
        click.echo("🔍 Pump.fun System Status")
        click.echo("=" * 40)
        
        # Data storage status
        storage = DataStorage(config.model_dump())
        stats = storage.get_storage_stats()
        
        click.echo(f"📊 Data Storage:")
        # Use the correct keys from get_storage_stats()
        total_size_gb = stats.get('total_size_mb', 0) / 1024  # Convert MB to GB
        click.echo(f"  Total size: {total_size_gb:.2f} GB")
        
        # Sum up file counts across all data types
        total_files = sum(stats.get('file_counts', {}).values())
        click.echo(f"  Total files: {total_files}")
        
        # Show data types and their counts
        data_types = list(stats.get('file_counts', {}).keys())
        if data_types:
            click.echo(f"  Data types: {', '.join(data_types)}")
            for data_type in data_types:
                file_count = stats.get('file_counts', {}).get(data_type, 0)
                token_count = stats.get('token_counts', {}).get(data_type, 0)
                click.echo(f"    {data_type}: {file_count} files, {token_count} tokens")
        else:
            click.echo("  No data found")
        
        # Model status
        model_registry = ModelRegistry(config.model_dump())
        available_models = model_registry.list_models()
        
        click.echo(f"\n🤖 Models:")
        click.echo(f"  Available: {len(available_models)}")
        for model_name in available_models:
            click.echo(f"    - {model_name}")
        
        # Feature status
        feature_store = FeatureStore()
        feature_tokens = feature_store.get_all_tokens()
        
        click.echo(f"\n⚡ Features:")
        click.echo(f"  Tokens with features: {len(feature_tokens)}")
        for token_id in feature_tokens[:5]:  # Show first 5
            click.echo(f"    - {token_id}")
        if len(feature_tokens) > 5:
            click.echo(f"    ... and {len(feature_tokens) - 5} more")
        
        click.echo("\n✅ System is running")
        
    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export_config(output):
    """Export current configuration"""
    try:
        config_dict = config.model_dump()
        
        if output:
            with open(output, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            click.echo(f"Configuration exported to {output}")
        else:
            click.echo(json.dumps(config_dict, indent=2, default=str))
            
    except Exception as e:
        click.echo(f"Error exporting config: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
